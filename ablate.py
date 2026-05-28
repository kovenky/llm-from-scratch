"""ablate.py — Ablation runner for tiny-gpt on TinyStories.

Runs a series of single-variable ablations against a fixed baseline config.
Each ablation trains a fresh model with the same seed (1337), records the
loss curve, generation samples, ability metrics, wallclock, and peak memory,
and appends one JSON line per ablation to a single JSONL file.

Resumable: if the JSONL already contains a result for an ablation name,
that ablation is skipped. Delete the line (or the file) to re-run.

Usage:
    python ablate.py                          # run everything
    python ablate.py --list                   # list ablations, don't run
    python ablate.py --only baseline,lr_1e-3  # run a subset
    python ablate.py --skip sgd_momentum      # skip specific ones
    python ablate.py --out results.jsonl      # custom output path

Skipped (need external code/deps, see SKIPPED_NOTES at bottom):
    muon, fp8, bpe, curriculum_blocksize
"""
import argparse
import datetime
import json
import math
import os
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config (baseline values match tiny-gpt.py small() preset)
# ============================================================
@dataclass
class Config:
    # Architecture
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 256
    vocab_size: int = 0           # filled from data

    # Training
    batch_size: int = 64
    max_iters: int = 4000
    eval_interval: int = 500
    eval_iters: int = 50
    lr: float = 3e-3
    min_lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_iters: int = 100

    # Data
    dataset: str = "tinystories"
    data_mb: int = 50

    # Numerics
    use_compile: bool = True
    compile_mode: str = "default"
    dtype: str = "bfloat16"
    use_autocast: bool = True
    seed: int = 1337

    # Variant flags (defaults reproduce the baseline architecture)
    norm: str = "layernorm"          # "layernorm" | "rmsnorm"
    activation: str = "gelu"         # "gelu" | "relu2" | "swiglu"
    pos_encoding: str = "learned"    # "learned" | "rope" | "none"
    tie_embeddings: bool = True
    dropout: float = 0.0
    post_norm: bool = False
    qk_norm: bool = False
    optimizer: str = "adamw"         # "adamw" | "sgd"
    init_scheme: str = "scaled"      # "scaled" | "default" | "embd_std_0.1"
    decoupled_decay: bool = False    # no WD on 1D params & embeddings


# ============================================================
# Data (mirrors tiny-gpt.py — kept here so runner is standalone)
# ============================================================
from pathlib import Path


def load_text(cfg: Config) -> str:
    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir / f"tinystories_{cfg.data_mb}mb.txt"
    if not path.exists():
        print(f"downloading TinyStories (~{cfg.data_mb}MB target)...")
        from datasets import load_dataset
        pct = max(1, int(cfg.data_mb / 25) + 1)
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{pct}%]")
        text = "\n\n".join(ds["text"])
        text = text.encode("ascii", errors="ignore").decode("ascii")
        text = text[: cfg.data_mb * 1024 * 1024]
        path.write_text(text)
    return path.read_text()


def encode_text(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    lut = np.zeros(256, dtype=np.int64)
    for c, i in stoi.items():
        lut[ord(c)] = i
    arr = lut[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]
    return torch.from_numpy(arr), stoi, itos, len(chars)


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), device=data.device)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ============================================================
# Model components with variant toggles
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def make_norm(cfg: Config, dim: int):
    if cfg.norm == "layernorm":
        return nn.LayerNorm(dim, bias=False)
    if cfg.norm == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"unknown norm: {cfg.norm}")


def build_rope_cache(block_size, head_dim, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(block_size, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.sin(), emb.cos()


def apply_rope(x, sin, cos):
    # x: (B, H, T, D); sin/cos: (T, D) broadcasted
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos[..., :d // 2] - x2 * sin[..., :d // 2],
                      x1 * sin[..., d // 2:] + x2 * cos[..., d // 2:]], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.use_rope = (cfg.pos_encoding == "rope")
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, rope=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope and rope is not None:
            sin, cos = rope
            sin = sin[:T].unsqueeze(0).unsqueeze(0)
            cos = cos[:T].unsqueeze(0).unsqueeze(0)
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.activation = cfg.activation
        if cfg.activation == "swiglu":
            hidden = int(8 * cfg.n_embd / 3)
            hidden = ((hidden + 7) // 8) * 8  # round to multiple of 8
            self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
            self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        else:
            self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
            self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        if self.activation == "swiglu":
            return self.c_proj(F.silu(self.w_gate(x)) * self.w_up(x))
        if self.activation == "relu2":
            return self.c_proj(F.relu(self.c_fc(x)) ** 2)
        return self.c_proj(F.gelu(self.c_fc(x)))  # gelu


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = make_norm(cfg, cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = make_norm(cfg, cfg.n_embd)
        self.mlp = MLP(cfg)
        self.post_norm = cfg.post_norm
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x, rope=None):
        if self.post_norm:
            x = self.ln1(x + self.drop(self.attn(x, rope=rope)))
            x = self.ln2(x + self.drop(self.mlp(x)))
        else:
            x = x + self.drop(self.attn(self.ln1(x), rope=rope))
            x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = (nn.Embedding(cfg.block_size, cfg.n_embd)
                        if cfg.pos_encoding == "learned" else None)
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = make_norm(cfg, cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self._init_weights()
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self._rope_cache = None

    def _init_weights(self):
        cfg = self.cfg
        if cfg.init_scheme == "default":
            return  # PyTorch defaults — for ablation #23
        emb_std = 0.1 if cfg.init_scheme == "embd_std_0.1" else 0.02
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=emb_std)
        # scaled residual-stream output projection init
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def get_rope(self, device):
        if self.cfg.pos_encoding != "rope":
            return None
        if self._rope_cache is None or self._rope_cache[0].device != device:
            head_dim = self.cfg.n_embd // self.cfg.n_head
            self._rope_cache = build_rope_cache(self.cfg.block_size, head_dim, device)
        return self._rope_cache

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        rope = self.get_rope(idx.device)
        for block in self.blocks:
            x = block(x, rope=rope)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ============================================================
# Optimizer and LR schedule
# ============================================================
def make_optimizer(model, cfg: Config):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    if cfg.decoupled_decay:
        decay, nodecay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or "tok_emb" in n or "pos_emb" in n:
                nodecay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay,
    )


def get_lr(it, cfg: Config) -> float:
    if cfg.warmup_iters > 0 and it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ============================================================
# Eval and metrics
# ============================================================
def _autocast_ctx(cfg, device, autocast_dtype):
    if cfg.use_autocast and device == "cuda":
        return torch.autocast(device_type=device, dtype=autocast_dtype)
    return nullcontext()


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, autocast_dtype, device):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size)
            with _autocast_ctx(cfg, device, autocast_dtype):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def generation_metrics(model, stoi, itos, device):
    model.eval()
    prompts = ["Once upon a time", "The little girl", "One day"]
    samples = []
    for prompt in prompts:
        ids = torch.tensor([[stoi.get(c, 0) for c in prompt]],
                           dtype=torch.long, device=device)
        out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=40)
        text = "".join(itos[i] for i in out[0].tolist())
        samples.append({"prompt": prompt, "completion": text})

    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=1000, temperature=0.8, top_k=40)
    gen = "".join(itos[i] for i in out[0].tolist())
    words = gen.split()
    word_count = max(1, len(words))
    bigrams = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
    repeat_rate = 1 - len(set(bigrams)) / max(1, len(bigrams))

    model.train()
    return {
        "samples": samples,
        "chars_generated": len(gen),
        "word_count": word_count,
        "unique_words": len(set(words)),
        "lexical_diversity": len(set(words)) / word_count,
        "avg_word_length": sum(len(w) for w in words) / word_count,
        "alphabetic_ratio": sum(w.isalpha() for w in words) / word_count,
        "bigram_repeat_rate": repeat_rate,
    }


# ============================================================
# Checkpoint save/load
# ============================================================
def _save_checkpoint(model, cfg: Config, stoi, itos, vocab_size,
                     name, category, final_metrics, checkpoint_dir):
    """Save a self-contained checkpoint for later inference."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Strip torch.compile's '_orig_mod.' prefix so the dict loads cleanly into
    # an un-compiled GPT instance.
    sd = model.state_dict()
    sd = {k.replace("_orig_mod.", ""): v.detach().cpu() for k, v in sd.items()}
    path = os.path.join(checkpoint_dir, f"{name}_seed{cfg.seed}.pt")
    torch.save({
        "model_state_dict": sd,
        "config": asdict(cfg),     # asdict preserves tuples; pickle round-trips them
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
        "name": name,
        "category": category,
        "seed": cfg.seed,
        "final": final_metrics,
    }, path)
    return path


# ============================================================
# Single ablation run
# ============================================================
def run_one(name, category, description, cfg: Config,
            train_data, val_data, stoi, itos, device,
            checkpoint_dir=None):
    # Reseed at the start of each ablation so all runs share the same data order
    torch.manual_seed(cfg.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats()

    autocast_dtype = (
        torch.bfloat16
        if (cfg.dtype == "bfloat16" and device == "cuda" and cfg.use_autocast)
        else torch.float32
    )

    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_unique = n_params - (cfg.vocab_size * cfg.n_embd if cfg.tie_embeddings else 0)

    if cfg.use_compile and device == "cuda":
        try:
            model = torch.compile(model, mode=cfg.compile_mode)
        except Exception as e:
            print(f"  [warn] torch.compile failed: {e}; running eager")

    opt = make_optimizer(model, cfg)

    steps_list, train_losses, val_losses = [], [], []
    t0 = time.time()
    diverged = False
    diverge_step = None

    for step in range(cfg.max_iters + 1):
        lr = get_lr(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, autocast_dtype, device)
            steps_list.append(step)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            bpc = losses["val"] / math.log(2) if math.isfinite(losses["val"]) else float("nan")
            print(f"  step {step:5d} | lr {lr:.5f} | "
                  f"train {losses['train']:7.4f} | val {losses['val']:7.4f} | "
                  f"bpc {bpc:6.3f} | {time.time() - t0:5.1f}s")
            if not math.isfinite(losses["val"]):
                diverged = True
                diverge_step = step
                break

        if step == cfg.max_iters:
            break

        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size)
        with _autocast_ctx(cfg, device, autocast_dtype):
            _, loss = model(x, y)
        if not torch.isfinite(loss):
            diverged = True
            diverge_step = step
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    elapsed = time.time() - t0
    peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else 0.0

    final = {
        "train_loss": train_losses[-1] if train_losses else float("nan"),
        "val_loss": val_losses[-1] if val_losses else float("nan"),
        "bpc": (val_losses[-1] / math.log(2)) if val_losses and math.isfinite(val_losses[-1]) else float("nan"),
        "ppl": math.exp(val_losses[-1]) if val_losses and math.isfinite(val_losses[-1]) else float("nan"),
    }

    if diverged:
        gen = {"samples": [], "note": f"skipped (diverged at step {diverge_step})"}
    else:
        try:
            gen = generation_metrics(model, stoi, itos, device)
        except Exception as e:
            gen = {"samples": [], "note": f"gen failed: {e}"}

    # Save checkpoint (skip if requested or if the run diverged — no point
    # keeping NaN weights).
    ckpt_path = None
    if checkpoint_dir is not None and not diverged:
        try:
            ckpt_path = _save_checkpoint(model, cfg, stoi, itos, cfg.vocab_size,
                                         name, category, final, checkpoint_dir)
            print(f"  saved checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"  [warn] checkpoint save failed: {e}")

    cfg_dict = asdict(cfg)
    cfg_dict["betas"] = list(cfg_dict["betas"])  # tuples don't json cleanly

    return {
        "name": name,
        "category": category,
        "description": description,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "config": cfg_dict,
        "n_params": n_params,
        "n_params_unique": n_unique,
        "loss_curve": {"steps": steps_list,
                       "train_loss": train_losses,
                       "val_loss": val_losses},
        "final": final,
        "generation": gen,
        "wallclock_seconds": round(elapsed, 2),
        "peak_memory_mb": round(peak_mem_mb, 1),
        "diverged": diverged,
        "diverge_step": diverge_step,
        "checkpoint_path": ckpt_path,
        "error": None,
    }


# ============================================================
# Ablation registry
# ============================================================
def _baseline_cfg() -> Config:
    return Config()


def make_ablations():
    """List of (name, category, description, factory). One row per JSONL line."""
    A = []

    def add(name, category, description, **overrides):
        def factory(_o=overrides):
            c = _baseline_cfg()
            for k, v in _o.items():
                setattr(c, k, v)
            return c
        A.append({"name": name, "category": category,
                  "description": description, "factory": factory})

    # ---- Baseline ----
    add("baseline", "baseline", "as written")

    # ---- Optim ----
    add("lr_1e-4", "optim", "lr=1e-4 (expected: barely moves)",
        lr=1e-4, min_lr=1e-5)
    add("lr_1e-3", "optim", "lr=1e-3 (expected: slower than 3e-3 but stable)",
        lr=1e-3, min_lr=1e-4)
    add("lr_1e-2", "optim", "lr=1e-2 (expected: NaN / oscillation)",
        lr=1e-2, min_lr=1e-3)
    add("no_warmup", "optim", "warmup_iters=0",
        warmup_iters=0)
    add("sgd_momentum", "optim", "SGD lr=0.5 momentum=0.9 (expected: much worse)",
        optimizer="sgd", lr=0.5, min_lr=0.05)
    add("wd_zero", "optim", "weight_decay=0 (expected: train<val gap)",
        weight_decay=0.0)
    add("no_grad_clip", "optim", "grad_clip=0",
        grad_clip=0.0)
    add("decoupled_wd", "optim", "no WD on 1D params and embeddings",
        decoupled_decay=True)

    # ---- Arch-size ----
    add("n_layer_1", "arch-size", "n_layer=1",
        n_layer=1)
    add("n_layer_4", "arch-size", "n_layer=4",
        n_layer=4)
    add("n_embd_32", "arch-size", "n_embd=32",
        n_embd=32, n_head=4)
    add("wide_shallow", "arch-size", "n_embd=96, n_layer=1 (iso-param vs deep_narrow)",
        n_embd=96, n_layer=1, n_head=4)
    add("deep_narrow", "arch-size", "n_embd=48, n_layer=4 (iso-param vs wide_shallow)",
        n_embd=48, n_layer=4, n_head=4)

    # ---- Arch ----
    add("rmsnorm", "arch", "LayerNorm -> RMSNorm",
        norm="rmsnorm")
    add("post_norm", "arch", "post-norm residual",
        post_norm=True)
    add("swiglu", "arch", "SwiGLU MLP (~8/3 hidden)",
        activation="swiglu")
    add("relu2", "arch", "ReLU^2 activation",
        activation="relu2")
    add("rope", "arch", "RoPE position encoding (replaces learned)",
        pos_encoding="rope")
    add("nope", "arch", "no position encoding",
        pos_encoding="none")
    add("untied", "arch", "untied input/output embeddings",
        tie_embeddings=False)
    add("qk_norm", "arch", "RMSNorm on Q, K",
        qk_norm=True)

    # ---- Init ----
    add("default_init", "init", "PyTorch default init (skip scaled init)",
        init_scheme="default")
    add("embd_std_0.1", "init", "embedding std=0.1 (expected: diverges at lr=3e-3)",
        init_scheme="embd_std_0.1")

    # ---- Reg ----
    add("dropout_0.1", "reg", "dropout=0.1 in blocks and embedding",
        dropout=0.1)

    # ---- Numerics ----
    add("fp32", "numerics", "no autocast, fp32 only",
        use_autocast=False, dtype="float32")
    add("compile_max_autotune", "numerics", "torch.compile(mode='max-autotune')",
        compile_mode="max-autotune")

    # ---- Composition (Phase 3) ----
    # Stack all 3 Phase-2-confirmed improvements: swiglu + qk_norm + lr_1e-2
    add("recipe_v1", "composition",
        "swiglu + qk_norm + lr=1e-2 (all 3 Phase-2-confirmed improvements)",
        activation="swiglu", qk_norm=True, lr=1e-2, min_lr=1e-3)

    return A


# Ablations from the original matrix that need additional code/deps.
# Logged here for the record; not run by this script.
SKIPPED_NOTES = {
    "muon": "Requires Muon optimizer (external library: keller-jordan/Muon).",
    "fp8": "Requires TransformerEngine + Blackwell-native fp8 kernels.",
    "bpe": "Requires tiktoken/sentencepiece + rework of data pipeline.",
    "curriculum_blocksize": "Requires training-loop changes (growing block_size).",
    "cosine_schedule": "Already in baseline.",
    "warmup_100": "Already in baseline.",
    "n_embd_128": "Already in baseline.",
    "tinystories_dataset": "Already in baseline.",
    "grad_clip_1.0": "Already in baseline (see `no_grad_clip` for the inverse).",
}


# ============================================================
# Runner
# ============================================================
def load_done(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("error") is None:
                    done.add(rec["name"])
            except json.JSONDecodeError:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ablation_results.jsonl")
    ap.add_argument("--only", default=None, help="comma-separated names to run")
    ap.add_argument("--skip", default=None, help="comma-separated names to skip")
    ap.add_argument("--list", action="store_true", help="list ablations and exit")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                    help="directory to save model weights (default: checkpoints/)")
    ap.add_argument("--no-checkpoints", action="store_true",
                    help="skip saving model weights")
    args = ap.parse_args()

    ablations = make_ablations()

    if args.list:
        print(f"{'name':<25s} {'category':<11s} description")
        print("-" * 90)
        for a in ablations:
            print(f"{a['name']:<25s} {a['category']:<11s} {a['description']}")
        print("\nSkipped (need additional code/deps):")
        for k, v in SKIPPED_NOTES.items():
            print(f"  {k:<25s} {v}")
        return

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision("high")

    cfg0 = _baseline_cfg()
    print("loading data...")
    t = time.time()
    text = load_text(cfg0)
    data, stoi, itos, vocab_size = encode_text(text)
    data = data.to(device)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"data: {len(data):,} chars, vocab={vocab_size} ({time.time() - t:.1f}s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = load_done(args.out)
    queued = [a for a in ablations
              if a["name"] not in done
              and a["name"] not in skip
              and (only is None or a["name"] in only)]
    print(f"output: {args.out}")
    print(f"already done ({len(done)}): {sorted(done)}")
    print(f"queued ({len(queued)}): {[a['name'] for a in queued]}\n")

    t_total = time.time()
    for i, a in enumerate(queued, 1):
        name = a["name"]
        cfg = a["factory"]()
        cfg.vocab_size = vocab_size

        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(queued)}] {name}  ({a['category']})")
        print(f"      {a['description']}")
        print('=' * 70)

        try:
            rec = run_one(name, a["category"], a["description"], cfg,
                          train_data, val_data, stoi, itos, device,
                          checkpoint_dir=None if args.no_checkpoints else args.checkpoint_dir)
            print(f"  -> val={rec['final']['val_loss']:.4f}  "
                  f"bpc={rec['final']['bpc']:.3f}  "
                  f"time={rec['wallclock_seconds']:.1f}s  "
                  f"peak_mem={rec['peak_memory_mb']:.0f}MB  "
                  f"{'DIVERGED' if rec['diverged'] else ''}")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  !! ERROR: {e}\n{tb}")
            rec = {
                "name": name,
                "category": a["category"],
                "description": a["description"],
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "error": str(e),
                "traceback": tb,
            }

        # Append immediately so a crash later doesn't lose this result
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"\nAll done. Total wallclock: {time.time() - t_total:.1f}s")
    print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()
    