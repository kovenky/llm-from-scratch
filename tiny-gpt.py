"""tiny_gpt.py — char-level GPT pretrainer for the 1-min iteration loop.

Three presets at the bottom: tiny() / small() / medium(). Pick one, run.
All hyperparams in the Config dataclass below — edit in one place.
"""
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config — every knob lives here, ablate by editing
# ============================================================
@dataclass
class Config:
    # --- Model architecture ---
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 256
    vocab_size: int = 0           # filled from data

    # --- Training ---
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

    # --- Data ---
    dataset: str = "tinystories"   # "tinystories" or "shakespeare"
    data_mb: int = 50              # cap text size in MB (after ASCII filter)

    # --- Numerics ---
    use_compile: bool = True
    dtype: str = "bfloat16"        # "bfloat16" or "float32"
    seed: int = 1337


def tiny():
    """~111K params, ~30s. Fastest iteration; use for architectural ablations."""
    return Config(
        n_layer=2, n_head=4, n_embd=64, block_size=128,
        max_iters=3000, batch_size=64, data_mb=10,
    )

def small():
    """~1.2M params, ~90s. Sweet spot — coherent TinyStories generations."""
    return Config(
        n_layer=6, n_head=4, n_embd=128, block_size=256,
        max_iters=4000, batch_size=64, data_mb=50,
    )

def medium():
    """~5M params, ~3min. Pushes the 2-min budget; readable stories."""
    return Config(
        n_layer=8, n_head=6, n_embd=192, block_size=256,
        max_iters=5000, batch_size=48, data_mb=100,
    )


# ============================================================
# Data — TinyStories or Shakespeare, char-level, ASCII-only
# ============================================================
def load_text(cfg: Config) -> str:
    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)

    if cfg.dataset == "shakespeare":
        path = cache_dir / "shakespeare.txt"
        if not path.exists():
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, path)
        return path.read_text()

    if cfg.dataset == "tinystories":
        path = cache_dir / f"tinystories_{cfg.data_mb}mb.txt"
        if not path.exists():
            print(f"downloading TinyStories (~{cfg.data_mb}MB target)...")
            from datasets import load_dataset
            # TinyStories train is ~2.7GB; download a little more than needed and trim
            pct = max(1, int(cfg.data_mb / 25) + 1)
            ds = load_dataset("roneneldan/TinyStories", split=f"train[:{pct}%]")
            text = "\n\n".join(ds["text"])
            # ASCII-only for small vocab + fast lookup-table tokenization
            text = text.encode("ascii", errors="ignore").decode("ascii")
            text = text[: cfg.data_mb * 1024 * 1024]
            path.write_text(text)
        return path.read_text()

    raise ValueError(f"unknown dataset: {cfg.dataset}")


def encode_text(text: str):
    """Char-level tokenization using a numpy lookup table (fast at scale)."""
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
# Model — minimal GPT, all the standard pieces, no flourishes
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        h = C // self.n_head
        q = q.view(B, T, self.n_head, h).transpose(1, 2)
        k = k.view(B, T, self.n_head, h).transpose(1, 2)
        v = v.view(B, T, self.n_head, h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=False)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # init BEFORE tying, so we set both copies' source tensor cleanly
        self.apply(self._init_weights)

        # tie weights (head shares storage with tok_emb)
        self.head.weight = self.tok_emb.weight

        # scaled init for residual-stream output projections
        # (must come AFTER self.apply, or it gets overwritten)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
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
# LR schedule — linear warmup + cosine decay
# ============================================================
def get_lr(it, cfg: Config) -> float:
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ============================================================
# Evaluation — quantitative loss + qualitative ability tests
# ============================================================
@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, autocast_dtype, device):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def ability_tests(model, stoi, itos, device, dataset):
    """Generation samples + simple metrics on a long unconditional generation."""
    model.eval()

    prompts = (
        ["Once upon a time", "The little girl", "One day"]
        if dataset == "tinystories"
        else ["ROMEO:", "To be, or not to be", "JULIET:"]
    )

    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)
    for prompt in prompts:
        ids = torch.tensor(
            [[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device
        )
        out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=40)
        text = "".join(itos[i] for i in out[0].tolist())
        print(f"\n[prompt: {prompt!r}]")
        print(text)

    print("\n" + "=" * 60)
    print("ABILITY METRICS (on 1000-char unconditional generation)")
    print("=" * 60)
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=1000, temperature=0.8, top_k=40)
    gen = "".join(itos[i] for i in out[0].tolist())

    words = gen.split()
    word_count = max(1, len(words))
    unique_words = len(set(words))
    avg_word_len = sum(len(w) for w in words) / word_count
    alpha_ratio = sum(w.isalpha() for w in words) / word_count

    # bigram repetition rate — tiny models love to loop
    bigrams = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
    repeat_rate = 1 - len(set(bigrams)) / max(1, len(bigrams))

    print(f"  chars generated:    {len(gen)}")
    print(f"  word count:         {word_count}")
    print(f"  unique words:       {unique_words}")
    print(f"  lexical diversity:  {unique_words / word_count:.3f}")
    print(f"  avg word length:    {avg_word_len:.2f}")
    print(f"  alphabetic words:   {alpha_ratio:.1%}")
    print(f"  bigram repeat rate: {repeat_rate:.1%}")

    model.train()


# ============================================================
# Main training entry point
# ============================================================
def train(cfg: Config):
    print(f"\n{'=' * 60}\nCONFIG\n{'=' * 60}")
    for k, v in cfg.__dict__.items():
        print(f"  {k:18s} = {v}")

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data ---
    t = time.time()
    text = load_text(cfg)
    data, stoi, itos, vocab_size = encode_text(text)
    cfg.vocab_size = vocab_size
    data = data.to(device)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"\ndata: {len(data):,} chars, vocab={vocab_size} "
          f"({time.time() - t:.1f}s to load)")

    # --- Model ---
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_unique = n_params - cfg.vocab_size * cfg.n_embd  # tied head
    print(f"model: {n_params:,} params ({n_unique:,} unique)")

    if cfg.use_compile and device == "cuda":
        model = torch.compile(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    autocast_dtype = (
        torch.bfloat16 if (cfg.dtype == "bfloat16" and device == "cuda") else torch.float32
    )

    # --- Train loop ---
    print("\ntraining...")
    t0 = time.time()
    for step in range(cfg.max_iters + 1):
        lr = get_lr(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, autocast_dtype, device)
            bpc = losses["val"] / math.log(2)
            ppl = math.exp(losses["val"])
            print(f"step {step:5d} | lr {lr:.4f} | train {losses['train']:.4f} "
                  f"| val {losses['val']:.4f} | bpc {bpc:.3f} | ppl {ppl:.2f} "
                  f"| {time.time() - t0:5.1f}s")

        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size)
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    elapsed = time.time() - t0
    print(f"\ntotal training time: {elapsed:.1f}s")

    # --- Ability evaluation ---
    ability_tests(model, stoi, itos, device, cfg.dataset)

    return model, stoi, itos


if __name__ == "__main__":
    # Pick one. Run again with a different preset to compare.
    cfg = small()          # tiny() | small() | medium()
    train(cfg)
    