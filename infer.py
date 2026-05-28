"""infer.py — Generate text from a saved ablation checkpoint.

Loads a checkpoint produced by ablate.py or seed_sweep.py and generates
continuations from a prompt. The checkpoint is self-contained (weights +
config + vocab), so no other files are needed.

Usage:
    # Single completion
    python infer.py --ckpt checkpoints/baseline_seed1337.pt \
        --prompt "Once upon a time"

    # Multiple samples with different sampling parameters
    python infer.py --ckpt checkpoints/rope_seed1337.pt \
        --prompt "The little girl" --n-samples 5 --temperature 0.9 --top-k 80

    # List metadata for all saved checkpoints in a directory
    python infer.py --list checkpoints/

    # Reproducible generation
    python infer.py --ckpt checkpoints/swiglu_seed1337.pt \
        --prompt "One day" --seed 42
"""
import argparse
import json
import os
import sys
import time

import torch

import ablate  # for Config and GPT


def load_checkpoint(path, device):
    """Load a checkpoint and reconstruct the model on `device`."""
    if not os.path.exists(path):
        sys.exit(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Reconstruct Config. asdict preserves the betas tuple through pickle, but
    # filter to known fields in case the dataclass schema evolves later.
    valid_fields = set(ablate.Config.__dataclass_fields__.keys())
    cfg_dict = {k: v for k, v in ckpt["config"].items() if k in valid_fields}
    cfg = ablate.Config(**cfg_dict)

    # Build fresh GPT and load the state_dict. The save side strips
    # torch.compile's '_orig_mod.' prefix, so plain load works.
    model = ablate.GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, cfg, ckpt


def list_checkpoints(directory):
    """Print one line per .pt file with name, seed, val_loss, params."""
    if not os.path.isdir(directory):
        sys.exit(f"not a directory: {directory}")
    paths = sorted(p for p in os.listdir(directory) if p.endswith(".pt"))
    if not paths:
        print(f"no .pt files in {directory}")
        return

    rows = []
    for p in paths:
        full = os.path.join(directory, p)
        try:
            ckpt = torch.load(full, map_location="cpu", weights_only=False)
            name = ckpt.get("name", "?")
            seed = ckpt.get("seed", "?")
            val = ckpt.get("final", {}).get("val_loss", float("nan"))
            cfg = ckpt.get("config", {})
            n_layer = cfg.get("n_layer", "?")
            n_embd = cfg.get("n_embd", "?")
            size_mb = os.path.getsize(full) / (1024 ** 2)
            rows.append((val, name, seed, n_layer, n_embd, size_mb, p))
        except Exception as e:
            rows.append((float("nan"), "?", "?", "?", "?", 0.0, f"{p} (error: {e})"))

    rows.sort(key=lambda r: (r[0] if isinstance(r[0], (int, float)) else float("inf")))

    print(f"{'val_loss':>9} {'name':<25} {'seed':>6} {'layers':>7} {'n_embd':>7} "
          f"{'size_MB':>8}  file")
    print("-" * 95)
    for val, name, seed, n_layer, n_embd, size_mb, p in rows:
        val_str = f"{val:.4f}" if isinstance(val, float) and val == val else "—"
        print(f"{val_str:>9} {str(name):<25} {str(seed):>6} {str(n_layer):>7} "
              f"{str(n_embd):>7} {size_mb:>8.2f}  {p}")


def generate(model, cfg, ckpt, prompt, max_new_tokens, temperature, top_k, n_samples):
    """Sample n continuations from the model."""
    device = next(model.parameters()).device
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    # itos keys may have come through as strings if anything intermediate
    # serialized through JSON. Coerce to int just in case.
    itos = {int(k): v for k, v in itos.items()}

    # Encode the prompt; unknown chars map to token 0 (same convention as
    # ablate.ability_tests).
    prompt_ids = [stoi.get(c, 0) for c in prompt]
    if not prompt_ids:
        prompt_ids = [0]
    ids_template = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"# checkpoint: {ckpt.get('name', '?')}  (seed={ckpt.get('seed', '?')})")
    print(f"# val_loss: {ckpt['final']['val_loss']:.4f}  "
          f"bpc: {ckpt['final']['bpc']:.3f}  "
          f"ppl: {ckpt['final']['ppl']:.2f}")
    print(f"# sampling: temperature={temperature}, top_k={top_k}, max_new_tokens={max_new_tokens}")
    print()

    for i in range(n_samples):
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids_template, max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_k=top_k)
        text = "".join(itos[int(t)] for t in out[0].tolist())
        elapsed = time.time() - t0

        if n_samples > 1:
            print(f"--- sample {i + 1}/{n_samples}  ({elapsed:.1f}s) ---")
        print(text)
        if n_samples > 1:
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="path to checkpoint .pt file")
    ap.add_argument("--list", metavar="DIR",
                    help="list all checkpoints in a directory and exit")
    ap.add_argument("--prompt", default="Once upon a time",
                    help="text to start generation from")
    ap.add_argument("--max-tokens", type=int, default=200,
                    dest="max_new_tokens")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for sampling RNG (default: not reseeded)")
    args = ap.parse_args()

    if args.list:
        list_checkpoints(args.list)
        return

    if not args.ckpt:
        sys.exit("either --ckpt or --list is required (run with -h for help)")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ckpt = load_checkpoint(args.ckpt, device)

    generate(model, cfg, ckpt, args.prompt,
             max_new_tokens=args.max_new_tokens,
             temperature=args.temperature,
             top_k=args.top_k,
             n_samples=args.n_samples)


if __name__ == "__main__":
    main()