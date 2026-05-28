"""scale_study.py — Fixed-compute scaling study for recipe_v1.

Applies recipe_v1 (swiglu + qk_norm + lr=1e-2) at 4 model sizes.
All runs use the same compute budget: 4000 steps × 64 batch × 256 block = 65.5M tokens.
Data: existing tinystories_50mb.txt (≈1.4 epochs per run; note in DECISIONS.md).

Resumable: completed configs (no error) are skipped on re-run.

Outputs:
    scale_study_results.jsonl    — one JSON line per config
    scaling_curve.png            — val loss vs. params and vs. FLOPs
"""
import json
import math
import os
import time
import traceback
from dataclasses import asdict
from datetime import datetime

import matplotlib.pyplot as plt
import torch

import ablate


RESULTS_PATH = "results/scale_study_results.jsonl"
PLOT_PATH = "figures/scaling_curve.png"
SEED = 1337

# Fixed compute budget for all configs
BATCH = 64
BLOCK = 256
ITERS = 4000
# Tokens per run: ITERS × BATCH × BLOCK = 65,536,000

# Recipe overrides (validated in Phase 3 step 1)
RECIPE = dict(activation="swiglu", qk_norm=True, lr=1e-2, min_lr=1e-3)

# Scaling ladder: (name, description, model-size overrides)
# n_head must divide n_embd evenly; head_dim = n_embd // n_head
SCALING_CONFIGS = [
    (
        "micro_recipe",
        "n_embd=64, n_layer=4, n_head=4 + recipe_v1",
        dict(n_embd=64, n_layer=4, n_head=4),
    ),
    (
        "small_recipe",
        "n_embd=128, n_layer=6, n_head=4 + recipe_v1 (= recipe_v1 at small scale)",
        dict(n_embd=128, n_layer=6, n_head=4),
    ),
    (
        "medium_recipe",
        "n_embd=256, n_layer=8, n_head=4 + recipe_v1",
        dict(n_embd=256, n_layer=8, n_head=4),
    ),
    (
        "large_recipe",
        "n_embd=384, n_layer=12, n_head=6 + recipe_v1",
        dict(n_embd=384, n_layer=12, n_head=6),
    ),
]


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def make_config(size_overrides):
    cfg = ablate.Config(
        batch_size=BATCH,
        block_size=BLOCK,
        max_iters=ITERS,
        seed=SEED,
        data_mb=50,
    )
    for k, v in {**RECIPE, **size_overrides}.items():
        setattr(cfg, k, v)
    return cfg


def run_scaling(device, train_data, val_data, stoi, itos, vocab_size):
    done = {r["name"] for r in load_jsonl(RESULTS_PATH) if not r.get("error")}

    total = len(SCALING_CONFIGS)
    t_start = time.time()

    for i, (name, desc, size_overrides) in enumerate(SCALING_CONFIGS, 1):
        tag = f"[{i}/{total}] {name}"
        if name in done:
            print(f"{tag} — skipped (already in JSONL)")
            continue

        cfg = make_config(size_overrides)
        cfg.vocab_size = vocab_size

        print(f"\n{'=' * 70}")
        print(tag)
        print(f"config: n_embd={cfg.n_embd}, n_layer={cfg.n_layer}, "
              f"n_head={cfg.n_head}, iters={cfg.max_iters}")
        print('=' * 70)

        try:
            rec = ablate.run_one(
                name, "scaling", desc,
                cfg, train_data, val_data, stoi, itos, device,
                checkpoint_dir="checkpoints",
            )
            print(f"  -> val={rec['final']['val_loss']:.4f}  "
                  f"params={rec['n_params']:,}  "
                  f"time={rec['wallclock_seconds']:.1f}s")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  !! ERROR: {e}\n{tb}")
            rec = {
                "name": name,
                "category": "scaling",
                "description": desc,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "error": str(e),
                "traceback": tb,
            }

        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"\nAll runs done. Total wallclock: {time.time() - t_start:.0f}s")


def compute_flops(n_params, n_tokens):
    """Rough FLOPs estimate: 6 × N × D (standard approximation for transformer training)."""
    return 6 * n_params * n_tokens


def plot_scaling(rows):
    rows = sorted(rows, key=lambda r: r["n_params"])
    params = [r["n_params"] for r in rows]
    val_losses = [r["final"]["val_loss"] for r in rows]
    names = [r["name"] for r in rows]
    tokens = ITERS * BATCH * BLOCK
    flops = [compute_flops(p, tokens) for p in params]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: val loss vs. param count ---
    ax = axes[0]
    ax.plot(params, val_losses, "o-", color="#1f77b4", linewidth=2, markersize=8)
    for p, v, n in zip(params, val_losses, names):
        ax.annotate(f"{n}\n({p/1e6:.2f}M)", (p, v),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Val loss (↓ better)")
    ax.set_title("Val loss vs. parameter count\n(recipe_v1: swiglu + qk_norm + lr=1e-2)")
    ax.grid(True, alpha=0.3)

    # --- Plot 2: val loss vs. FLOPs ---
    ax = axes[1]
    ax.plot(flops, val_losses, "o-", color="#2ca02c", linewidth=2, markersize=8)
    for f, v, n in zip(flops, val_losses, names):
        ax.annotate(f"{n}", (f, v),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Estimated FLOPs (6·N·D, log scale)")
    ax.set_ylabel("Val loss (↓ better)")
    ax.set_title("Val loss vs. compute (FLOPs)\n(fixed D = 65.5M tokens per run)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOT_PATH}")


def write_report(rows):
    rows = sorted(rows, key=lambda r: r["n_params"])
    tokens = ITERS * BATCH * BLOCK
    lines = [
        "# Scaling study results — recipe_v1",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}.",
        "",
        "**Recipe:** swiglu + qk_norm + lr=1e-2, min_lr=1e-3",
        f"**Fixed compute budget:** {ITERS} iters × {BATCH} batch × {BLOCK} block "
        f"= {tokens:,} tokens per run",
        "**Data:** tinystories_50mb.txt (≈1.4 epochs per run)",
        "**Seed:** 1337 (single run per config — trend, not point estimate)",
        "",
        "## Results",
        "",
        "| Config | Params | Val loss | BPC | FLOPs (6ND) | Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        n = r["n_params"]
        v = r["final"]["val_loss"]
        bpc = r["final"]["bpc"]
        t = r["wallclock_seconds"]
        flops = compute_flops(n, tokens)
        lines.append(
            f"| `{r['name']}` | {n/1e6:.3f}M | {v:.4f} | {bpc:.3f} | "
            f"{flops:.2e} | {t:.0f} |"
        )
    lines.append("")
    lines.append(
        "Note: single-seed estimates; noise is ~±0.01 val loss at this scale."
    )
    lines.append("See `scaling_curve.png` for the visual trend.")

    report_path = "reports/SCALING_RESULTS.md"
    os.makedirs("reports", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {report_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision("high")

    # Load data once (50MB, matches cached file)
    cfg0 = ablate.Config(data_mb=50)
    print("loading data...")
    t = time.time()
    text = ablate.load_text(cfg0)
    data, stoi, itos, vocab_size = ablate.encode_text(text)
    data = data.to(device)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"data: {len(data):,} chars, vocab={vocab_size} ({time.time() - t:.1f}s)")

    # Estimate time
    done = {r["name"] for r in load_jsonl(RESULTS_PATH) if not r.get("error")}
    remaining = [n for n, _, _ in SCALING_CONFIGS if n not in done]
    print(f"\nplan: {len(SCALING_CONFIGS)} configs ({len(remaining)} remaining)")
    print(f"configs: {[n for n, _, _ in SCALING_CONFIGS]}")
    print()

    run_scaling(device, train_data, val_data, stoi, itos, vocab_size)

    # Analyze
    rows = [r for r in load_jsonl(RESULTS_PATH) if not r.get("error") and "n_params" in r]
    if rows:
        write_report(rows)
        plot_scaling(rows)
        print("\nScaling results:")
        for r in sorted(rows, key=lambda r: r["n_params"]):
            print(f"  {r['name']:20s}  params={r['n_params']/1e6:.3f}M  "
                  f"val={r['final']['val_loss']:.4f}  bpc={r['final']['bpc']:.3f}")
    else:
        print("No successful runs to analyze.")


if __name__ == "__main__":
    main()
