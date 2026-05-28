"""analyze.py — Summarize ablation results from ablation_results.jsonl.

Reads the JSONL produced by ablate.py and emits:

  RESULTS.md           — markdown report (summary, ranked table, per-category
                          breakdown, diverged runs, generation samples)
  loss_curves.png      — per-category val-loss curves vs baseline
  delta_chart.png      — bar chart of final val_loss Δ from baseline
  results_summary.csv  — flat CSV of one row per ablation for further analysis

Usage:
    python analyze.py                              # reads ablation_results.jsonl
    python analyze.py --in results.jsonl --out-dir analysis/
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt


# ============================================================
# Load and shape
# ============================================================
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def split_rows(rows):
    """Partition into (ok, diverged, errored). Last occurrence wins per name."""
    by_name = {}
    for r in rows:
        by_name[r["name"]] = r  # later entries override earlier
    ok, diverged, errored = [], [], []
    for r in by_name.values():
        if r.get("error"):
            errored.append(r)
        elif r.get("diverged"):
            diverged.append(r)
        else:
            ok.append(r)
    return ok, diverged, errored


def get_baseline(ok_rows):
    for r in ok_rows:
        if r["name"] == "baseline":
            return r
    return None


# ============================================================
# Markdown report
# ============================================================
def fmt_params(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def write_markdown(ok, diverged, errored, baseline, out_path, source_path):
    lines = []
    lines.append("# tiny-gpt ablation results")
    lines.append("")
    lines.append(f"Generated from `{source_path}` on "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M')}.")
    lines.append("")

    # Summary
    total = len(ok) + len(diverged) + len(errored)
    total_time = sum(r["wallclock_seconds"] for r in ok + diverged)
    if baseline:
        lines.append(f"**Baseline:** val_loss = {baseline['final']['val_loss']:.4f}, "
                     f"bpc = {baseline['final']['bpc']:.3f}, "
                     f"params = {fmt_params(baseline['n_params'])}, "
                     f"seed = {baseline['seed']}")
    lines.append(f"**Ablations:** {total} total ({len(ok)} ok, "
                 f"{len(diverged)} diverged, {len(errored)} errored)")
    lines.append(f"**Total wallclock:** {total_time:.0f}s "
                 f"({total_time / 60:.1f}m)")
    lines.append("")
    lines.append("**Caveats:** Single seed per ablation (no variance estimate). "
                 "Differences smaller than ~0.01 val loss are likely noise. "
                 "Generation metrics depend on sampling RNG, not just the model.")
    lines.append("")

    if ok:
        best = min(ok, key=lambda r: r["final"]["val_loss"])
        worst = max(ok, key=lambda r: r["final"]["val_loss"])
        if baseline:
            b = baseline["final"]["val_loss"]
            lines.append(f"**Best:** `{best['name']}` "
                         f"val={best['final']['val_loss']:.4f} "
                         f"(Δ {best['final']['val_loss'] - b:+.4f})")
            lines.append(f"**Worst (non-diverged):** `{worst['name']}` "
                         f"val={worst['final']['val_loss']:.4f} "
                         f"(Δ {worst['final']['val_loss'] - b:+.4f})")
        lines.append("")

    # Ranked table
    lines.append("## Ranked by val_loss (lower = better)")
    lines.append("")
    lines.append("| Rank | Name | Category | Params | Val | Δ Base | BPC | "
                 "Lex.Div | Bigram% | Time(s) | Mem(MB) |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    sorted_ok = sorted(ok, key=lambda r: r["final"]["val_loss"])
    base_val = baseline["final"]["val_loss"] if baseline else None
    for i, r in enumerate(sorted_ok, 1):
        v = r["final"]["val_loss"]
        delta = f"{v - base_val:+.4f}" if base_val is not None else "—"
        gen = r.get("generation", {})
        lex = gen.get("lexical_diversity", float("nan"))
        rep = gen.get("bigram_repeat_rate", float("nan"))
        marker = " ⭐" if r["name"] == "baseline" else ""
        lines.append(
            f"| {i} | `{r['name']}`{marker} | {r['category']} | "
            f"{fmt_params(r['n_params'])} | "
            f"{v:.4f} | {delta} | {r['final']['bpc']:.3f} | "
            f"{lex:.3f} | {rep * 100:.1f} | "
            f"{r['wallclock_seconds']:.1f} | {r['peak_memory_mb']:.0f} |"
        )
    lines.append("")

    # Per-category
    lines.append("## By category")
    lines.append("")
    by_cat = defaultdict(list)
    for r in ok:
        by_cat[r["category"]].append(r)
    cat_order = ["baseline", "optim", "arch-size", "arch", "init", "reg", "numerics"]
    for cat in cat_order:
        if cat not in by_cat:
            continue
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Name | Val | Δ Base | BPC | Description |")
        lines.append("|---|---:|---:|---:|---|")
        cat_rows = sorted(by_cat[cat], key=lambda r: r["final"]["val_loss"])
        for r in cat_rows:
            v = r["final"]["val_loss"]
            delta = f"{v - base_val:+.4f}" if base_val is not None else "—"
            lines.append(f"| `{r['name']}` | {v:.4f} | {delta} | "
                         f"{r['final']['bpc']:.3f} | {r['description']} |")
        lines.append("")

    # Diverged
    if diverged:
        lines.append("## Diverged runs")
        lines.append("")
        lines.append("| Name | Diverged at step | Last val | Description |")
        lines.append("|---|---:|---:|---|")
        for r in diverged:
            last_val = (r["loss_curve"]["val_loss"][-1]
                        if r["loss_curve"]["val_loss"] else float("nan"))
            step = r.get("diverge_step", "—")
            lines.append(f"| `{r['name']}` | {step} | "
                         f"{last_val if isinstance(last_val, str) else f'{last_val:.4f}'} | "
                         f"{r['description']} |")
        lines.append("")

    # Errored
    if errored:
        lines.append("## Errored runs")
        lines.append("")
        for r in errored:
            lines.append(f"- `{r['name']}` ({r['category']}): {r.get('error', 'unknown error')}")
        lines.append("")

    # Generation samples (baseline + top-3 + bottom-3 by val_loss)
    lines.append("## Generation samples")
    lines.append("")
    sample_set = []
    if baseline:
        sample_set.append(("baseline", baseline))
    top = sorted_ok[:3]
    bot = sorted_ok[-3:]
    for r in top:
        if r["name"] != "baseline":
            sample_set.append((f"top: {r['name']}", r))
    for r in bot:
        if r["name"] != "baseline" and r not in top:
            sample_set.append((f"bottom: {r['name']}", r))

    for label, r in sample_set:
        lines.append(f"### {label} (val_loss = {r['final']['val_loss']:.4f})")
        lines.append("")
        gen = r.get("generation", {})
        for s in gen.get("samples", []):
            lines.append(f"**prompt:** `{s['prompt']}`")
            lines.append("")
            lines.append("```")
            lines.append(s["completion"])
            lines.append("```")
            lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


# ============================================================
# Plots
# ============================================================
def plot_loss_curves(ok, baseline, out_path):
    """Grid of subplots, one per category, each showing val loss vs step.
    Baseline is overlaid in black on every subplot for reference."""
    by_cat = defaultdict(list)
    for r in ok:
        if r["name"] == "baseline":
            continue
        by_cat[r["category"]].append(r)

    cats = [c for c in ["optim", "arch-size", "arch", "init", "reg", "numerics"]
            if c in by_cat]
    if not cats:
        print("no non-baseline categories to plot")
        return

    ncols = 2
    nrows = (len(cats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows), squeeze=False)

    cmap_names = ["tab10", "Set2", "Dark2"]

    for idx, cat in enumerate(cats):
        ax = axes[idx // ncols][idx % ncols]
        rows = sorted(by_cat[cat], key=lambda r: r["final"]["val_loss"])
        cmap = plt.get_cmap(cmap_names[idx % len(cmap_names)])

        if baseline:
            steps = baseline["loss_curve"]["steps"]
            vals = baseline["loss_curve"]["val_loss"]
            ax.plot(steps, vals, color="black", linewidth=2.5,
                    label="baseline", linestyle="--", alpha=0.8)

        for j, r in enumerate(rows):
            steps = r["loss_curve"]["steps"]
            vals = r["loss_curve"]["val_loss"]
            color = cmap(j % 10)
            ax.plot(steps, vals, color=color, linewidth=1.5,
                    label=r["name"], alpha=0.9)

        ax.set_title(f"{cat}  ({len(rows)} ablations)", fontsize=11)
        ax.set_xlabel("step")
        ax.set_ylabel("val loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    # Hide unused subplots
    for k in range(len(cats), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_delta_chart(ok, baseline, out_path):
    """Horizontal bar chart of final val_loss Δ from baseline, sorted, color by category."""
    if not baseline:
        print("no baseline; skipping delta chart")
        return

    base_val = baseline["final"]["val_loss"]
    rows = [(r["name"], r["category"], r["final"]["val_loss"] - base_val)
            for r in ok if r["name"] != "baseline"]
    rows.sort(key=lambda x: x[2])

    if not rows:
        print("no non-baseline rows for delta chart")
        return

    names = [r[0] for r in rows]
    cats = [r[1] for r in rows]
    deltas = [r[2] for r in rows]

    cat_palette = {
        "optim": "#1f77b4", "arch-size": "#ff7f0e", "arch": "#2ca02c",
        "init": "#d62728", "reg": "#9467bd", "numerics": "#8c564b",
    }
    colors = [cat_palette.get(c, "#7f7f7f") for c in cats]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(rows))))
    ypos = list(range(len(rows)))
    ax.barh(ypos, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"val_loss − baseline  (baseline = {base_val:.4f})")
    ax.set_title("Final val loss vs baseline (negative = better)")
    ax.grid(True, axis="x", alpha=0.3)

    # value labels
    for i, d in enumerate(deltas):
        offset = 0.001 if d >= 0 else -0.001
        ha = "left" if d >= 0 else "right"
        ax.text(d + offset, i, f"{d:+.3f}", va="center", ha=ha, fontsize=8)

    # category legend
    from matplotlib.patches import Patch
    handles = [Patch(color=v, label=k) for k, v in cat_palette.items()
               if k in cats]
    ax.legend(handles=handles, loc="lower right", fontsize=9, title="category")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ============================================================
# CSV
# ============================================================
def write_csv(ok, diverged, errored, baseline, out_path):
    fieldnames = [
        "name", "category", "description", "n_params", "val_loss", "delta_baseline",
        "bpc", "ppl", "lexical_diversity", "bigram_repeat_rate", "alphabetic_ratio",
        "avg_word_length", "wallclock_seconds", "peak_memory_mb",
        "diverged", "diverge_step", "error",
    ]
    base_val = baseline["final"]["val_loss"] if baseline else None

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in ok + diverged + errored:
            gen = r.get("generation", {})
            final = r.get("final", {})
            v = final.get("val_loss", float("nan"))
            delta = (v - base_val) if (base_val is not None and not r.get("error")
                                       and not math.isnan(v)) else ""
            w.writerow({
                "name": r["name"],
                "category": r["category"],
                "description": r["description"],
                "n_params": r.get("n_params", ""),
                "val_loss": v if not math.isnan(v) else "",
                "delta_baseline": delta if delta == "" else f"{delta:.4f}",
                "bpc": final.get("bpc", ""),
                "ppl": final.get("ppl", ""),
                "lexical_diversity": gen.get("lexical_diversity", ""),
                "bigram_repeat_rate": gen.get("bigram_repeat_rate", ""),
                "alphabetic_ratio": gen.get("alphabetic_ratio", ""),
                "avg_word_length": gen.get("avg_word_length", ""),
                "wallclock_seconds": r.get("wallclock_seconds", ""),
                "peak_memory_mb": r.get("peak_memory_mb", ""),
                "diverged": r.get("diverged", False),
                "diverge_step": r.get("diverge_step", ""),
                "error": r.get("error") or "",
            })
    print(f"wrote {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="results/ablation_results.jsonl",
                    help="path to JSONL produced by ablate.py")
    ap.add_argument("--report", default="reports/RESULTS.md",
                    help="path for the markdown report")
    ap.add_argument("--figures-dir", dest="figures_dir", default="figures",
                    help="directory for PNG charts")
    ap.add_argument("--csv", default="results/results_summary.csv",
                    help="path for the CSV summary")
    args = ap.parse_args()

    if not os.path.exists(args.in_path):
        print(f"ERROR: {args.in_path} not found. Run ablate.py first.")
        return

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    rows = load_jsonl(args.in_path)
    if not rows:
        print(f"ERROR: no valid rows in {args.in_path}")
        return

    ok, diverged, errored = split_rows(rows)
    baseline = get_baseline(ok)

    print(f"loaded {len(rows)} rows ({len(ok)} ok, "
          f"{len(diverged)} diverged, {len(errored)} errored)")
    if baseline:
        print(f"baseline val_loss = {baseline['final']['val_loss']:.4f}")
    else:
        print("WARNING: no 'baseline' run found — Δ-from-baseline columns will be missing")

    write_markdown(ok, diverged, errored, baseline, args.report, args.in_path)
    plot_loss_curves(ok, baseline,
                     os.path.join(args.figures_dir, "loss_curves.png"))
    plot_delta_chart(ok, baseline,
                     os.path.join(args.figures_dir, "delta_chart.png"))
    write_csv(ok, diverged, errored, baseline, args.csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
    