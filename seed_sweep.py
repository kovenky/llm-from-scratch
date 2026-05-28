"""seed_sweep.py — Multi-seed validation for promising ablations.

For each ablation in --names (or the top-N from ablation_results.jsonl), runs
the baseline AND the ablation across multiple seeds, then computes paired
deltas with bootstrap 95% CIs.

The key question this answers: is the val_loss difference between an ablation
and the baseline real, or within seed-to-seed noise?

If the 95% CI of the paired delta excludes zero, the effect is likely real
(at n=5 seeds, this is the most you can claim — small samples give wide CIs).

Outputs:
  seed_sweep_results.jsonl   — one JSON line per (name, seed) run; resumable
  SEED_SWEEP_RESULTS.md      — markdown report with paired-delta table
  seed_sweep_chart.png       — forest plot of paired deltas with 95% CIs

Usage:
  python seed_sweep.py                                 # auto: top-3 from ablation_results.jsonl, seeds 0..4
  python seed_sweep.py --names rope,swiglu,decoupled_wd
  python seed_sweep.py --top 5 --seeds 0,1,2,3,4,5,6,7
  python seed_sweep.py --analyze-only                  # just rebuild report from existing JSONL
"""
import argparse
import json
import os
import random
import time
import traceback
from collections import defaultdict
from datetime import datetime
from statistics import mean, stdev

import matplotlib.pyplot as plt

import ablate  # reuse Config, GPT, run_one, load_text, encode_text, make_ablations


# ============================================================
# JSONL utilities
# ============================================================
def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
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


def pick_top_ablations(jsonl_path, n):
    """From ablation_results.jsonl, pick the top-N non-baseline ablations by val_loss."""
    rows = load_jsonl(jsonl_path)
    ok = [r for r in rows
          if not r.get("error") and not r.get("diverged") and r["name"] != "baseline"]
    if not ok:
        return []
    ranked = sorted(ok, key=lambda r: r["final"]["val_loss"])
    return [r["name"] for r in ranked[:n]]


# ============================================================
# Statistics — paired bootstrap CI
# ============================================================
def bootstrap_ci(values, n_resample=2000, alpha=0.05, seed=12345):
    """Percentile bootstrap CI for the mean. Returns (mean, ci_lo, ci_hi)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resample):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(alpha / 2 * n_resample)
    hi_idx = int((1 - alpha / 2) * n_resample)
    return sum(values) / n, means[lo_idx], means[min(hi_idx, n_resample - 1)]


# ============================================================
# Sweep runner
# ============================================================
def run_sweep(ablation_names, seeds, out_path, device,
              train_data, val_data, stoi, itos, vocab_size,
              checkpoint_dir=None):
    """For each (name, seed), train and append result to the JSONL."""
    done = set()
    for r in load_jsonl(out_path):
        if r.get("error") is None:
            done.add((r["name"], r["seed"]))

    ablations = {a["name"]: a for a in ablate.make_ablations()}

    # baseline must be in the list for paired comparison
    if "baseline" not in ablation_names:
        ablation_names = ["baseline"] + list(ablation_names)

    # Validate names
    unknown = [n for n in ablation_names if n not in ablations]
    if unknown:
        print(f"ERROR: unknown ablation names: {unknown}")
        print(f"Available: {sorted(ablations.keys())}")
        return

    total = len(ablation_names) * len(seeds)
    i = 0
    t_start = time.time()

    for name in ablation_names:
        for seed in seeds:
            i += 1
            tag = f"[{i}/{total}] {name} seed={seed}"
            if (name, seed) in done:
                print(f"{tag} — skipped (already in JSONL)")
                continue

            cfg = ablations[name]["factory"]()
            cfg.vocab_size = vocab_size
            cfg.seed = seed

            print(f"\n{'=' * 70}")
            print(tag)
            print('=' * 70)

            try:
                rec = ablate.run_one(
                    name, ablations[name]["category"],
                    ablations[name]["description"],
                    cfg, train_data, val_data, stoi, itos, device,
                    checkpoint_dir=checkpoint_dir,
                )
                print(f"  -> val={rec['final']['val_loss']:.4f}  "
                      f"time={rec['wallclock_seconds']:.1f}s  "
                      f"{'DIVERGED' if rec['diverged'] else ''}")
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  !! ERROR: {e}\n{tb}")
                rec = {
                    "name": name, "seed": seed,
                    "category": ablations[name]["category"],
                    "description": ablations[name]["description"],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "error": str(e), "traceback": tb,
                }
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    print(f"\nSweep complete. Total wallclock: {time.time() - t_start:.0f}s")


# ============================================================
# Analyze
# ============================================================
def compute_summary(rows, n_bootstrap=2000):
    """Group rows by name, compute per-ablation stats and paired deltas vs baseline."""
    # Group successful runs by name; track per-seed val_loss
    by_name = defaultdict(dict)  # name -> {seed: val_loss}
    diverged_count = defaultdict(int)
    for r in rows:
        if r.get("error"):
            continue
        if r.get("diverged"):
            diverged_count[r["name"]] += 1
            continue
        by_name[r["name"]][r["seed"]] = r["final"]["val_loss"]

    if "baseline" not in by_name:
        print("WARNING: no successful baseline runs — paired comparison unavailable")
        baseline_by_seed = {}
    else:
        baseline_by_seed = by_name["baseline"]

    summary = []
    for name, seed_vals in by_name.items():
        vals = list(seed_vals.values())
        mean_v, lo, hi = bootstrap_ci(vals, n_resample=n_bootstrap)

        # Paired deltas: only for seeds where we also have a baseline run
        paired_deltas = [
            seed_vals[s] - baseline_by_seed[s]
            for s in seed_vals
            if s in baseline_by_seed and s in seed_vals
        ]
        d_mean, d_lo, d_hi = bootstrap_ci(paired_deltas, n_resample=n_bootstrap)

        # CI of delta excludes zero → likely real effect
        likely_real = (
            len(paired_deltas) >= 3
            and name != "baseline"
            and (d_hi < 0 or d_lo > 0)
        )

        summary.append({
            "name": name,
            "n_seeds": len(vals),
            "vals": sorted(vals),
            "mean_val": mean_v, "val_ci": (lo, hi),
            "paired_deltas": paired_deltas,
            "mean_delta": d_mean, "delta_ci": (d_lo, d_hi),
            "likely_real": likely_real,
            "diverged_runs": diverged_count.get(name, 0),
        })

    summary.sort(key=lambda s: s["mean_val"])
    return summary


def write_markdown(summary, out_path, in_path, seeds, n_bootstrap):
    lines = []
    lines.append("# Multi-seed sweep results")
    lines.append("")
    lines.append(f"Generated from `{in_path}` on "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M')}.")
    lines.append("")
    lines.append(f"**Seeds:** {sorted(set(seeds))}  "
                 f"({len(set(seeds))} seeds per ablation)")
    lines.append(f"**Bootstrap resamples:** {n_bootstrap} (95% CI)")
    lines.append("")
    lines.append("**Reading this report:**")
    lines.append("")
    lines.append("- `mean val` and `[ci_lo, ci_hi]` are the bootstrap CI of the "
                 "mean val_loss across seeds.")
    lines.append("- `mean Δ` and `[ci_lo, ci_hi]` are the bootstrap CI of the "
                 "*paired* difference (ablation_seed - baseline_seed). "
                 "Paired comparison removes seed-to-seed shared noise.")
    lines.append("- `Δ CI excludes 0?` is the key column — if yes, the effect "
                 "is likely real at the 95% level for this sample size.")
    lines.append("- With small n (typically 3-5 seeds), CIs are wide. Don't "
                 "over-interpret marginal cases. Negative results (CI includes 0) "
                 "mean \"no detectable effect at this sample size,\" not \"no effect.\"")
    lines.append("")

    baseline = next((s for s in summary if s["name"] == "baseline"), None)
    if baseline:
        lines.append(f"**Baseline:** mean val = {baseline['mean_val']:.4f} "
                     f"[{baseline['val_ci'][0]:.4f}, {baseline['val_ci'][1]:.4f}], "
                     f"n = {baseline['n_seeds']}")
        if baseline["n_seeds"] >= 2:
            sd = stdev(baseline["vals"])
            lines.append(f"**Baseline seed-to-seed sd:** {sd:.4f} "
                         f"— deltas smaller than ~{sd:.3f} are noise-level")
        lines.append("")

    # Main table
    lines.append("## Results (sorted by mean val_loss)")
    lines.append("")
    lines.append("| Name | n | mean val | val CI | mean Δ | Δ CI | Δ excl. 0? |")
    lines.append("|---|---:|---:|:---:|---:|:---:|:---:|")
    for s in summary:
        mv = s["mean_val"]
        vlo, vhi = s["val_ci"]
        dm = s["mean_delta"]
        dlo, dhi = s["delta_ci"]
        if s["name"] == "baseline":
            mark = "(baseline)"
        elif s["likely_real"]:
            mark = "**yes**" if dm < 0 else "**yes (worse)**"
        else:
            mark = "no"
        diverge_note = f" ⚠️ {s['diverged_runs']} diverged" if s["diverged_runs"] else ""
        lines.append(
            f"| `{s['name']}` | {s['n_seeds']} | {mv:.4f} | "
            f"[{vlo:.4f}, {vhi:.4f}] | {dm:+.4f} | "
            f"[{dlo:+.4f}, {dhi:+.4f}] | {mark}{diverge_note} |"
        )
    lines.append("")

    # Recommendation section
    lines.append("## Interpretation")
    lines.append("")
    real_wins = [s for s in summary
                 if s["likely_real"] and s["mean_delta"] < 0 and s["name"] != "baseline"]
    real_losses = [s for s in summary
                   if s["likely_real"] and s["mean_delta"] > 0 and s["name"] != "baseline"]
    null_results = [s for s in summary
                    if not s["likely_real"] and s["name"] != "baseline"]

    if real_wins:
        lines.append("**Likely real improvements (CI of Δ excludes 0, mean Δ < 0):**")
        for s in real_wins:
            lines.append(f"- `{s['name']}`: Δ = {s['mean_delta']:+.4f} "
                         f"[{s['delta_ci'][0]:+.4f}, {s['delta_ci'][1]:+.4f}]")
        lines.append("")
    if real_losses:
        lines.append("**Likely real regressions (CI of Δ excludes 0, mean Δ > 0):**")
        for s in real_losses:
            lines.append(f"- `{s['name']}`: Δ = {s['mean_delta']:+.4f} "
                         f"[{s['delta_ci'][0]:+.4f}, {s['delta_ci'][1]:+.4f}]")
        lines.append("")
    if null_results:
        lines.append("**Null results (CI of Δ includes 0):** "
                     "no detectable difference from baseline at this sample size.")
        names = ", ".join(f"`{s['name']}`" for s in null_results)
        lines.append(names)
        lines.append("")
    if not real_wins and not real_losses:
        lines.append("No ablation reached statistical separation from baseline at "
                     "this sample size. Either the effects are smaller than the "
                     "seed noise, or you need more seeds.")
        lines.append("")

    # Per-seed details
    lines.append("## Per-seed val_loss")
    lines.append("")
    lines.append("| Name | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | ... |")
    lines.append("|---|---|---|---|---|---|---|")
    rows_by_name_seed = defaultdict(dict)
    rows = load_jsonl(in_path)
    for r in rows:
        if r.get("error") or r.get("diverged"):
            continue
        rows_by_name_seed[r["name"]][r["seed"]] = r["final"]["val_loss"]
    all_seeds = sorted({s for r in rows for s in [r.get("seed")] if s is not None})
    for s in summary:
        vals_by_seed = rows_by_name_seed.get(s["name"], {})
        cells = [f"{vals_by_seed[sd]:.4f}" if sd in vals_by_seed else "—"
                 for sd in all_seeds[:6]]
        while len(cells) < 5:
            cells.append("")
        line = f"| `{s['name']}` | " + " | ".join(cells) + " |"
        lines.append(line)
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


def plot_forest(summary, out_path):
    """Forest plot of paired deltas with 95% CIs. Excludes baseline."""
    rows = [s for s in summary if s["name"] != "baseline" and s["paired_deltas"]]
    if not rows:
        print("no non-baseline rows with paired deltas; skipping plot")
        return
    rows.sort(key=lambda s: s["mean_delta"])

    names = [s["name"] for s in rows]
    means = [s["mean_delta"] for s in rows]
    lo = [s["delta_ci"][0] for s in rows]
    hi = [s["delta_ci"][1] for s in rows]
    real = [s["likely_real"] for s in rows]

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.55 * len(rows))))
    ypos = list(range(len(rows)))

    for i, (m, l, h, r) in enumerate(zip(means, lo, hi, real)):
        color = "#2ca02c" if (r and m < 0) else ("#d62728" if (r and m > 0) else "#888888")
        ax.errorbar(m, i, xerr=[[m - l], [h - m]],
                    fmt="o", color=color, ecolor=color,
                    markersize=8, capsize=4, linewidth=2)
        # individual deltas as small dots
        for d in rows[i]["paired_deltas"]:
            ax.plot(d, i, "o", color=color, alpha=0.25, markersize=4)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("paired Δ val_loss (ablation − baseline, same seed)")
    ax.set_title("Forest plot: paired Δ from baseline, 95% bootstrap CI\n"
                 "green = likely real improvement, red = likely real regression, gray = no effect detected")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-source", default="results/ablation_results.jsonl",
                    help="source JSONL for --top auto-pick")
    ap.add_argument("--out", default="results/seed_sweep_results.jsonl",
                    help="JSONL output (one row per (name, seed))")
    ap.add_argument("--report", default="reports/SEED_SWEEP_RESULTS.md",
                    help="markdown report path")
    ap.add_argument("--chart", default="figures/seed_sweep_chart.png",
                    help="forest plot path")
    ap.add_argument("--names", default=None,
                    help="comma-separated ablation names to sweep (overrides --top)")
    ap.add_argument("--top", type=int, default=3,
                    help="auto-pick top-N from --in-source if --names not given")
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="comma-separated seeds (default: 0,1,2,3,4)")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="bootstrap resamples for CIs")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip training; just rebuild the report from --out JSONL")
    ap.add_argument("--checkpoint-dir", default="checkpoints",
                    help="directory to save weights (default: checkpoints/)")
    ap.add_argument("--no-checkpoints", action="store_true",
                    help="skip saving model weights")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.chart) or ".", exist_ok=True)

    if not args.analyze_only:
        # Pick ablations
        if args.names:
            names = [n.strip() for n in args.names.split(",")]
        else:
            if not os.path.exists(args.in_source):
                print(f"ERROR: --in-source {args.in_source} not found "
                      f"(need it to auto-pick top-N; use --names to specify).")
                return
            names = pick_top_ablations(args.in_source, args.top)
            if not names:
                print(f"ERROR: no ablations to sweep from {args.in_source}")
                return
            print(f"Auto-picked top-{args.top} ablations: {names}")

        # Load data once (matches ablate.py's flow)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {device}")
        if device == "cuda":
            print(f"gpu: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

        cfg0 = ablate.Config()
        print("loading data...")
        t = time.time()
        text = ablate.load_text(cfg0)
        data, stoi, itos, vocab_size = ablate.encode_text(text)
        data = data.to(device)
        n_train = int(0.9 * len(data))
        train_data, val_data = data[:n_train], data[n_train:]
        print(f"data: {len(data):,} chars, vocab={vocab_size} ({time.time() - t:.1f}s)")

        # Estimate total time
        per_run = 75  # seconds, rough
        n_total = (len(names) + 1) * len(seeds)  # +1 for baseline
        already_done = sum(1 for r in load_jsonl(args.out) if r.get("error") is None)
        remaining = max(0, n_total - already_done)
        print(f"\nplan: {len(names)} ablations + baseline × {len(seeds)} seeds "
              f"= {n_total} runs ({remaining} remaining)")
        print(f"estimated remaining time: ~{remaining * per_run / 60:.0f}m")
        print()

        run_sweep(names, seeds, args.out, device,
                  train_data, val_data, stoi, itos, vocab_size,
                  checkpoint_dir=None if args.no_checkpoints else args.checkpoint_dir)

    # Analyze
    rows = load_jsonl(args.out)
    if not rows:
        print(f"ERROR: no results in {args.out}")
        return
    print(f"\nanalyzing {len(rows)} runs...")
    summary = compute_summary(rows, n_bootstrap=args.bootstrap)
    write_markdown(summary, args.report, args.out, seeds, args.bootstrap)
    plot_forest(summary, args.chart)

    print(f"\nDone. See {args.report} and {args.chart}")


if __name__ == "__main__":
    main()
    