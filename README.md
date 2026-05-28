# LLM-SCRATCH — Char-level GPT Ablation & Scaling Study

A controlled experimental program on a char-level GPT trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
The goal: run a full ablation matrix at small scale, identify which changes
genuinely help (via multi-seed statistical validation), compose the winners
into a best recipe, and run a deliberate scaling study.

**Hardware:** NVIDIA GB10 (Blackwell), single GPU.  
**Dataset:** TinyStories (ASCII, char-level), 50 MB sample.  
**Baseline:** `small()` — 1.22M params, ~90s/run, val loss ≈ 0.783, bpc ≈ 1.13.

---

## Repository layout

```
# Scripts (entry points)
ablate.py              Main ablation runner. Single-variable ablations vs. baseline.
analyze.py             Reads results/ablation_results.jsonl → reports/ + figures/.
seed_sweep.py          Multi-seed validation for candidate ablations (bootstrap CIs).
scale_study.py         Fixed-compute scaling study for the best recipe.
infer.py               Text generation from saved checkpoints.
tiny-gpt.py            Reference trainer with tiny/small/medium presets.

# Project docs (root)
README.md              This file.
DECISIONS.md           Dated log of decisions and conclusions from each session.
OBSERVATIONS.md        Notes on unexpected behavior and surprises.

# Raw data
results/
  ablation_results.jsonl      One JSON line per completed ablation (resumable).
  seed_sweep_results.jsonl    One JSON line per (name, seed) pair from the sweep.
  scale_study_results.jsonl   One JSON line per scaling config.
  results_summary.csv         CSV export of ablation results.

# Generated reports (markdown)
reports/
  RESULTS.md                  Ranked ablation table + generation samples.
  SEED_SWEEP_RESULTS.md       Paired-delta table with 95% bootstrap CIs.
  SCALING_RESULTS.md          Val loss vs. params and FLOPs for the best recipe.

# Generated charts (PNG)
figures/
  loss_curves.png             Loss curves for all ablations.
  delta_chart.png             Bar chart of val-loss deltas.
  seed_sweep_chart.png        Forest plot of paired deltas.
  scaling_curve.png           Scaling curve (val loss vs. params and FLOPs).

# Runtime dirs (gitignored)
data/                  Cached dataset files.
checkpoints/           Saved model weights.
venv/                  Python virtual environment.
```

---

## Quick start

```bash
# Install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Run ablations (resumable — re-run if interrupted)
python ablate.py
python analyze.py       # → RESULTS.md, loss_curves.png, delta_chart.png

# 2. Validate top candidates across seeds
python seed_sweep.py --names swiglu,qk_norm,lr_1e-2 --seeds 0,1,2,3,4

# 3. Validate composition
python seed_sweep.py --names recipe_v1 --seeds 0,1,2,3,4

# 4. Scaling study
python scale_study.py

# Generate text from a saved checkpoint
python infer.py --checkpoint checkpoints/small_recipe_seed1337.pt --prompt "Once upon a time"
```

---

## Experimental program

### Phase 1 — Ablation matrix (27 ablations, single seed)

Every ablation changes exactly one variable from the baseline. Results ranked by
val loss in `RESULTS.md`. Top candidates (Δ > ~0.01 relative to baseline noise):

| Rank | Name | Δ val | Note |
|---|---:|---|
| 1 | `swiglu` | −0.066 | SwiGLU MLP (8/3 hidden) |
| 2 | `lr_1e-2` | −0.064 | lr=1e-2 (surprisingly didn't diverge) |
| 3 | `qk_norm` | −0.060 | RMSNorm on Q and K |
| 4 | `embd_std_0.1` | −0.049 | Embedding init std=0.1 |
| 5 | `relu2` | −0.047 | ReLU² activation |
| … | `post_norm` | +2.27 | Post-norm diverges (expected) |

### Phase 2 — Seed sweep (5 seeds, paired bootstrap CI)

Top 3 candidates from Phase 1 swept across seeds 0–4. All 3 confirmed as real
improvements (CI of paired Δ excludes 0 at 95% level):

| Name | mean Δ | 95% CI |
|---|---:|:---:|
| `lr_1e-2` | −0.0357 | [−0.043, −0.029] |
| `swiglu` | −0.0335 | [−0.038, −0.029] |
| `qk_norm` | −0.0285 | [−0.035, −0.020] |

Baseline seed-to-seed sd = 0.0063 (differences below ~0.006 are noise).

### Phase 3 — Composition + scaling

**recipe_v1** stacks all 3 confirmed improvements:
`activation=swiglu, qk_norm=True, lr=1e-2, min_lr=1e-3`

Composition validated across seeds 0–4:

| Name | mean val | mean Δ | 95% CI |
|---|---:|---:|:---:|
| `recipe_v1` | 0.7055 | −0.0517 | [−0.057, −0.046] |

Partial additivity: individual improvements sum to −0.098; composition achieves
−0.052 (≈ 53%). swiglu and lr_1e-2 share optimization-dynamic pathways — expected
redundancy at this scale.

**Scaling study** (fixed compute: 65.5M tokens per run, seed=1337):

| Config | Params | Val loss | BPC |
|---|---:|---:|---:|
| `micro_recipe` | 0.22M | 0.858 | 1.237 |
| `small_recipe` | 1.23M | 0.705 | 1.017 |
| `medium_recipe` | 6.42M | 0.675 | 0.974 |
| `large_recipe` | 21.4M | 0.626 | 0.904 |

See `SCALING_RESULTS.md` and `scaling_curve.png` for the full curve.

---

## Key findings

- **SwiGLU, QK-norm, and lr=1e-2 all produce real, statistically validated
  improvements** at the small scale (1.2M params on TinyStories).
- **The improvements partially compose**: stacking them gives −0.052 val loss vs.
  the individually predicted −0.098 (53% additivity). Some gain is shared.
- **Single-seed ablations are candidates, not conclusions**: the seed-to-seed sd is
  ±0.006, so differences below ~0.01 are noise at n=1.
- **Model is capacity-bound, not data-bound** at 1.2M params (train/val gap ≈ 0.007).
  More data alone doesn't help; bigger models do.
- The `post_norm` ablation diverged catastrophically (val=3.06), confirming that
  pre-norm is load-bearing at this scale without additional stabilization.

---

## Config reference

The baseline (`small()`) and all ablations use `ablate.Config`:

```python
Config(
    n_layer=6, n_head=4, n_embd=128, block_size=256,
    batch_size=64, max_iters=4000,
    lr=3e-3, min_lr=3e-4, weight_decay=0.1,
    grad_clip=1.0, warmup_iters=100,
    activation="gelu", norm="layernorm",
    pos_encoding="learned", tie_embeddings=True,
    qk_norm=False, dropout=0.0, post_norm=False,
    optimizer="adamw", init_scheme="scaled",
    dtype="bfloat16", use_compile=True,
)
```

`recipe_v1` overrides: `activation="swiglu", qk_norm=True, lr=1e-2, min_lr=1e-3`.

---

## Tokens consumed per run

`max_iters × batch_size × block_size = 4000 × 64 × 256 = 65,536,000 tokens`

The 50 MB dataset provides ≈ 47 MB of training chars. Each run consumes ≈ 1.4
epochs. At 1.2M params, no overfitting is observed (train/val gap ≈ 0.007).
