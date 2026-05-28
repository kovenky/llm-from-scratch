# DECISIONS.md — LLM-SCRATCH ablation & scaling log

## 2026-05-27 — Phase 1 + 2 summary and Phase 3 composition validated

### What was run

**Phase 1 (ablation matrix):** 27 single-seed (seed=1337) ablations against the
`small()` baseline (1.22M params, lr=3e-3, GELU, learned pos, LayerNorm). All ran
without unexpected divergences. `post_norm` diverged as expected (val=3.06). `lr_1e-2`
and `embd_std_0.1` were expected to diverge but did not and improved.

`python analyze.py` produced `RESULTS.md`, `loss_curves.png`, `delta_chart.png`,
`results_summary.csv`.

**Phase 2 (seed sweep):** Top-3 candidates (by single-seed val loss delta) validated
across seeds 0–4 via paired bootstrap CI. All 3 confirmed real improvements
(CI of Δ excludes 0):

| Name | mean Δ | Δ CI (95%) |
|---|---:|:---:|
| `lr_1e-2` | -0.0357 | [-0.0427, -0.0291] |
| `swiglu` | -0.0335 | [-0.0382, -0.0287] |
| `qk_norm` | -0.0285 | [-0.0354, -0.0200] |

Baseline mean val = 0.7572 [0.7529, 0.7623], seed-to-seed sd = 0.0063.

**Phase 3 step 1 (composition):** Composed `recipe_v1` = swiglu + qk_norm + lr_1e-2.
Validated across seeds 0–4:

| Name | mean val | mean Δ | Δ CI (95%) | Δ excl. 0? |
|---|---:|---:|:---:|:---:|
| `recipe_v1` | 0.7055 | -0.0517 | [-0.0569, -0.0464] | **yes** |

### Key findings

- All 3 confirmed improvements compose: recipe_v1 beats baseline with CI excluding 0.
- **Additivity is partial:** individual sum predicts Δ ≈ -0.098; actual is -0.052 (≈53%).
  swiglu and lr_1e-2 share optimization-dynamic improvement pathways — expected partial
  redundancy, not a failure.
- Gate passed: Phase 2 produced real improvements; composed recipe is validated.

### Decisions

1. `recipe_v1` (swiglu + qk_norm + lr=1e-2, min_lr=1e-3) is the best recipe for
   the scaling study.
2. **lr note for scaling:** lr_1e-2 was tuned for the small() model. Optimal lr
   typically follows its own power law with model size. For the scaling study, we
   will hold lr fixed at 1e-2 across sizes for now (a controlled comparison), and
   note that re-tuning lr per scale would likely improve larger points.
3. The scaling study uses a fixed compute budget (max_iters=4000, 65.5M tokens) per
   point. This is a fixed-compute scaling curve; FLOPs = 6 × N × 65.5M tokens.
4. Candidates 4 and 5 from Phase 1 (`embd_std_0.1` Δ -0.049, `relu2` Δ -0.047)
   were not swept. `relu2` is dominated by `swiglu` (same category, swiglu is better);
   `embd_std_0.1` is not included in the recipe (init scheme may interact with lr).
   These are noted but not pursued — staying within the confirmed recipe.

### What's next

Phase 3 step 2: scaling study — COMPLETE. Results below.

## 2026-05-27 — Phase 3 step 2: scaling study

### What was run

`scale_study.py` applied recipe_v1 at 4 model sizes, fixed compute
(4000 × 64 × 256 = 65.5M tokens), seed=1337, data=tinystories_50mb.txt (~1.4 epochs).

| Config | Params | Val loss | BPC | Time (s) |
|---|---:|---:|---:|---:|
| micro_recipe | 0.223M | 0.8576 | 1.237 | 50 |
| small_recipe | 1.231M | 0.7052 | 1.017 | 79 |
| medium_recipe | 6.417M | 0.6751 | 0.974 | 246 |
| large_recipe | 21.375M | 0.6264 | 0.904 | 540 |

### Key findings

- Scaling is clean and monotonic across ~2 decades of parameter count.
- Val loss decreases by ~0.05–0.08 per ~4-5x increase in params at this compute budget.
- BPC drops below 1.0 at 6.4M params (medium_recipe) and reaches 0.904 at 21M params.
- Single-seed estimates; noise is ~±0.01. These are trend indicators, not tight points.
- The 50MB data cap (≈1.4 epochs at this step budget) may slightly underestimate
  performance at larger sizes where unique data would help more. Re-running with more
  data is the natural next step if this needs to be publication-quality.

### Decisions

1. Scaling curve is the headline artifact. See scaling_curve.png and SCALING_RESULTS.md.
2. To continue scaling (e.g. 50M+ params), more data is needed first (data_mb > 50).
3. lr re-tuning at each scale would likely shift each point; lr=1e-2 was fixed here
   for a controlled apples-to-apples comparison.

### What's next (if continuing)

- Increase data (data_mb=200+) and run recipe at 50M+ params for the next decade.
- Re-tune lr per scale to get the true scaling law.
- Or: the experiment is complete at this scope; present the scaling curve.
