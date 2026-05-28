# Multi-seed sweep results

Generated from `seed_sweep_results.jsonl` on 2026-05-27 23:20.

**Seeds:** [0, 1, 2, 3, 4]  (5 seeds per ablation)
**Bootstrap resamples:** 2000 (95% CI)

**Reading this report:**

- `mean val` and `[ci_lo, ci_hi]` are the bootstrap CI of the mean val_loss across seeds.
- `mean Δ` and `[ci_lo, ci_hi]` are the bootstrap CI of the *paired* difference (ablation_seed - baseline_seed). Paired comparison removes seed-to-seed shared noise.
- `Δ CI excludes 0?` is the key column — if yes, the effect is likely real at the 95% level for this sample size.
- With small n (typically 3-5 seeds), CIs are wide. Don't over-interpret marginal cases. Negative results (CI includes 0) mean "no detectable effect at this sample size," not "no effect."

**Baseline:** mean val = 0.7572 [0.7529, 0.7623], n = 5
**Baseline seed-to-seed sd:** 0.0063 — deltas smaller than ~0.006 are noise-level

## Results (sorted by mean val_loss)

| Name | n | mean val | val CI | mean Δ | Δ CI | Δ excl. 0? |
|---|---:|---:|:---:|---:|:---:|:---:|
| `recipe_v1` | 5 | 0.7055 | [0.7012, 0.7094] | -0.0517 | [-0.0569, -0.0464] | **yes** |
| `lr_1e-2` | 5 | 0.7216 | [0.7178, 0.7253] | -0.0357 | [-0.0427, -0.0291] | **yes** |
| `swiglu` | 5 | 0.7238 | [0.7204, 0.7268] | -0.0335 | [-0.0382, -0.0287] | **yes** |
| `qk_norm` | 5 | 0.7287 | [0.7236, 0.7352] | -0.0285 | [-0.0354, -0.0200] | **yes** |
| `baseline` | 5 | 0.7572 | [0.7529, 0.7623] | +0.0000 | [+0.0000, +0.0000] | (baseline) |

## Interpretation

**Likely real improvements (CI of Δ excludes 0, mean Δ < 0):**
- `recipe_v1`: Δ = -0.0517 [-0.0569, -0.0464]
- `lr_1e-2`: Δ = -0.0357 [-0.0427, -0.0291]
- `swiglu`: Δ = -0.0335 [-0.0382, -0.0287]
- `qk_norm`: Δ = -0.0285 [-0.0354, -0.0200]

## Per-seed val_loss

| Name | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | ... |
|---|---|---|---|---|---|---|
| `recipe_v1` | 0.7083 | 0.7041 | 0.6978 | 0.7127 | 0.7046 |
| `lr_1e-2` | 0.7180 | 0.7274 | 0.7168 | 0.7194 | 0.7262 |
| `swiglu` | 0.7210 | 0.7248 | 0.7183 | 0.7275 | 0.7272 |
| `qk_norm` | 0.7231 | 0.7416 | 0.7291 | 0.7275 | 0.7223 |
| `baseline` | 0.7508 | 0.7541 | 0.7592 | 0.7672 | 0.7548 |
