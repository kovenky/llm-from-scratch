# Scaling study results — recipe_v1

Generated 2026-05-27 23:42.

**Recipe:** swiglu + qk_norm + lr=1e-2, min_lr=1e-3
**Fixed compute budget:** 4000 iters × 64 batch × 256 block = 65,536,000 tokens per run
**Data:** tinystories_50mb.txt (≈1.4 epochs per run)
**Seed:** 1337 (single run per config — trend, not point estimate)

## Results

| Config | Params | Val loss | BPC | FLOPs (6ND) | Time (s) |
|---|---:|---:|---:|---:|---:|
| `micro_recipe` | 0.223M | 0.8576 | 1.237 | 8.78e+13 | 50 |
| `small_recipe` | 1.231M | 0.7052 | 1.017 | 4.84e+14 | 79 |
| `medium_recipe` | 6.417M | 0.6751 | 0.974 | 2.52e+15 | 246 |
| `large_recipe` | 21.375M | 0.6264 | 0.904 | 8.41e+15 | 540 |

Note: single-seed estimates; noise is ~±0.01 val loss at this scale.
See `scaling_curve.png` for the visual trend.