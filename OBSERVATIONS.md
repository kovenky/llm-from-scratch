# OBSERVATIONS.md — Surprises and unexpected behavior

## 2026-05-27 — Ablation matrix and seed sweep

### lr_1e-2 didn't diverge

The ablation `lr_1e-2` was labeled "expected: NaN / oscillation" in the registry.
Instead it was the top-ranked single-seed result (Δ −0.064) and was confirmed real
by the seed sweep (Δ −0.036 CI [−0.043, −0.029]). Plausible reason: the cosine
decay schedule with 100-step warmup keeps the effective lr low for the first 100
steps, and by midway through training the lr has already decayed substantially below
the peak. So lr_1e-2 → lr_1e-3 (min) with cosine looks much more like a steep-then-
soft lr schedule than a fixed high lr. The stability comes from cosine decay, not the
peak value alone.

### embd_std_0.1 also didn't diverge

`embd_std_0.1` was also labeled "expected: diverges at lr=3e-3." It didn't and
achieved Δ −0.049. Mechanistically: the embedding std affects the scale of the
residual stream early in training. A lower embedding std (0.1 vs 0.02) effectively
scales down the token representations, which may act as implicit label smoothing
early in training. Combined with cosine decay, the model adapts.

Neither of these was included in the composition recipe. Their single-seed results
were not swept in Phase 2 (only top 3 were swept). `embd_std_0.1` may interact with
`lr_1e-2` in a complex way.

### Partial additivity of the recipe

recipe_v1 (swiglu + qk_norm + lr_1e-2) achieved mean Δ = −0.052 against a baseline
of −0.098 predicted from summing individual effects. The improvements compose at
about 53%. The most likely source of redundancy: swiglu (better MLP) and lr_1e-2
(faster early convergence) both accelerate optimization. At a fixed step budget, both
target the same bottleneck. qk_norm is more orthogonal (stabilizes attention) and
likely contributes more independently.

### medium_recipe BPC < 1 (bpc = 0.974)

At 6.4M params (medium_recipe), the model achieves val BPC = 0.974 — below 1 bit per
char. For reference, a model with bpc < 1 is saying "on average I need less than 1
bit to encode each character." This indicates the model has learned strong linguistic
patterns in TinyStories. The text is simple and repetitive, which lowers the entropy
floor. This is expected given the dataset, not a measurement error.

### post_norm catastrophic divergence

`post_norm` achieved val=3.06 (bpc=4.4) — much worse than even random (ln(84) ≈ 4.43).
The model essentially produced near-uniform distributions on the val set, indicating
that training failed entirely. Post-norm (applying LayerNorm after the residual
addition) is known to cause training instability without careful initialization or
warmup strategies. The baseline uses pre-norm (apply norm before), which is standard
in modern transformers.
