# tiny-gpt ablation results

Generated from `ablation_results.jsonl` on 2026-05-27 20:19.

**Baseline:** val_loss = 0.7834, bpc = 1.130, params = 1.22M, seed = 1337
**Ablations:** 27 total (27 ok, 0 diverged, 0 errored)
**Total wallclock:** 2093s (34.9m)

**Caveats:** Single seed per ablation (no variance estimate). Differences smaller than ~0.01 val loss are likely noise. Generation metrics depend on sampling RNG, not just the model.

**Best:** `swiglu` val=0.7173 (Δ -0.0662)
**Worst (non-diverged):** `post_norm` val=3.0582 (Δ +2.2747)

## Ranked by val_loss (lower = better)

| Rank | Name | Category | Params | Val | Δ Base | BPC | Lex.Div | Bigram% | Time(s) | Mem(MB) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `swiglu` | arch | 1.23M | 0.7173 | -0.0662 | 1.035 | 0.517 | 15.0 | 89.1 | 1044 |
| 2 | `lr_1e-2` | optim | 1.22M | 0.7197 | -0.0637 | 1.038 | 0.635 | 4.5 | 68.2 | 944 |
| 3 | `qk_norm` | arch | 1.23M | 0.7237 | -0.0597 | 1.044 | 0.562 | 13.0 | 131.2 | 1197 |
| 4 | `embd_std_0.1` | init | 1.22M | 0.7348 | -0.0487 | 1.060 | 0.615 | 3.9 | 71.4 | 944 |
| 5 | `relu2` | arch | 1.22M | 0.7366 | -0.0468 | 1.063 | 0.594 | 9.2 | 132.3 | 1230 |
| 6 | `rope` | arch | 1.19M | 0.7509 | -0.0326 | 1.083 | 0.682 | 4.1 | 104.0 | 1025 |
| 7 | `decoupled_wd` | optim | 1.22M | 0.7534 | -0.0301 | 1.087 | 0.610 | 6.9 | 68.7 | 944 |
| 8 | `wd_zero` | optim | 1.22M | 0.7636 | -0.0198 | 1.102 | 0.573 | 9.6 | 68.7 | 944 |
| 9 | `fp32` | numerics | 1.22M | 0.7649 | -0.0186 | 1.103 | 0.613 | 5.6 | 132.6 | 1295 |
| 10 | `no_grad_clip` | optim | 1.22M | 0.7708 | -0.0126 | 1.112 | 0.600 | 10.1 | 69.8 | 944 |
| 11 | `compile_max_autotune` | numerics | 1.22M | 0.7732 | -0.0103 | 1.115 | 0.608 | 5.1 | 80.3 | 954 |
| 12 | `rmsnorm` | arch | 1.22M | 0.7744 | -0.0090 | 1.117 | 0.627 | 5.0 | 109.6 | 1070 |
| 13 | `baseline` ⭐ | baseline | 1.22M | 0.7834 | +0.0000 | 1.130 | 0.644 | 5.4 | 75.8 | 944 |
| 14 | `untied` | arch | 1.24M | 0.7879 | +0.0045 | 1.137 | 0.623 | 5.4 | 80.8 | 954 |
| 15 | `default_init` | init | 1.22M | 0.8077 | +0.0243 | 1.165 | 0.607 | 7.8 | 70.9 | 944 |
| 16 | `n_layer_4` | arch-size | 831.1K | 0.8121 | +0.0287 | 1.172 | 0.742 | 1.7 | 54.2 | 794 |
| 17 | `nope` | arch | 1.19M | 0.8310 | +0.0476 | 1.199 | 0.649 | 3.0 | 82.7 | 953 |
| 18 | `dropout_0.1` | reg | 1.22M | 0.8335 | +0.0501 | 1.203 | 0.538 | 9.7 | 84.5 | 980 |
| 19 | `lr_1e-3` | optim | 1.22M | 0.8453 | +0.0618 | 1.219 | 0.590 | 5.4 | 68.0 | 944 |
| 20 | `no_warmup` | optim | 1.22M | 0.8532 | +0.0698 | 1.231 | 0.665 | 4.2 | 68.2 | 944 |
| 21 | `sgd_momentum` | optim | 1.22M | 0.8975 | +0.1141 | 1.295 | 0.618 | 5.4 | 67.7 | 940 |
| 22 | `n_layer_1` | arch-size | 240.5K | 1.0432 | +0.2598 | 1.505 | 0.603 | 5.9 | 29.2 | 577 |
| 23 | `deep_narrow` | arch-size | 127.3K | 1.1339 | +0.3504 | 1.636 | 0.635 | 3.5 | 59.8 | 615 |
| 24 | `wide_shallow` | arch-size | 143.5K | 1.1345 | +0.3511 | 1.637 | 0.644 | 5.5 | 30.0 | 550 |
| 25 | `n_embd_32` | arch-size | 85.0K | 1.2290 | +0.4456 | 1.773 | 0.636 | 2.0 | 50.1 | 617 |
| 26 | `lr_1e-4` | optim | 1.22M | 1.3118 | +0.5283 | 1.893 | 0.986 | 0.0 | 67.9 | 944 |
| 27 | `post_norm` | arch | 1.22M | 3.0582 | +2.2747 | 4.412 | 0.853 | 0.0 | 77.6 | 954 |

## By category

### baseline

| Name | Val | Δ Base | BPC | Description |
|---|---:|---:|---:|---|
| `baseline` | 0.7834 | +0.0000 | 1.130 | as written |

### optim

| Name | Val | Δ Base | BPC | Description |
|---|---:|---:|---:|---|
| `lr_1e-2` | 0.7197 | -0.0637 | 1.038 | lr=1e-2 (expected: NaN / oscillation) |
| `decoupled_wd` | 0.7534 | -0.0301 | 1.087 | no WD on 1D params and embeddings |
| `wd_zero` | 0.7636 | -0.0198 | 1.102 | weight_decay=0 (expected: train<val gap) |
| `no_grad_clip` | 0.7708 | -0.0126 | 1.112 | grad_clip=0 |
| `lr_1e-3` | 0.8453 | +0.0618 | 1.219 | lr=1e-3 (expected: slower than 3e-3 but stable) |
| `no_warmup` | 0.8532 | +0.0698 | 1.231 | warmup_iters=0 |
| `sgd_momentum` | 0.8975 | +0.1141 | 1.295 | SGD lr=0.5 momentum=0.9 (expected: much worse) |
| `lr_1e-4` | 1.3118 | +0.5283 | 1.893 | lr=1e-4 (expected: barely moves) |

### arch-size

| Name | Val | Δ Base | BPC | Description |
|---|---:|---:|---:|---|
| `n_layer_4` | 0.8121 | +0.0287 | 1.172 | n_layer=4 |
| `n_layer_1` | 1.0432 | +0.2598 | 1.505 | n_layer=1 |
| `deep_narrow` | 1.1339 | +0.3504 | 1.636 | n_embd=48, n_layer=4 (iso-param vs wide_shallow) |
| `wide_shallow` | 1.1345 | +0.3511 | 1.637 | n_embd=96, n_layer=1 (iso-param vs deep_narrow) |
| `n_embd_32` | 1.2290 | +0.4456 | 1.773 | n_embd=32 |

### arch

| Name | Val | Δ Base | BPC | Description |
|---|---:|---:|---:|---|
| `swiglu` | 0.7173 | -0.0662 | 1.035 | SwiGLU MLP (~8/3 hidden) |
| `qk_norm` | 0.7237 | -0.0597 | 1.044 | RMSNorm on Q, K |
| `relu2` | 0.7366 | -0.0468 | 1.063 | ReLU^2 activation |
| `rope` | 0.7509 | -0.0326 | 1.083 | RoPE position encoding (replaces learned) |
| `rmsnorm` | 0.7744 | -0.0090 | 1.117 | LayerNorm -> RMSNorm |
| `untied` | 0.7879 | +0.0045 | 1.137 | untied input/output embeddings |
| `nope` | 0.8310 | +0.0476 | 1.199 | no position encoding |
| `post_norm` | 3.0582 | +2.2747 | 4.412 | post-norm residual |

### init

| Name | Val | Δ Base | BPC | Description |
|---|---:|---:|---:|---|
| `embd_std_0.1` | 0.7348 | -0.0487 | 1.060 | embedding std=0.1 (expected: diverges at lr=3e-3) |
| `default_init` | 0.8077 | +0.0243 | 1.165 | PyTorch default init (skip scaled init) |

### reg

| Name | Val | Δ Base | BPC | Description |
|---|---:|---:|---:|---|
| `dropout_0.1` | 0.8335 | +0.0501 | 1.203 | dropout=0.1 in blocks and embedding |

### numerics

| Name | Val | Δ Base | BPC | Description |
|---|---:|---:|---:|---|
| `fp32` | 0.7649 | -0.0186 | 1.103 | no autocast, fp32 only |
| `compile_max_autotune` | 0.7732 | -0.0103 | 1.115 | torch.compile(mode='max-autotune') |

## Generation samples

### baseline (val_loss = 0.7834)

**prompt:** `Once upon a time`

```
Once upon a time, there was a ball who was a makeured. He was so happy and he felt sad and cleaned. He had a toy happily and he was different blocks and happy. 

One day, the balloon was playing with the balloon's ho
```

**prompt:** `The little girl`

```
The little girl and the valley brown in the forest. They are very excited. The little girl was happy to see them for her such an island. The little girl asked her to help her find her mom.

The bird made her favorit
```

**prompt:** `One day`

```
One day, the little girl a cat who lived in the ground. She finally folded a big hole in her hand. It was polish and licked its waiting from the pole. So, she went to the zoo and took the interesting for her
```

### top: swiglu (val_loss = 0.7173)

**prompt:** `Once upon a time`

```
Once upon a time, there was a boy who lived in a small house with his yacht. He loved to explore the world and explore the world around him. Every day he went away, the sun would explored the park until he met a rain
```

**prompt:** `The little girl`

```
The little girl and the villagers were gone for the rest of the day.

The little girl and her mom were very happy to remember the silver sun still there with everything. They lived happily ever after and laughed.

O
```

**prompt:** `One day`

```
One day, the lion got very cooler and Lily saw a big box of flowers that lived in the garden. Lily was so excited. She carried the box and tried to fix out the trees. She obedied the box with a piece of appl
```

### top: lr_1e-2 (val_loss = 0.7197)

**prompt:** `Once upon a time`

```
Once upon a time, there was a big mouse. The mouse was sitting in the yard and it was a beautiful pistol that looked like for siles. One day, the mouse decided to watch the sun and the family passed the pistol curly.
```

**prompt:** `The little girl`

```
The little girl and the tree were very happy.

Once upon a time there was a big tree with many colors. The truck wanted to spray a place and spray the string. The tree was very happy and could not stop Timmy's skin.
```

**prompt:** `One day`

```
One day, the little girl asked why she wanted some special spicy food. She asked her mom if she could see it before. Her mom meowed to play food at the other side. Lisa felt sad and went back home, but she c
```

### top: qk_norm (val_loss = 0.7237)

**prompt:** `Once upon a time`

```
Once upon a time, there was a boy who lived in a house. He had a big hat that he loved to play and he went on the ground. He loved to take his truck and listen to his favorite toy about his truck. Every day, he would
```

**prompt:** `The little girl`

```
The little girl and the vase were very happy.

Once upon a time there was a big tree. It was boat and always like the hole. One day, the little girl was feeling very happy. She wanted to see it, so she decided to ha
```

**prompt:** `One day`

```
One day, Lily's mom asked. She was pleased with a little girl who was so happy to see her eyes. Her mom said thank her and hugged her and said goodbye. Lily was so happy welcomed to her eyes and never gave h
```

### bottom: n_embd_32 (val_loss = 1.2290)

**prompt:** `Once upon a time`

```
Once upon a time, there was a ball was so much keard. As the cast, but and hands a game and went grass. He water he colour was so very special to he candle, the found a sunder that to bear.

"The drove and make use t
```

**prompt:** `The little girl`

```
The little girl and the tround was ice girl everyone nearne day all jess needing he safelt, thought with something for the sparkly in and playing that some make he tree.

Then thanked the lave aby a many when that i
```

**prompt:** `One day`

```
One day, go there gamoon. Instelf a learnt for a little very mom. She softed to explored her her wave animal beautially was smelly to spot after the bagon and went to clower.

As then her inter in a nearnver
```

### bottom: lr_1e-4 (val_loss = 1.3118)

**prompt:** `Once upon a time`

```
Once upon a time, buse the can was was Sara. The arend sit the started and back. The said, "I want to be park. While to play while." They wanted to underly, the for the sunder. They said, Lry play cat to the sta's pr
```

**prompt:** `The little girl`

```
The little girl and the tround to big agaramed. He monned his were gots to the big firendt, they do with to them it next to play still and the spack. He sLily was stry to get and loud to carve a soot and the the the
```

**prompt:** `One day`

```
One day, so her mom with a nextled more every said, "Commy, find it this. The little for her chiler to like in the did."

Janme was that him something good to wanter hingroby. He gooddcedbed he"

Her mommy a
```

### bottom: post_norm (val_loss = 3.0582)

**prompt:** `Once upon a time`

```
Once upon a timehohty 
ns ee    er w n S ima eak artniesi mn c isa u yyat     o.la uaonnae  wyhM gpts n  ewoht e h eeelau   T hiotvaditky rh,lrl.n  uaanrn ,at    owdtea    dlo hhteleeee ararn p s     pte eymro c'rn s
```

**prompt:** `The little girl`

```
The little girloaauntm gtr  ea orc ioi girame
m um  ennehah   e ei gnssh e,edi  t t saee th tToon e wtads r
 ueh   nnrhrius pnst stieoaau p  nts aO t ee n em ee hpnteh . aI hn n.al! u 
koi avntuby o m  Tslhd  tsathr
```

**prompt:** `One day`

```
One day ag f lt meg  ive a nttol d  nen eriiiyseaen.tk evglddmntproet io. aaet lehted  d he hihslwe tlaei po   
 eeaplhaanr  meel  " naooh  f
omeot. tdgo   h rw aia   nn kebeagirs  oweh eeint o"infsonfiaah. 
```
