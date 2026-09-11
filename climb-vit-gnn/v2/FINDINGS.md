# ClimbGNN v2 — results ledger (2026-09-11)

Goal: a grade model that works on **real gym photos**, using Kilter Board
routes as pre-training (the "foundation model for indoor grades" idea).

## Data

| Set | Size | Use |
|---|---|---|
| Kilter routes (`data/processed/{train,val,test}.json`) | 24,844 / 5,335 / 5,046 | pre-training, early stopping, in-domain test |
| Gym routes (HF `ranbirr1/climb-route-w-grade`) | 38 labeled routes on 35 distinct photos (V0–V8) | adaptation + evaluation, photo-grouped CV only |

Kilter hold crops are grey shape renders on a neutral background; gym holds
are coloured, lit, textured, on walls, photographed at odd angles. Every
number below comes from held-out data: Kilter *test* routes were never seen
in training, and gym numbers are out-of-fold predictions from 5-fold CV
grouped by photo, repeated 3 times (each route predicted 3× by models that
never saw its photo).

## 1. Where the v1 model stood (deployed until today)

| Evaluation | MAE (V-grades) | bias | within ±1 | Spearman |
|---|---|---|---|---|
| Recorded app predictions on the 38 labeled gym routes | **2.29** [1.71, 2.92] | +1.18 | 45% | 0.30 |
| Same model re-run with grey/masked crops | 2.26–2.45 | +0.8 to +1.7 | 37–45% | 0.11–0.25 |
| Constant "always V4/V5" predictor | 2.24 | | 39% | |

Two bugs explain part of this: the `foot` role index was never trained
(Kilter data has only start/middle/finish), and hold positions were
normalised by board size, which is meaningless for arbitrary photo framing.

## 2. What changed in v2 (`v2/model_v2.py`, `v2/train_v2.py`)

* **Scale-invariant geometry**: positions centred on the route and scaled by
  its extent; edge distances normalised by the median nearest-neighbour
  distance; rank-normalised coordinates and local density as extra node
  features. Same code path for board and photo.
* **Photo-like augmentation of the renders** (`v2/augment_embed.py`): each of
  the 692 Kilter holds is colourised, composited onto synthetic wall
  backgrounds, rotated/scaled/blurred/noised into 23 extra views; DINOv2
  embeddings of all views are precomputed and sampled during training.
* **Roles**: start / middle / finish / foot all trained (Kilter's small
  screw-on "foot" holds are relabelled as feet with p=0.5).
* **Auxiliary heads** predict hold type / size / depth from the projected
  embedding (labels come from the Kilter hold table) so the projection keeps
  physical hold information.
* **Distributional head**: 21 difficulty bins, soft-label cross-entropy;
  expected value is the point estimate, 16–84% quantiles give the range.
* Route-level augmentation: mirror, hold dropout; ascent-count sample
  weights; 3-seed ensemble.

## 3. Kilter in-domain test (5,046 routes, never trained on)

| Seed | MAE (difficulty units) | within ±1 unit | MAE (V-grades) | within ±1 V-grade |
|---|---|---|---|---|
| 0 | 1.659 | 39.0% | 1.08 | 71.5% |
| 1 | 1.649 | 38.1% | 1.07 | 72.4% |
| 2 | 1.673 | 38.0% | 1.09 | 71.6% |

v1 reported MAE 1.27 on its own (smaller, differently filtered) test split,
so these are not directly comparable, and v2 gives up some in-domain
precision by design: it no longer sees absolute board coordinates, and it
trains on noisy augmented embeddings. That trade is the point: v2 has to
work on photos.

## 4. Gym routes, photo-grouped 5-fold CV × 3 repeats (38 routes)

Crop mode = grey-masked crops; ensemble of the 3 seeds above.

| Strategy | MAE (V-grades) [95% CI] | bias | within ±1 | Spearman | P(beats v1) |
|---|---|---|---|---|---|
| v1 recorded predictions | 2.29 [1.71, 2.92] | +1.18 | 45% | 0.30 | — |
| constant median | 2.24 [1.76, 2.71] | +0.45 | 39% | — | |
| ridge on mean DINOv2 (PCA-8), gym-only | 2.03 [1.71, 2.34] | −0.13 | 37% | 0.35 | |
| v2 zero-shot (no gym training) | 2.26 [1.74, 2.84] | +1.47 | 42% | 0.35 | 0.53 |
| v2 + affine calibration | 2.03 [1.68, 2.39] | +0.03 | 34% | 0.33 | 0.85 |
| **v2 fine-tuned on gym folds** (proj. unfrozen, lr 1e-4) | **1.34 [0.97, 1.76]** | −0.24 | **61%** | **0.64** | **0.997** |
| v2 fine-tuned, projection frozen | 1.45 [1.08, 1.87] | −0.29 | 58% | 0.59 | 0.995 |
| v2 fine-tuned, raw crops | 1.39–1.45 | −0.2 | 55% | 0.70 | 0.996 |
| **Control: same fine-tuning from random init** | 1.92 [1.53, 2.32] | +0.03 | 42% | 0.43 | 0.88 |

Reading:
* Pre-training alone does not transfer (zero-shot ≈ constant predictor).
  The visual domain gap between renders and photos is too large for
  augmentation to close by itself.
* Fine-tuning on ~30 gym routes per fold **does** transfer, and it needs the
  Kilter pre-training: the random-init control is 0.58 grades worse and
  ranks routes much less well. Pre-training gives the network a prior over
  "what a route graph means" that 30 examples cannot teach.
* Letting the visual projection adapt helps (1.45 → 1.34), consistent with
  the diagnostic (`v2/diagnose_gym.py`) that the strongest single signal in
  gym photos is hold appearance, not geometry.
* Differences among the fine-tuned variants are inside the CI; the choice of
  the deployed variant (grey-masked crops, projection unfrozen) is a mild
  selection over 4 configs on the same CV. Fine-tuning hyper-parameters
  (lr 3e-4, 25 epochs, σ 1.5, 6 px jitter, mirror) were fixed a priori.

## 5. Deployed model

`v2/runs/adapt_gm_proj/deployment_v2.pt`: the 3 pre-trained members
fine-tuned on **all 38** labeled routes (same recipe), no affine correction
(the CV chose plain fine-tuning). Expected accuracy is the CV row above,
not the training-set fit. Copied to the Space as `models/deployment_v2.pt`;
inference code in `climb-app/src/inference/{v2.py,holds_v2.py,predict.py}`.

Sanity check on the two demo walls (both in the training set, so only a
plumbing check): rainbow overhang, gym V4 → V4 (V3–V6); red route, gym V7 →
V7 (V5–V8). Under v1 these were V9 and V6.

## 6. What would move the needle next

1. **More labeled gym routes.** 38 is the bottleneck; every "was it right?"
   answer on the site lands in the HF dataset. At ~150 routes the CI would
   shrink enough to tune the recipe honestly.
2. **Tension / Moonboard pre-training** for hold-shape diversity. The
   boardlib Tension download failed (APK mirror SSL); worth retrying from a
   different network.
3. **Kilter DB for absolute scale.** Re-adding physical scale (hold size
   from the mask, or board coordinates when known) should recover in-domain
   precision without hurting transfer.
4. Photo-domain hold embeddings: fine-tune or LoRA the DINOv2 backbone on
   real hold crops once there are a few thousand tapped holds.

## Reproduce

```bash
python v2/augment_embed.py --what both            # DINOv2 views (GPU, ~10 min)
python v2/train_v2.py --seed 0 --out v2/runs/v2_s0  # x3 seeds, ~7 min each
python v2/adapt_gym.py --runs v2/runs/v2_s0 v2/runs/v2_s1 v2/runs/v2_s2 \
    --key gray_mask --ft-proj-lr 1e-4 --out v2/runs/adapt_gm_proj
python v2/eval_gym.py                              # v1 baseline table
python v2/diagnose_gym.py                          # cheap-feature ceiling
```
