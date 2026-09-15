# Pipeline 3 Exact 5,000-Area Training Report

Date: 2026-09-15

## Executive summary

The exact verified 5,000-area BigEarthNet v2 multimodal population was used
without changing any identity, split, target, sample weight, or test protocol.
The established Pipeline 3 scene-coverage comparison was reproduced with 4,600
training areas and 200 validation areas. The 200 test areas remained sealed
through configuration selection, bottleneck triage, and the seed repeat.

The predeclared selection rule (minimum validation soft-target cross-entropy)
selected the 830-dimensional physical+joint-CROMA hybrid at epoch 54 with seed
17. No repair was scientifically justified. A predetermined seed-29 repeat was
stable. After the final configuration was frozen, one test event scored the
constant and five established learned ablations together.

The final hybrid obtained **4.1034 percentage-point MAE**, **9.5873 pp RMSE**,
and **65.0% dominant-class accuracy** on the 200 held-out areas. Its whole-area
bootstrap MAE 95% interval is **[3.8376, 4.3741] pp**. Relative to the historical
1,000-area hybrid, this is 0.7812 pp lower MAE, 1.4378 pp lower RMSE, and 5.5
percentage points higher dominant-class accuracy.

Scientific verdict: **A — HEALTHY**. The current Pipeline 3 scene-level
experiment is internally valid, reproducible at the tested seeds, and improves
materially with the larger training population. This does not validate VQA,
captioning, grounding, temporal reasoning, or the live untrained 192-vector as
a predictor.

## Exact dataset identity and split

- Dataset: BigEarthNet v2 exact selected 5,000-area subset.
- Complete multimodal rows: 5,000/5,000.
- Split: 4,600 train, 200 validation, 200 test.
- Dataset fingerprint: `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`.
- Split fingerprint: `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`.
- Dataset-manifest SHA-256: `9f0446c011c1c054cea0fae5e7a2ccf196dcbdb50df6b7fc86757e764b0758be`.

The verified audit found zero cross-split identity, S1-content,
reference-content, S2-area-content, or positive spatial-overlap leakage. The
138 overlapping pairs and 103 duplicate-reference groups occur only within
training and were retained; no difficult or repeated samples were excluded.

## Scientific contract

Optical order is `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12`; SAR order
is `VV,VH`. B02 defines the north-up 120x120, 10 m common grid. Optical native
10/20/60 m rasters are converted to float32 and bilinearly resampled. Reference
maps are read categorically with nearest semantics and finite integer codes.
CROMA uses the current `CROMAAdapter._croma_normalize`: per-image/channel mean
± two sample standard deviations, clipped to [0,1], with constant channels set
to zero.

CROMA is the official Base model at source revision
`59505a6bcadbf36ba20767270154bf9f3067c5e7`; checkpoint SHA-256 is
`0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63`.
Feature extraction performed the load-time shape, mask, finite-value, CRS,
extent, and grid assertions required for computation. It did not repeat the
exhaustive acquisition/raster-content audit.

Each reference is mapped to the established 19 classes. The 225 token
histograms are summed to one scene target and normalized over labeled pixels
only. Unknown/excluded pixels are omitted, not redistributed.

| Ablation | Input dimension |
|---|---:|
| Constant | train-target mean |
| Physical | 62 |
| Optical CROMA GAP | 768 |
| SAR CROMA GAP | 768 |
| Joint CROMA GAP | 768 |
| Scientific hybrid | 62 physical + 768 joint CROMA = 830 |

The live `HybridFusion(62,2304)` emits a 192-vector using seeded, untrained
projection weights. It is representation-only and was not represented as a
learned predictor. The evaluated 830-vector is the established controlled Pass
3 feature-concatenation hybrid.

Every learned ablation used a single linear 19-output head, softmax prediction,
and soft-target cross-entropy. CROMA stayed frozen. Optimization used AdamW,
learning rate 0.001, weight decay 0.01, batch size 1,024, maximum 60 epochs,
patience 10, and seed 17. Training-only mean/std standardization was folded
into each head. There was no scheduler, augmentation, or class weighting.

The configuration was frozen before fitting; its SHA-256 is
`d99d1726e7668a534496732ce27afe5f9d053350d43b5546b64bd07d22d540f1`.

## Baseline training and validation

| Model | Epoch | Train MAE | Val MAE | Train RMSE | Val RMSE | Train accuracy | Val accuracy | Best val CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Constant | — | 7.9350 | 7.8628 | 15.8952 | 15.7804 | 22.4% | 20.0% | 2.459371 |
| Physical | 60 | 5.9318 | 6.1945 | 12.4119 | 12.8150 | 50.7% | 51.0% | 1.930609 |
| Optical CROMA | 57 | 4.0570 | 4.8284 | 9.4058 | 11.4467 | 68.1% | 58.0% | 1.687452 |
| SAR CROMA | 58 | 4.6551 | 5.2004 | 10.6867 | 12.0028 | 61.7% | 54.5% | 1.766388 |
| Joint CROMA | 57 | 3.6386 | 4.5085 | 8.6645 | 10.8762 | 71.6% | 61.5% | 1.549628 |
| Hybrid | 54 | **3.4706** | **4.4064** | **8.3456** | **10.8373** | **72.4%** | 60.0% | **1.526572** |

The selected hybrid train/validation gap is 0.9358 pp MAE. CROMA-based models
selected epochs 54–58 and plateaued near the cap. Physical-only was improving
at epoch 60 but remained far behind hybrid. Histories are retained in
`experiments/pipeline3_5000/baseline/history.json`.

## Dataset and label analysis

Development analysis loaded only train and validation rows. The split is
country-balanced across ten countries: each has 460 train and 20 validation
areas. Mean target entropy is 0.7330 nats in train and 0.7429 in validation.

Class support is strongly imbalanced. Training positive-area support ranges
from 8 coastal-wetland and 10 beach areas to 2,162 arable-land areas.
Validation support ranges from 1 for several rare classes to 101 for arable
land. This limits rare-class inference. Class weighting was not introduced
without a controlled validation hypothesis because it would change the
objective and might degrade the stated macro-over-class coverage MAE.

Repeated training geography may slightly overweight some places but cannot
cause held-out leakage. With no predeclared group-weighting experiment and no
selection evidence for one, all areas retained equal weight.

## Bottleneck triage and repair decision

| Priority | Finding | Decision |
|---|---|---|
| P0 | No correctness defect after manifest, raster-load, target, split, and independent-metric checks | No repair |
| P1 | No demonstrated major predictive limitation; joint CROMA and hybrid generalize strongly | No repair |
| P2 | Physical-only still improves at epoch 60; rare-class support remains limited | Record limitation |
| P3 | One-time CROMA extraction dominates cost; source ZIP cache dominates storage | Keep resumable shards |
| P4 | Live untrained 192-vector and evaluated 830-vector can be conflated | Document explicitly |

The ablations show that physical improves the constant; each single-sensor
CROMA improves further; joint CROMA improves both sensors; and physical plus
joint CROMA provides the best validation loss/MAE. This supports complementary
information rather than a predictor-capacity bottleneck.

Problem: decide whether behavior warrants a scientific change. Evidence: the
winner selected epoch 54, beat joint CROMA, and showed no instability.
Hypothesis: it converged under the established schedule. Change: none. Risk of
a change: validation overfit and loss of comparability. Validation plan: repeat
the same hybrid using predetermined seed 29. Result: stable. Blind schedule,
loss, head, or fusion changes were not performed.

## Repeatability

| Run | Epoch | Validation CE | Validation MAE | Validation accuracy |
|---|---:|---:|---:|---:|
| Primary seed 17 | 54 | 1.526572 | 4.4064 | 60.0% |
| Repeat seed 29 | 59 | 1.528328 | 4.3811 | 58.5% |

The repeat differs by −0.0253 pp MAE and +0.00176 CE. Seed 17 remained primary
because seeds were not selection candidates and its CE is the criterion.

## Final frozen test evaluation

The final configuration was frozen before test access. A receipt prevents a
second execution. Independent recomputation differed by exactly 0.0.

| Model | Test MAE | Test RMSE | Accuracy | Test−validation MAE |
|---|---:|---:|---:|---:|
| Constant | 7.6803 | 15.0932 | 22.0% | −0.1825 |
| Physical | 6.3836 | 12.9160 | 39.0% | +0.1891 |
| Optical CROMA | 4.7686 | 10.7049 | 60.5% | −0.0598 |
| SAR CROMA | 5.1967 | 11.5806 | 51.0% | −0.0038 |
| Joint CROMA | 4.2690 | 9.8306 | 62.5% | −0.2395 |
| Hybrid | **4.1034** | **9.5873** | **65.0%** | **−0.3030** |

Hybrid whole-area bootstrap MAE 95% interval: [3.8376, 4.3741] pp. The negative
final gap was seen only after freezing and influenced no decision.

## Final hybrid per-class results

Support is the number of test areas with nonzero target coverage.

| Class | Support | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Urban fabric | 55 | 2.6090 | 4.7307 | +0.3246 |
| Industrial/commercial | 13 | 0.8206 | 2.5746 | −0.0466 |
| Arable land | 103 | 9.2886 | 15.5603 | −4.0022 |
| Permanent crops | 4 | 1.6654 | 5.7910 | +0.5536 |
| Pastures | 76 | 7.5813 | 12.4272 | −0.6677 |
| Complex cultivation | 50 | 9.0610 | 14.1315 | +3.0770 |
| Agriculture with natural vegetation | 54 | 6.3761 | 9.6558 | +0.4618 |
| Agro-forestry | 7 | 1.8968 | 7.1369 | +0.3848 |
| Broad-leaved forest | 96 | 10.6979 | 17.7370 | −1.7104 |
| Coniferous forest | 45 | 6.6377 | 12.1951 | +3.0956 |
| Mixed forest | 79 | 10.2427 | 17.6869 | −3.0017 |
| Natural grassland/sparse vegetation | 7 | 1.5559 | 4.0765 | +0.4321 |
| Moors/heath/sclerophyllous | 4 | 1.0637 | 4.1049 | +0.2966 |
| Transitional woodland/shrub | 61 | 5.2876 | 10.6518 | −0.6900 |
| Beaches/dunes/sands | 1 | 0.3332 | 0.4715 | +0.3332 |
| Inland wetlands | 2 | 0.7621 | 1.6866 | +0.6463 |
| Coastal wetlands | 0 | 0.3425 | 0.5049 | +0.3425 |
| Inland waters | 21 | 1.1339 | 3.4144 | −0.2453 |
| Marine waters | 6 | 0.6093 | 2.3218 | +0.4157 |

Largest errors occur in broad-leaved forest, mixed forest, arable land, and
complex cultivation. Rare-class MAEs can appear small because most areas have
zero coverage; they do not prove strong rare-class detection. Full confusion
matrices are in `experiments/pipeline3_5000/final/ablations.json`.

## Historical comparison

| Metric | Prior 1,000 hybrid | 5,000 baseline/final | Difference |
|---|---:|---:|---:|
| MAE | 4.8846 | 4.1034 | −0.7812 |
| RMSE | 11.0251 | 9.5873 | −1.4378 |
| Dominant accuracy | 59.5% | 65.0% | +5.5 pp |

The final MAE interval does not include the old point estimate, consistent with
a meaningful improvement. This is contextual, not a paired significance test:
training populations differ and the historical value is itself an estimate.

## Reproducibility, compute, and limitations

- Code commit at start: `831b68b1825fc145ea6772de5dedf8770f2f9ccc` with a recorded dirty worktree.
- Selected checkpoint file SHA-256: `0694dd82d4a3c03663ed7128adb4be6df729178e07b4ff8a1ec4295ed912130f`.
- Selected normalized state SHA-256: `85d9390c66a887276db24a7cadb58c398770cde2e17488a1a9c42e16819da634`.
- Final-config SHA-256: `32c8776f44a070fca265a8ae81d92fe4f22649b190af08f10fdff9eea2c9e80e`.
- Baseline fingerprint: `9e7aa5977c6bc2f1f7671b2f5955c4414fda7d426c155e02be2ae1b183e77f4a`.
- Final fingerprint: `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`.
- Feature extraction/load validation: 910.7 s; peak allocated GPU memory 904,975,360 bytes.
- Five baseline fits/evaluation: 91.8 s; peak allocated GPU memory 38,397,952 bytes.
- Seed-29 repeat: 20.5 s; final test event: 1.43 s.
- Feature shards: 44,280,954 bytes; source cache: 511,289,395 bytes; checkpoints: 319,342 bytes.

The final fingerprint excludes timing, paths, and run IDs, but includes
scientific identities, config, model state, seed, class order, and metrics.
Weights and imagery remain outside Git.

This result is authoritative for the exact selected scene-level 19-class
coverage task. It does not validate token grounding or other product
capabilities. Several rare classes cannot support strong conclusions, and two
seeds do not characterize every variance source. Within those limits, the
experiment is healthy and the next architecture/research phase may proceed.
