# SatQuery AI Scientific Evaluation Report

Date: 2026-09-11

## Dataset and split

The controlled experiment uses the pinned BigEarthNet v2 1,000-area subset already materialized outside Git under `D:\Satquery_ai datasets\comparison`. The immutable area-level split is 600 train, 200 validation, and 200 test areas. Eligible fully labelled blocks are 134,255 train, 44,848 validation, and 44,723 test. Patch and Sentinel-1 identities were checked for cross-split overlap; none was reported.

## Preprocessing and CROMA

Optical order is `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12`; SAR order is `VV,VH`. B02 defines the 120x120 common grid. The pinned profile applies per-image/channel mean +/- two sample standard deviations, clipping, 8-bit quantization, and division by 255. Frozen CROMA Base produces optical, SAR, and joint `[N,225,768]` tensors plus three `[N,768]` scene vectors.

## Target and model

Each reference map is divided row-major into 225 non-overlapping 8x8 blocks. The target is the 19-class pixel coverage fraction. Blocks containing excluded/unknown pixels are retained in target artifacts but excluded from the reported fully-labelled evaluation mask. The reference model is frozen joint CROMA followed by a trainable `Linear(768,19)` and softmax. Training uses soft-target cross-entropy, AdamW, learning rate 0.001, weight decay 0.01, batch size 1024, seed 17, validation-loss checkpoint selection, and patience 10. Test labels are not used for epoch selection.

## Held-out results

| Model | Selected epoch | Dominant accuracy | MAE (pp) | RMSE | Bias |
|---|---:|---:|---:|---:|---:|
| Constant training mean | n/a | not recorded in consolidated artifact | 9.2137 | not recorded | not recorded |
| Frozen joint CROMA + linear coverage head | 5 | 53.8743% | 6.1553 | available in detailed external report | available in detailed external report |
| Existing SatQuery scene-level hybrid + linear probe | 2 | 41.9170% | 7.1989 | available in detailed external report | available in detailed external report |

The reference coverage model reduces MAE by 3.0584 percentage points, or 33.19%, versus the constant baseline. The existing SatQuery scene-level hybrid is 1.0436 pp worse than token-level joint CROMA and 11.957 percentage points lower in dominant accuracy. A whole-area bootstrap reported SatQuery-minus-reference MAE +1.044 pp with a 95% interval of `[+0.818,+1.257]`.

No unrecorded metric is reconstructed or invented in this document. Detailed per-class results remain in `docs/phase3_6_bigearthnet_comparative_evaluation.md`.

## Fresh verification in this audit

The real sample `61_39` was re-run through local discovery, raster materialization, Phase 3 selection/provenance, Phase 1 preprocessing, official frozen CROMA, Phase 2 physical features, and the existing untrained hybrid representation. Observed shapes were optical/SAR/joint CROMA `[1,225,768]`, all scene outputs `[1,768]`, physical `[1,62]`, pooled CROMA `[1,2304]`, and hybrid `[1,192]`. The artifact bundle was saved and hash/reload verified under `experiments/outputs/final_audit/sample-61_39`.

This fresh run does not include a reference target, training epoch, or held-out prediction and is therefore pipeline-integration evidence, not a new scientific evaluation.

## Annotation linkage

The pinned BigEarthNet.txt extraction contains 9,553,962 source records. On the controlled subset, 955 images matched and 45 did not; 20,453 linked records comprise 7,529 binary Q&A, 6,818 MCQ, 955 captions, 2,477 text-referenced boxes, and 2,674 point-guided boxes. This is data linkage only, not VLM training or evaluation.

## Limitations and reproducibility

- The evaluated subset is controlled and country-balanced, not the full BigEarthNet distribution.
- Rare classes have limited or zero positive test support.
- The current SatQuery hybrid is scene-level and untrained, making it structurally mismatched to 80 m token coverage.
- A token-aligned physical/CROMA hybrid has not been validated on a fresh final holdout.
- The external 1,000-area artifact directories were permission-inaccessible during this fresh audit, so the larger experiment was not rerun; reported values are the existing checked-in executed report, not newly recomputed values.
- No VQA, captioning, learned grounding, temporal prediction, or calibrated confidence is claimed.

Reproduction commands and external asset locations are documented in the README and the Phase 3.6 report. Cross-device bitwise equality is not claimed.
