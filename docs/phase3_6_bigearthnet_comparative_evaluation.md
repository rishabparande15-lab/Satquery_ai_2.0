# Phase 3.6 — BigEarthNet Comparative Evaluation

## 1. Objective

Reproduce Ravi `Sat_Query` on its exact 1,000-area BigEarthNet v2 subset, evaluate the unchanged SatQuery HybridFusion representation on the same splits/targets, and reproduce BigEarthNet.txt linkage. This is a held-out coverage benchmark, not VQA accuracy or a deployment claim.

## 2. Repositories Audited

SatQuery AI commit `2bc9c888e7c4adb758abfab946fc7fe6e837a7dc` and Ravi `Sat_Query` commit `ac5541086c637ece2e9fd3c0a79cfa8bea23304d` were inspected. The audit covered downloading, TIFF construction, metadata, preprocessing, CROMA, targets, artifact joins, validation-selected training, evaluation, prediction export, reports, annotation matching, tests, and SatQuery Phases 1–3.5.

## 3. Dataset Source

The reference downloader used pinned BigEarthNet v2 metadata/reference-map archive revision `3cf3a5910a5302d449fdb8e570e5b78de24fe07f` and selective byte ranges from BigEarthNetV2-LMDB revision `118d1b6285c080ba8e4078414e1b8a243b18c9bd`. Seed 17 selects country-balanced records inside official splits, removes duplicate optical/SAR identities, and excludes the three demo IDs.

## 4. Dataset Extraction

No 1,000-area data existed locally. The reference downloader fetched only the controlled subset: ten verified ZIP shards containing 15,000 TIFFs. Archives were preserved under the external dataset root; extracted data and the combined metadata remain outside Git. The full 155 GB LMDB was not downloaded.

## 5. Canonical Dataset Contract

Optical order is `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12`; B10 is absent. SAR is `VV,VH`. B02 defines the 120×120 10 m grid. Reference maps are integer `[N,120,120]`. CROMA produces three spatial `[N,225,768]` and three scene `[N,768]` outputs. Tokens are row-major 15×15 blocks anchored to 8×8 pixels (80×80 m), while CROMA attention includes wider context.

## 6. Sample Identity

The generated canonical manifest records patch ID, explicit Sentinel-1/2 identities, official split, country, reference-map identity, every source path/hash, CRS, bounds, transform, resolution, acquisition timestamps, normalization profile, and CROMA source revision. Joins use identities and manifest hashes, never incidental directory order.

## 7. Train/Validation/Test Split

The exact split is 600 train, 200 validation, and 200 test areas. All blocks from an area remain together. Fully labelled blocks are 134,255 train, 44,848 validation, and 44,723 test. Unknown/excluded pixels are retained and partially/unlabelled blocks are excluded by the shared mask.

## 8. Ravi Pipeline Reconstruction

The actual reference path was run: strict native-grid validation → B02 alignment → per-image/channel mean ±2 sample-standard-deviation normalization → clipping → uint8 quantization → `/255` → frozen pinned CROMA Base → joint token encodings → official 19-class fractions → train-only standardization → `Linear(768,19)`.

Training used soft-target cross-entropy, AdamW, learning rate 0.001, weight decay 0.01, batch 1024, seed 17, at most 60 epochs and patience 10. Validation loss selected the checkpoint; test labels were untouched until final evaluation.

## 9. Ravi Reproduction Results

- Selected epoch: 5 of 15 run.
- Dominant-class accuracy: **53.8743%**.
- Coverage MAE: **6.1553 percentage points**.
- Training-mean baseline MAE: **9.2137 percentage points**.
- Error reduction versus constant baseline: **33.19%**.

## 10. Ravi Reported Results

The checked-in reference report contains the same 200 areas, 44,723 blocks, selected epoch 5, 53.8743% dominant accuracy and 6.1553 pp MAE (rounded in prose to 53.87% and 6.16 pp).

## 11. Ravi Reproduction Discrepancies

Primary metrics and epoch reproduce. Manifest/result byte hashes differ because artifacts were regenerated under current Windows/Python/PyTorch/Parquet serialization rather than copied from the historical Colab run. This does not alter identities, targets, predictions, or reported metrics. The first preprocessing publication attempt failed after processing because package distribution metadata was absent; installing the exact reference package metadata and rerunning resolved it with zero excluded areas.

## 12. SatQuery Pipeline

The same prepared raw tensors and CROMA exports feed `LocalRasterFeatureProvider`, producing the existing 62-value physical vector. Existing optical, SAR and joint GAP vectors concatenate to 2,304 values. The unchanged, deterministic seed-42 `HybridFusion(62,2304)` produces one 192-value representation per area.

## 13. SatQuery Representation

HybridFusion applies separate LayerNorm/Linear/GELU projections to physical and CROMA scene vectors, concatenates them, and applies its existing output Linear/LayerNorm/GELU block. Its weights are the current seeded untrained prototype. Because the representation is scene-level, the same 192 values are associated with every eligible block in an area. Ravi instead supplies a distinct 768-value joint CROMA token for each block. This is the dominant representation/task mismatch and a major limitation.

## 14. Evaluation Head

A benchmark-only `Linear(192,19)` probe was trained with the same soft-target loss, train-only scaling, AdamW settings, seed, batch size, validation selection, patience and metrics as Ravi. HybridFusion and CROMA stayed frozen. This probe is not the production SatQuery model.

## 15. SatQuery Results

- Selected epoch: 2 of 12 run.
- Dominant-class accuracy: **41.9170%**.
- Coverage MAE: **7.1989 pp**.
- Repeat run: bitwise-identical predictions and metrics on this CUDA environment.

## 16. Overall Comparison

SatQuery is **11.957 percentage points lower** in dominant accuracy and **1.0436 pp higher** in MAE. Relative MAE increased by **16.95%**. The current representation therefore does not outperform Ravi on the shared held-out coverage task.

## 17. Per-Class Comparison

| Class | Ravi MAE pp | SatQuery MAE pp | SatQuery − Ravi |
|---|---:|---:|---:|
| Urban fabric | 3.894 | 6.311 | +2.417 |
| Industrial/commercial | 1.670 | 2.061 | +0.391 |
| Arable land | 16.168 | 20.092 | +3.924 |
| Permanent crops | 2.093 | 2.202 | +0.109 |
| Pastures | 12.967 | 14.027 | +1.061 |
| Complex cultivation | 11.695 | 12.185 | +0.490 |
| Agriculture + natural vegetation | 8.441 | 9.262 | +0.822 |
| Agro-forestry | 2.408 | 2.537 | +0.129 |
| Broad-leaved forest | 17.113 | 21.591 | **+4.478** |
| Coniferous forest | 10.367 | 11.692 | +1.326 |
| Mixed forest | 14.436 | 16.324 | +1.888 |
| Natural grassland/sparse vegetation | 2.042 | 2.088 | +0.046 |
| Moors/heath/sclerophyll | 1.093 | 1.429 | +0.336 |
| Transitional woodland/shrub | 7.972 | 7.747 | **−0.225** |
| Beaches/dunes/sands | 0.431 | 0.696 | +0.265 |
| Inland wetlands | 1.247 | 1.277 | +0.030 |
| Coastal wetlands | 0.429 | 0.648 | +0.218 |
| Inland waters | 1.512 | 2.973 | +1.461 |
| Marine waters | 0.974 | 1.636 | +0.661 |

The best relative class outcome is transitional woodland/shrub; the worst regression is broad-leaved forest. Coastal wetlands has zero positive test support, and several rare classes cannot support strong conclusions.

## 18. Statistical Analysis

A paired 2,000-repeat percentile bootstrap resampled whole image areas. SatQuery-minus-Ravi MAE was +1.044 pp, 95% interval **[+0.818,+1.257]**. Accuracy difference was −0.1196, interval **[−0.1572,−0.0818]**. The sampled MAE difference never crossed zero. This supports a regression on this controlled subset, but does not establish general performance across the full BigEarthNet distribution.

## 19. Reproducibility

Two complete SatQuery probe fits using the same representations/configuration produced exactly equal predictions and zero metric difference on the tested CUDA/software environment. Cross-device bitwise equality is not claimed.

## 20. Failure Analysis

Dataset/raster, CROMA, physical-feature, HybridFusion, model and evaluation failures: zero. One resolved environment failure occurred when the source checkout lacked installed distribution metadata at export publication. A second report-finalization invocation needed the project root on `PYTHONPATH`; it did not affect scientific artifacts.

## 21. BigEarthNet.txt Extraction

The official pinned Parquet table at revision `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316`, SHA-256 `d3b97f…a1554`, was downloaded separately. It contains 9,553,962 source records. No imagery was downloaded again.

## 22. BigEarthNet.txt Matching

Both Sentinel identities were checked. Results exactly match Ravi's counts: 955 matched images, 45 missing; 20,453 records comprising 7,529 binary Q&A, 6,818 MCQ, 955 captions, 2,477 text-referenced boxes and 2,674 point-guided boxes. Missing annotations are not negative labels.

## 23. BigEarthNet.txt Split Distribution

Linked record partitions are 12,090 train, 4,175 validation, 4,159 test and 29 benchmark holdout. Image counts are 569 train, 197 validation, 188 test, one benchmark-held-out, plus 45 missing. No split conflicts were found.

## 24. Leakage Analysis

**PASS.** Patch IDs and Sentinel-1 identities are disjoint across image splits. Feature/target joins preserve manifest identities. Image-derived annotation records retain whole-image partitions. Scaling uses train only; epoch selection uses validation only; test data does not tune thresholds, weights, preprocessing statistics or epochs.

## 25. Runtime/Resource Comparison

On the available RTX 5060 Laptop GPU, reference head training/evaluation took about 27.7 seconds after feature materialization. SatQuery representation construction plus two reproducibility fits/evaluation took about 29.0 seconds as recorded by the probe run. Download, extraction and preprocessing are one-time data stages and are not used for unstable system-speed claims. Ravi uses 768 values per token; SatQuery uses one 192-value vector per area after consuming 62 physical and 2,304 pooled CROMA inputs.

## 26. Scientific Interpretation

Pipeline correctness is established for both systems. Predictive performance favors Ravi on this target. This does not show that physical features are intrinsically harmful; it shows that the current scene-pooled, untrained HybridFusion representation loses spatial information required for 80 m block coverage prediction. More inputs and a larger upstream vector do not imply a better representation.

## 27. Limitations

This is a country-balanced 1,000-area subset, not the full benchmark distribution. Classes are highly imbalanced, blocks within areas are correlated, some classes have tiny or zero support, and the SatQuery representation is scene-level while the target is token-level. HybridFusion is untrained and random-seeded; only the linear probe learns. Hardware/software differ from historical Colab even though Ravi metrics reproduced. BigEarthNet.txt artifact hashes vary with serialization, though all semantic counts match. No VLM task was evaluated.

## 28. Final Verdict

**YELLOW — TECHNICALLY VALID BUT SCIENTIFICALLY LIMITED.** The apples-to-apples data/split/target protocol is valid and reproducible, but SatQuery does not improve this task. Its current representation is structurally mismatched to spatial coverage targets.

| Item | Result |
|---|---|
| Dataset | Exact Ravi country-balanced BigEarthNet v2 subset |
| Dataset version | Pinned archive `3cf3a59`, LMDB `118d1b6` |
| Train areas | 600 |
| Validation areas | 200 |
| Test areas | 200 |
| Eligible test blocks | 44,723 |
| Ravi reported accuracy | 53.8743% |
| Ravi reproduced accuracy | 53.8743% |
| SatQuery accuracy | 41.9170% |
| Accuracy difference | −11.957 points |
| Ravi reported MAE | 6.1553 pp |
| Ravi reproduced MAE | 6.1553 pp |
| SatQuery MAE | 7.1989 pp |
| MAE difference | +1.0436 pp |
| Relative improvement | −16.95% (relative MAE regression) |
| Best SatQuery class | Transitional woodland/shrub relative to Ravi (−0.225 pp) |
| Worst SatQuery class | Broad-leaved forest relative to Ravi (+4.478 pp) |
| BigEarthNet.txt matched images | 955 |
| BigEarthNet.txt unmatched images | 45 |
| Total annotations matched | 20,453 |
| Leakage check | PASS |
| Reproducibility | EXACTLY REPRODUCIBLE in tested environment |
| Final verdict | YELLOW — TECHNICALLY VALID BUT SCIENTIFICALLY LIMITED |

## 29. Recommended Next Phase

Before VLM adaptation or added complexity, build a spatially preserving SatQuery evaluation path that fuses physical/geospatial evidence with each of the 225 CROMA tokens, specify it without inspecting test labels, and compare it on validation first. Reserve the existing test set from further architecture selection or obtain a fresh final holdout.
