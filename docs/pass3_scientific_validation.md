# Pass 3 Scientific Validation

Date: 2026-09-11

## Dataset

The experiment used the accessible controlled BigEarthNet v2 subset at `D:\Satquery_ai datasets\comparison\raw-1000`. Strict discovery found 1,000 areas with 12 optical bands, VV/VH SAR, integer reference maps, and metadata for every area. All 1,000 areas passed the existing strict loader, including CRS, orientation, bounds, finite-value, grid-alignment, reference, and tensor checks.

The validated machine-readable manifest is `experiments/pass3/dataset_manifest.json`. Existing prepared, CROMA-feature, and target artifacts were reused from `D:\Satquery_ai datasets\comparison\pipeline-1000`; their validation receipts report 1,000 aligned samples, six finite CROMA outputs per sample, and 225-token targets. CROMA feature extraction was not recomputed in this pass because the validated cached artifacts were available and traceable.

## Split

The canonical geographic-area split is in `experiments/pass3/split.json`:

| Split | Areas |
|---|---:|
| Train | 600 |
| Validation | 200 |
| Test | 200 |

The split comes from the official metadata split, uses area identity as the unit, and has zero train/validation, train/test, or validation/test overlap. No spatial blocks from one area were placed in different splits.

## Preprocessing

All representations use the same existing contracts:

- Optical order: `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12`.
- SAR order: `VV,VH`.
- B02 defines the 120x120 common grid.
- CROMA uses the validated `croma_readme_patch_8bit_v1` profile.
- Physical features are the existing 62-feature provider output.
- Probe standardization is fit on training areas only and folded into each saved linear head.

## Target

This Pass 3 comparison evaluates the documented scene-level coverage task, not token-level spatial grounding. For each area, the 225 token class counts are summed by class and normalized over labelled pixels, producing one `[19]` target vector. Unknown/excluded pixels are omitted from the denominator and are not reassigned.

This is distinct from the historical token-level `[225,19]` benchmark. It is the correct target for comparing scene-level physical, GAP CROMA, and concatenated physical+CROMA representations.

## Models

Six models used the same areas, targets, metric implementation, seed, optimizer, batch size, and validation selection:

1. Constant: training-area mean target only.
2. Physical: existing 62 physical features.
3. Optical CROMA: optical GAP `[768]`.
4. SAR CROMA: SAR GAP `[768]`.
5. Joint CROMA: joint GAP `[768]`.
6. Hybrid: genuine feature concatenation `[62 physical + 768 joint CROMA] = [830]`.

CROMA representations remained frozen. Each trainable model used an AdamW linear 19-class soft-target head, learning rate `0.001`, weight decay `0.01`, batch size `1024`, seed `17`, maximum 60 epochs, patience 10, and validation soft-target cross-entropy for checkpoint selection.

## Training

Saved artifacts are under `experiments/pass3/comparison/`. Each trainable model contains `config.json`, `checkpoint.pt`, and `test_predictions.json`; the root contains `metrics.json`, `predictions.json`, and `provenance.json`.

Selected epochs were physical 60, optical CROMA 52, SAR CROMA 60, joint CROMA 60, and hybrid 60. Reaching the configured maximum is recorded; it is not evidence that further epochs would improve test performance.

## Test Protocol

The test set was evaluated only after validation selection. Metrics are MAE, RMSE, and bias in percentage points plus dominant-class accuracy. Whole-area bootstrap intervals used 2,000 repeats with seed 17. Independent metrics were recomputed from saved test predictions and saved test targets.

## Results

| Model | MAE (pp) | RMSE (pp) | Bias (pp) | Dominant accuracy | Relative MAE reduction |
|---|---:|---:|---:|---:|---:|
| Constant | 7.6984 | 15.1128 | 0.0000 | 22.0% | 0.00% |
| Physical | 7.7384 | 14.7769 | 0.0000 | 33.0% | -0.52% |
| Optical CROMA | 5.5708 | 12.1651 | 0.0000 | 51.0% | 27.64% |
| SAR CROMA | 5.6489 | 12.2389 | 0.0000 | 50.5% | 26.62% |
| Joint CROMA | 4.9815 | 10.9457 | 0.0000 | 60.5% | 35.29% |
| Hybrid | 4.8846 | 11.0251 | 0.0000 | 59.5% | 36.55% |

The hybrid has the lowest MAE, but joint CROMA has slightly lower RMSE and higher dominant accuracy. Physical-only does not beat the constant baseline on MAE, although its dominant accuracy is higher.

## Per-Class Results

The strongest broad-support outcomes were produced by hybrid for arable land (11.5219 pp MAE), pastures (8.6846 pp), broad-leaved forest (13.1955 pp), and transitional woodland/shrub (5.7693 pp). Optical CROMA was best for urban fabric (2.9605 pp) and complex cultivation patterns (8.4832 pp). Joint CROMA was best for coniferous forest (7.7198 pp), mixed forest (11.5920 pp), and inland waters (1.6724 pp).

Support is uneven: test-area positive support ranged from zero for coastal wetlands to 103 for arable land. Results for beaches/dunes (1), inland wetlands (2), permanent crops (4), moors/heath (4), and marine waters (6) are weakly supported. Per-class values for every model are retained in `experiments/pass3/comparison/metrics.json`.

## Statistical Analysis

The test set contains 200 geographic areas, which is materially stronger than the earlier six-area smoke run. Whole-area bootstrap intervals are stored in the model metric artifacts. The experiment supports a controlled held-out comparison on this country-balanced subset, but it does not establish performance over the full BigEarthNet distribution. Rare-class inference remains limited by support.

## Leakage Audit

No area identity appears in more than one split. The constant baseline uses only train targets. Probe scaling uses train features only. Validation loss selects epochs; test metrics are not used for training, scaling, architecture selection, or checkpoint selection. Feature and target artifacts were joined by batch metadata and area identity, not incidental file order. Detailed checks are in `docs/pass3_leakage_audit.md`.

## Reproducibility

A complete repeat run using the same dataset manifest, artifact manifests, split, seed, configuration, and CUDA environment produced identical test IDs, truth, predictions, and trainable checkpoint bytes for all five trainable models. Independent metric recomputation differed from stored metrics by exactly `0.0`.

## Performance

The comparison run took 5.17 seconds after cached feature materialization on Windows 11 with Python 3.13.14, PyTorch 2.11.0+cu128, CUDA 12.8, and an NVIDIA RTX 5060 Laptop GPU. The persisted comparison artifacts occupy approximately 2.86 MB. This timing excludes the one-time raw preparation and CROMA extraction stages because those validated artifacts were reused.

## Interpretation

1. Physical features did not outperform the constant baseline on scene MAE.
2. Optical and SAR CROMA both outperformed the constant baseline.
3. Joint CROMA outperformed both individual CROMA sensors.
4. The genuine physical+joint-CROMA hybrid had the best MAE, but not the best RMSE or dominant accuracy.
5. The comparison does not show that physical features are useless; it evaluates the current 62-feature scene summary under this target and protocol.
6. The result is valid for the controlled 1,000-area subset and scene-level coverage task, not a claim about VQA, grounding, temporal change, or the full BigEarthNet distribution.

## Limitations

- The subset is country-balanced and controlled, not the full BigEarthNet distribution.
- CROMA feature extraction was reused from validated cached artifacts rather than rerun in this pass.
- The hybrid is scene-level concatenation and does not establish token-level spatial grounding.
- Several classes have low or zero positive test support.
- Cross-device bitwise equality is not claimed.
- No calibrated confidence, VQA, captioning, learned grounding, or temporal model was evaluated.

## Conclusion

The current scene-level physical/CROMA coverage comparison is scientifically reproducible and leakage-audited on 600/200/200 geographic areas. Joint CROMA and the feature-level hybrid clearly outperform the constant baseline on MAE, with the hybrid best on MAE but joint CROMA best on RMSE and dominant accuracy. The appropriate project status remains **YELLOW — PARTIALLY VALIDATED**, because the experiment is limited to a controlled subset/task and does not validate the broader agentic product capabilities.
