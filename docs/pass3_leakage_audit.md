# Pass 3 Leakage Audit

Date: 2026-09-11

## Dataset and Identity

- Dataset: `D:\Satquery_ai datasets\comparison\raw-1000`.
- Validated areas: 1,000/1,000.
- Prepared, feature, and target artifacts each report 1,000 samples and passed their validation receipts.
- Area joins were checked against batch metadata and the Pass 3 manifest.
- Optical, SAR, and reference identities were preserved per area.

## Split Isolation

| Split | Areas |
|---|---:|
| Train | 600 |
| Validation | 200 |
| Test | 200 |

The split unit is the geographic area. Train/validation overlap, train/test overlap, and validation/test overlap are all empty. No spatial block from an area is present in more than one split.

## Target and Baseline

The constant baseline is the mean of train-area scene targets only. Validation and test targets do not influence it. Scene targets are produced from the existing reference-map class counts; excluded/unknown pixels are omitted consistently.

## Features and Preprocessing

- CROMA features came from the validated shared feature artifact manifest and were not recomputed or modified during fitting.
- Physical features were recomputed from the aligned cached raw tensors using the existing 62-feature provider.
- Probe feature standardization uses train-only mean and population standard deviation.
- The folded normalization is stored in each model configuration/checkpoint path through the established probe protocol.
- No target values are used to construct physical or CROMA features.

## Model Selection

All trainable models use the same AdamW/soft-target linear-head protocol. Epoch selection uses validation soft-target cross-entropy only. The test set is not passed to `fit_probe`, and test metrics are computed only after the selected checkpoint is restored.

## Artifact Pairing

Each prepared, feature, and target batch was required to have identical ordered `patch_id` values before loading. The Pass 3 manifest also requires every area to be valid and present in the metadata table. The prepared, feature, and target manifest hashes are recorded in `experiments/pass3/dataset_manifest.json`.

## Independent Metric Check

Saved test predictions and targets were reloaded and metrics were calculated independently. Maximum absolute difference for MAE, RMSE, bias, and dominant-class accuracy was `0.0`.

## Reproducibility Check

A complete repeat run with the same manifest, seed, configuration, and device produced:

- identical test area IDs;
- identical test truth;
- bitwise-identical predictions for constant, physical, optical CROMA, SAR CROMA, joint CROMA, and hybrid;
- byte-identical trainable checkpoints.

## Result

**PASS for the reported scene-level comparison.** The audit does not imply that the historical token-level benchmark, VQA, grounding, temporal prediction, or calibration tasks are validated by this run.
