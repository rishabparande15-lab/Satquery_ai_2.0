# Pass 5 — Leakage-Controlled SAR Generalization

## Dataset

The sole source was `bigearthnet-v2-5000-20260911T162804Z-1-001.zip`, SHA-256 `b3471e5650bd263367bcf4cc650405ab2981a86621579e21a1b1d08af630c595`. It was read without modification. The manifest verifies 5,000 metadata records, 5,000 VV/VH pairs, and 5,000 reference maps; 0 records were structurally invalid.

## Pass-3 Exclusion

All 1,000 Pass-3 areas were excluded using canonical area, Sentinel-1, and metadata identities. No Pass-3 area occurs in the Pass-5 test population.

## Pass-5 Split

Seed 53 produced a country-balanced 2,800 train / 600 validation / 600 test split from the 4,000 eligible areas. Each of the ten countries contributes 280/60/60. Splitting occurred before raster decoding, extraction, or training.

## Preprocessing

SAR is ordered VV, VH with shape `[2,120,120]`. Every experiment input was checked for decode, dimensions, dtype, finite values, CRS, bounds, north-up transform, VV/VH grid equality, and reference-grid equality. CROMA uses the existing official per-channel mean ± 2 sample-standard-deviation profile clipped to `[0,1]`.

## Target

The unchanged Pass-3 target code maps reference codes to the established 19-class order, excludes unlabeled codes, sums token class counts across the area, and normalizes labelled pixels to a `[19]` scene-coverage distribution. All 4,000 targets were finite and summed to one.

## Models

- Constant: Pass-5 training-target mean only.
- SAR physical: the mature existing SAR-only nine-feature `sar_v1` statistics contract and a trainable probe.
- SAR CROMA: official frozen CROMA base SAR encoder, `[225,768]` tokens, `[768]` GAP scene vector, and the existing trainable probe head.

## Training

Only the 2,800 training areas fit model parameters. The 600 validation areas selected epochs/early stopping. Test data was not used for selection. Saved configurations, logs, checkpoints, and validation metrics are under `experiments/pass5/sar_croma` and `sar_physical`.

## Test Protocol

All models used the identical 600-area held-out test set. Metrics are percentage-point MAE, RMSE, bias, dominant-class accuracy, per-class metrics, and a 2,000-repeat whole-area bootstrap interval.

## Results

| Model | MAE pp | RMSE pp | Bias pp | Dominant accuracy | MAE 95% area-bootstrap CI |
|---|---:|---:|---:|---:|---:|
| Constant | 7.8588 | 15.5361 | 0.0000 | 21.33% | [7.7649, 7.9461] |
| SAR physical | 7.2288 | 14.0275 | 0.0000 | 35.83% | [7.0947, 7.3609] |
| SAR CROMA | **4.9280** | **11.5264** | -0.0000 | **54.00%** | **[4.7306, 5.1232]** |

SAR CROMA improves MAE by 2.9308 pp (37.3%) versus the Pass-5 constant and by 2.3008 pp (31.8%) versus SAR physical. The area-bootstrap intervals are well separated. This supports meaningful generalization on this sampled population, not universal geographic generalization.

## Per-Area / Geographic Analysis

Country-level SAR-CROMA MAE ranges from 3.8706 pp (Ireland) to 6.5917 pp (Portugal), with 60 independent test areas per country. It beats the constant in all ten country groups. These groups support a consistency check, but country effects are descriptive because the split was not designed for causal country comparisons.

## Leakage Audit

All Pass-3/train, Pass-3/validation, Pass-3/test versus Pass-5/test intersections are empty. Pass-5 internal intersections are also empty. See `docs/pass5_leakage_audit.md`.

## Reproducibility

Saved numeric bundles reload with SHA-256 verification. Independent metrics recomputed from saved predictions and targets differ by `0.0`. Reloading the saved SAR-CROMA head reproduces all 600 test predictions exactly. Repeating raw-SAR → CROMA → saved head on test area `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_83_67` also produced maximum absolute prediction difference `0.0` and identical prediction hashes.

Failure injection explicitly rejected missing VV, missing VH, corrupt TIFF, wrong dimensions, NaN, Inf, mismatched reference grid, absent CRS, invalid orientation/bounds, and duplicate identity.

## Performance

On an NVIDIA GeForce RTX 5060 Laptop GPU, exhaustive loading/target validation took 23.09 s, CROMA forward time 0.46 s, and the full extraction phase 43.71 s. Probe fitting took 1.80 s for SAR CROMA and 0.88 s for SAR physical. Persisted Pass-5 artifacts occupy about 22 MB. Peak memory was not instrumented.

## Comparison to Pass 3

Pass-3 SAR CROMA was 5.6489 pp MAE; Pass-5 is 4.9280 pp, a descriptive improvement of 0.7209 pp. The experiments are not interchangeable: Pass 5 uses a larger new population and different split while preserving preprocessing and target meaning. No Pass-3 result was used for model selection or modified.

## Limitations

The archive is a curated ten-country selection rather than an unrestricted global sample; all countries are European. The test is independently area-held-out relative to Pass 3, but nearby or correlated acquisitions were not grouped into larger spatial clusters because authoritative higher-level grouping was unavailable. Rare-class country estimates remain unstable. The encoder is frozen and only its scene head is trained. Peak memory was not measured.

## Conclusion

**GREEN — VALIDATED.** On 600 genuinely new held-out areas, SAR CROMA materially and consistently outperforms both the Pass-5 train-only constant and SAR physical baselines. Leakage, independent metric recomputation, artifact reload/hash verification, and end-to-end reproducibility checks pass. The claim is limited to scene-level 19-class labelled-pixel coverage on this selected European population.
