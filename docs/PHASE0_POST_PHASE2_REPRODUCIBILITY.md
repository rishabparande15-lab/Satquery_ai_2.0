# Phase 0 - Post-Phase-2 Scientific Reproducibility Rerun

## 1. Environment

The rerun used the repository scientific environment and did not modify the
scientific pipeline or its existing outputs.

| Field | Fresh rerun | Previously recorded environment |
| --- | --- | --- |
| OS | Windows-10-10.0.26200-SP0 | Windows-11-10.0.26200-SP0 |
| Python | 3.11.15 | 3.13.14 |
| NumPy | 2.4.6 | 2.4.6 |
| PyTorch | 2.14.0+cpu | Recorded baseline environment differed |
| CUDA | None; unavailable | Not used by recorded final inference |
| Device | CPU | CPU |
| GPU memory | 0 bytes / no GPU | Not applicable |
| Fresh evaluator runtime | 1.6092593999928795 seconds | Historical runtime not used for comparison |
| Warnings/errors during evaluator | None | N/A |

The worktree had pre-existing staged/uncommitted changes. They were not
modified, unstaged, or included in this rerun. All fresh outputs were written
under the new isolated directory `.tmp/phase0_post_phase2/`.

## 2. Dataset identity

The exact frozen Pipeline 3 manifest was validated before inference:

- Dataset: BigEarthNet v2 exact selected 5,000-area subset.
- Manifest: `experiments/pipeline3_5000/dataset_manifest.json`.
- Manifest SHA-256: `9f0446c011c1c054cea0fae5e7a2ccf196dcbdb50df6b7fc86757e764b0758be`.
- Dataset fingerprint: `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`.
- Manifest rows: 5,000.
- Validated areas: 5,000.
- Archive SHA-256: `b3471e5650bd263367bcf4cc650405ab2981a86621579e21a1b1d08af630c595`.

No acquisition, preprocessing, feature generation, or representation
regeneration was performed.

## 3. Split identity

- Split manifest: `experiments/pipeline3_5000/split_manifest.json`.
- Split fingerprint: `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`.
- Train: 4,600 areas.
- Validation: 200 areas.
- Test: 200 areas.
- Test-ID SHA-256: `dcdd9bf59a1fe36d6fb59f636914cc78aacec4611c230fdb5ac00bc8283fc9fd`.

The fresh test IDs and ordering were byte-for-byte equal to the historical
final prediction artifact.

## 4. Representation identity

The rerun used the existing verified scene-feature cache and did not regenerate
CROMA or physical features.

- Representation manifest: external frozen cache manifest recorded in
  `experiments/pipeline3_5000/reproducibility.json`.
- Representation manifest SHA-256:
  `3e37a44b0dd291b15f503a9e1b2b3c32ff9f80ac278fa414a8689ac740e101ed`.
- Status: `verified_existing_cache`.
- Feature shapes: physical `[5000, 62]`, joint CROMA GAP `[5000, 768]`, hybrid
  `[5000, 830]`.
- Feature order: physical dimensions `[0:62]`, then joint CROMA GAP dimensions
  `[62:830]`.
- Shards: 157.
- Typed receipts: 15,000.
- Representation catalog fingerprint:
  `0f5ce8feceff2fe8f2a75e1570b36edabf8a4716c57159243ffc6063eafcfd30`.
- Receipt catalog SHA-256:
  `3deb01c15ecfa76f99098d54333bff0d0ce831b0d3af536d6b85cf9aea3deece`.

The CROMA identity remained unchanged:

- CROMA source revision: `59505a6bcadbf36ba20767270154bf9f3067c5e7`.
- CROMA checkpoint SHA-256:
  `0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63`.
- Image resolution: 120.
- Modality: optical + SAR.

## 5. Checkpoint identity

The existing selected scientific checkpoint was loaded; no training occurred.

- Checkpoint: `D:\Satquery_ai datasets\comparison\pipeline3-5000\models\baseline\hybrid.pt`.
- File SHA-256:
  `0694dd82d4a3c03663ed7128adb4be6df729178e07b4ff8a1ec4295ed912130f`.
- Model state SHA-256:
  `85d9390c66a887276db24a7cadb58c398770cde2e17488a1a9c42e16819da634`.
- Model: linear softmax probe over the 830-dimensional hybrid.
- Selected epoch: 54.
- Selection seed: 17.
- Selection criterion: minimum validation soft-target cross entropy.
- Final configuration SHA-256:
  `32c8776f44a070fca265a8ae81d92fe4f22649b190af08f10fdff9eea2c9e80e`.

## 6. Inference procedure

The existing evaluation implementation was used unchanged:

`scripts/run_pipeline3_5000_baseline.py final-test`

The command consumed the frozen external cache and existing checkpoint, while
redirecting `metrics`, `predictions`, `fingerprint`, and `receipt` outputs to
`.tmp/phase0_post_phase2/`. The evaluator used the existing metric definition:

- frozen hybrid model only for the selected result;
- 200 held-out test areas;
- test truth from the verified feature cache;
- MAE and RMSE in percentage points;
- dominant-class accuracy using the existing evaluator;
- area bootstrap with 2,000 repeats and seed 17;
- independent metric recomputation from the saved fresh predictions.

No test result informed model selection.

## 7. Previous baseline

The requested comparison reference and historical artifacts record:

| Metric | Previous reference |
| --- | ---: |
| Accuracy | 65.0% |
| MAE | 4.1034286465 pp |
| RMSE | 9.5873311030 pp |
| Bootstrap 95% area-MAE CI | [3.8376, 4.3741] pp |
| Checkpoint | `hybrid.pt` |
| Scientific fingerprint | `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae` |

The historical exact final metrics artifact additionally stores the full
floating-point result `4.103428673440559` pp MAE and
`9.587331212294444` pp RMSE. The fresh evaluator matched those exact values.

## 8. Fresh rerun results

| Metric | Fresh result |
| --- | ---: |
| Accuracy | 65.0% |
| MAE | 4.103428673440559 pp |
| RMSE | 9.587331212294444 pp |
| Bootstrap 95% area-MAE CI | [3.837560135168402, 4.374123808228858] pp |
| Correct test areas | 130 / 200 |
| Runtime | 1.6092593999928795 seconds |

The fresh scientific fingerprint is exactly:

`ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`

## 9. Metric deltas

Deltas below compare the fresh exact evaluator to the requested rounded
reference values. Relative differences use the previous value as denominator.

| Metric | Previous | Fresh | Absolute difference | Relative difference | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Accuracy | 0.6500000000 | 0.6500000000 | 0.0 | 0.0 | PASS |
| MAE pp | 4.1034286465 | 4.103428673440559 | 2.6940559e-8 | 6.565e-9 | PASS |
| RMSE pp | 9.5873311030 | 9.587331212294444 | 1.09294444e-7 | 1.140e-8 | PASS |

All deltas are within the required tolerances: exact accuracy equality and
`1e-6` absolute tolerance for MAE/RMSE.

## 10. Determinism and reproducibility assessment

The rerun provides stronger evidence than metric agreement alone:

- Fresh and historical test-ID arrays are identical.
- Fresh and historical truth arrays are identical.
- Fresh and historical hybrid prediction arrays are exactly equal.
- Maximum absolute prediction difference is `0.0`.
- Fresh and historical per-class diagnostics are equal.
- Independent MAE/RMSE/accuracy recomputation agrees with the fresh evaluator
  at maximum absolute difference `0.0`.
- Fresh and historical exact evaluator metrics are equal.
- The fresh scientific fingerprint equals the historical fingerprint.

The runtime environment is not byte-for-byte the same as the historical record:
Python is 3.11.15 versus 3.13.14, and the platform label differs. This is a
non-scientific runtime difference because the frozen predictions and all
scientific metrics are unchanged.

## 11. Discrepancy investigation

No scientific discrepancy was found. The only differences are:

1. Fresh output files contain a new runtime duration and therefore have a
   different metrics-file checksum from the historical metrics file.
2. The current runtime reports CPU-only PyTorch with no CUDA device. The
   historical reproducibility record also describes the final inference as CPU,
   but was produced under a different Python/platform environment.
3. The requested rounded baseline values differ from the full-precision final
   evaluator values by approximately `2.7e-8` MAE and `1.1e-7` RMSE. Both are
   well below the `1e-6` tolerance, and the fresh predictions are exactly equal
   to the historical predictions.

There was no changed input, split, representation, preprocessing, checkpoint,
metric implementation, or nondeterministic prediction divergence.

## 12. FINAL GATE

### REPRODUCED WITH NON-SCIENTIFIC RUNTIME DIFFERENCE
