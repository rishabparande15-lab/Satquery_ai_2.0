# Phase 6R.2 Pipeline 3 5,000-Area Revalidation

Date: 2026-09-16  
Repository commit: `2f7b241e4f4687d2aae263faf4b1d5c08eae9e5e`

## 1. Executive verdict

**RED**

The required fresh 5,000-area scientific revalidation was stopped before
feature extraction, inference, or training. The independent audit found only
1,000 of the 5,000 selected areas with complete optical S2 data. The missing
4,000 areas are all in the training split. The exact 5,000-area scientific
input therefore does not exist in the currently available local raster roots.

No fresh 5,000-area metrics are reported. Historical metrics are not treated as
fresh reproduction results.

## 2. Environment

The repaired environment used for the audit was:

| Component | Observed value |
|---|---|
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| PyTorch | 2.14.0+cpu |
| PyTorch CUDA | `None` |
| CUDA availability | `False` |
| GPU count | `0` |
| Rasterio | 1.4.4 |
| pandas | 3.0.5 |
| pyarrow | 25.0.1 |
| CUDA tensor test | CPU fallback only; no CUDA device available |

The environment imports successfully, but this host does not provide a CUDA
runtime through the installed PyTorch wheel. A real GPU CROMA workload was not
started after the dataset hard stop.

## 3. Dataset verification

The authoritative archive was:

`E:\SatQuery_ai_2.0\datasets_2.0\bigearthnet-v2-5000-20260911T162804Z-1-001.zip`

Its SHA-256 was independently recomputed as
`b3471e5650bd263367bcf4cc650405ab2981a86621579e21a1b1d08af630c595`, matching
the frozen manifest. The archive contained 5,000 metadata rows, 10,000 S1
members, and 5,000 reference-map members. Its outer, S1, and reference ZIP
CRC checks passed.

The local raster audit found:

| Input | Complete areas |
|---|---:|
| S1 VV/VH | 5,000 |
| S2 all 12 optical bands | 1,000 |
| Reference maps | 5,000 |
| Metadata | 5,000 |
| Complete multimodal areas | 1,000 |

The 4,000 missing optical areas are in `train`; validation and test each have
200 complete multimodal areas. This is a material input discrepancy, so the
scientific run stopped as required.

## 4. Split verification

The independent audit observed 4,600 train, 200 validation, and 200 test rows.
Declared counts matched metadata, split intersections were empty, and the
recomputed split fingerprint matched:

`2232ac5bc65d3ed20c6f39deb6037bb8e05410b247fd6c9e538eddf9feb86c`

The 1,000-area multimodal subset consisted of 600 train, 200 validation, and
200 test areas. The 4,000 additional manifest IDs are train-only optical
missing rows. This subset cannot be used as a substitute for the declared
5,000-area experiment.

## 5. Existing checkpoint verification

The declared frozen checkpoint exists at:

`D:\Satquery_ai datasets\comparison\pipeline3-5000\models\baseline\hybrid.pt`

It loaded successfully in the repaired Python 3.11 environment with:

- feature: `hybrid`
- input dimension: `830`
- state shape: `(19, 830)` plus `(19,)` bias
- seed: `17`
- selected epoch: `54`
- test evaluation flag: `False`
- file SHA-256: `0694dd82d4a3c03663ed7128adb4be6df729178e07b4ff8a1ec4295ed912130f`

The checkpoint was not overwritten, retrained, or evaluated during this phase.

## 6. Fresh 5,000-area inference

**NOT RUN.** The existing audit stopped before feature extraction because
4,000 train areas lack complete S2 optical inputs. Therefore there are no
fresh inference counts, skipped-area substitutions, prediction files, timing
measurements, or peak-memory claims.

## 7. Fresh training revalidation

**NOT RUN.** Training would violate the declared 4,600/200/200 exact-input
contract while 4,000 train areas are missing optical data. No revalidation
directory was created and the baseline directory was not modified.

## 8. Final test metrics

**NOT COMPUTED.** The sealed test set was not used for a fresh scientific
evaluation in this phase.

Historical reference values, retained only for comparison, are MAE `4.1034`
percentage points, RMSE `9.5873` percentage points, and dominant-class
accuracy `65.0%`. They are not Phase 6R.2 results.

## 9. Historical versus fresh comparison

No comparison is valid because no fresh prediction set was produced. Fresh
values, differences, relative differences, bootstrap intervals, and exact
accuracy numerator/denominator are therefore unavailable.

## 10. Leakage audit

The pre-execution dataset audit provided the following evidence:

| Check | Status | Evidence |
|---|---|---|
| Duplicate area IDs | PASS | `0` |
| Duplicate S1 identities | PASS | `0` |
| Duplicate S1 content groups | PASS | `0` |
| Duplicate S2 content groups | PASS | `0` |
| Cross-split duplicate S2 content | PASS | `0` |
| Cross-split duplicate reference content | PASS | `0` |
| Cross-split spatial overlap | PASS | `0` |
| Train/test image intersection | PASS | frozen split intersections empty |
| Label leakage | NOT DEMONSTRATED | no fresh feature run occurred |
| Feature leakage | NOT DEMONSTRATED | no fresh feature run occurred |
| Pipeline/cache leakage | NOT DEMONSTRATED | no fresh inference/training run occurred |

The audit reported 103 duplicate reference-content groups, but none crossed
splits. This is not evidence of cross-split leakage.

## 11. CROMA validation

**NOT DEMONSTRATED.** No full CROMA extraction was started after the exact
dataset stop. Token shapes, GAP shapes, repeatability, and numerical tolerance
are consequently not claimed.

## 12. Physical feature validation

**NOT DEMONSTRATED for the 5,000-area run.** The repository’s existing code
defines the 62D physical representation, but Phase 6R.2 did not extract it
from the incomplete selected population.

## 13. End-to-end representation connectivity

**NOT DEMONSTRATED.** No real area was traced through the complete declared
5,000-area Pipeline 3 path in this phase. No semantic grounding claim is made.

## 14. Evidence and provenance connectivity

**NOT DEMONSTRATED for fresh outputs.** No new representation artifacts,
receipts, evidence, interpretation results, or TaskResults were generated.

## 15. API connectivity

**NOT RUN.** The required real 5,000-area scientific input was unavailable,
so no fresh `/api/analyze` result is reported.

## 16. Performance and bottleneck audit

The measured blocker was dataset completeness, not model throughput:

- archive hash: approximately 0.91 seconds
- raster and manifest audit: approximately 148.98 seconds
- total audit: approximately 164.13 seconds
- CUDA: unavailable; no GPU memory or utilization measurement exists

No CROMA, raster-feature, model, evidence, or API bottleneck was measured.

## 17. Failure-mode audit

The exact-input completeness failure was explicit and fail-closed:

`status = blocked_before_training`

No fallback data, partial train set, stale checkpoint, or fabricated result was
used. The requested temporary malformed-input tests were not run after the
hard stop.

## 18. Reproducibility manifest

The audit evidence records:

- repository commit: `2f7b241e4f4687d2aae263faf4b1d5c08eae9e5e`
- dataset fingerprint: `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`
- split fingerprint: `2232ac5bc65d3ed20c6f39deb6037bb8e05410b247fd6c9e538eddf9feb86c`
- archive SHA-256: `b3471e5650bd263367bcf4cc650405ab2981a86621579e21a1b1d08af630c595`
- checkpoint file SHA-256: `0694dd82d4a3c03663ed7128adb4be6df729178e07b4ff8a1ec4295ed912130f`
- Python: `3.11.15`
- PyTorch: `2.14.0+cpu`
- CUDA available: `False`

No final scientific revalidation fingerprint was generated because the run did
not reach prediction or training.

## 19. Known limitations

1. The local S2 optical collection covers only 1,000 of the required 5,000
   areas; the missing 4,000 are train areas.
2. CUDA is unavailable in the repaired environment, so GPU CROMA behavior was
   not validated.
3. Fresh 5,000-area inference, retraining, metrics, API connectivity, and
   artifact connectivity remain unverified.
4. The declared archive and split manifests are present and internally
   consistent, but an archive alone does not provide the missing extracted S2
   rasters required by the current Pipeline 3 implementation.

## 20. Recommended next phase

Acquire or restore the exact missing 4,000 train-area S2 12-band rasters into
the manifest-declared local roots, then rerun the independent audit. Do not
train, evaluate, or change the checkpoint until the audit reports 5,000
complete multimodal areas and the expected split counts.

## Final stage table

| Stage | Status | Evidence |
|---|---|---|
| Dataset integrity | FAIL | 4,000 train areas missing all 12 S2 bands |
| Split integrity | PASS | 4,600/200/200 and expected split fingerprint |
| Environment | YELLOW | imports pass; CPU-only PyTorch, no CUDA |
| Checkpoint load | PASS | 830D, 19 outputs, seed 17, epoch 54 |
| Fresh 5k inference | NOT RUN | blocked by dataset completeness |
| Fresh metrics | NOT RUN | no fresh predictions |
| Fresh training | NOT RUN | stopped before training |
| Leakage audit | YELLOW | identity/split checks pass; fresh feature leakage not demonstrated |
| CROMA | NOT RUN | hard stop before extraction |
| Physical features | NOT RUN | hard stop before extraction |
| Hybrid representation | NOT RUN | hard stop before extraction |
| Evidence | NOT RUN | no fresh artifacts |
| Provenance | NOT RUN | no fresh artifacts |
| API | NOT RUN | no fresh 5k sample path |
| Performance | YELLOW | audit timing only; model bottleneck not measured |
| Failure handling | PASS | explicit `blocked_before_training` result |
| Reproducibility | YELLOW | input/checkpoint identities recorded; no fresh scientific fingerprint |

**Overall verdict: RED.**