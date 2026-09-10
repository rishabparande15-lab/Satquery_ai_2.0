# Phase 3.5 end-to-end verification

## 1. Objective

Verify one real local BigEarthNet paired sample through the connected path:

`Phase 3 scene selection -> local materialization -> spatial validation -> Phase 1 -> official CROMA -> Phase 2 adapter -> existing HybridFusion`.

This checkpoint adds verification only. It does not add retrieval, temporal
modeling, VQA, captioning, grounding, an agent/controller, training, or UI work.

## 2. Repository audit

The current checkpoint already contained Phase 3 contracts in
`src/phase3_orchestration.py`, Phase 1 in `src/phase1_foundation.py`, Phase 2 in
`src/phase2_integration.py`, and the authoritative advanced runner in
`src/hybrid_pipeline.py`. The Phase 3 bridge previously stopped at
`Phase2PipelineInput`; `hybrid_pipeline.py` performed real fusion but bypassed
Phase 3. The new verification runner closes only that execution gap.

The repository is at the post-Phase-2.5 hygiene state. Generated outputs remain
ignored and local; no dataset, checkpoint, or generated receipt is committed.

## 3. Existing pipeline identified

The existing advanced pipeline is `HybridFusion` in
`src/hybrid_fusion.py`, called with the existing 62-value physical feature
vector and the existing concatenated 2,304-value pooled CROMA vector. Its output
is the existing 192-value untrained representation. The Phase 1
`Linear(768,19)` coverage head was not used.

## 4. End-to-end architecture

`src/phase3_5_end_to_end.py` creates a stable `SatQueryRequest` and
`ExecutionPlan`, derives real Sentinel-1/Sentinel-2 scene candidates, selects
them through `LocalSceneAdapter` and `select_scene`, loads the paired local
BigEarthNet sample through the existing loader, validates optical/SAR spatial
metadata, constructs `ValidatedSatelliteDataset`, calls `bridge_to_phase2`, and
then invokes the unchanged `HybridFusion` module.

The only local wrapper is `_PreparedCroma`, which moves Phase 1-prepared tensors
to the already-loaded official CROMA device. It does not normalize, extract, or
fuse independently.

## 5. Data contracts

The verified real sample was `61_39`:

- Optical: `[1,12,120,120]`, order `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12`.
- SAR: `[1,2,120,120]`, order `VV,VH`.
- CROMA spatial outputs: `[1,225,768]` for optical, SAR, and joint.
- CROMA scene outputs: `[1,768]` for optical, SAR, and joint.
- Existing pooled fusion input: `[1,2304]`.
- Existing physical feature input: `[1,62]`.
- Existing fusion output: `[1,192]`.

The receipt preserves request ID, sample ID, product scene IDs, acquisition
timestamps, AOI, CRS, bounds, affine transform, native and processing
resolution, band orders, split, source hashes, normalization profile, CROMA
checkpoint hash, physical schema, adapter version, and advanced-pipeline schema.

## 6. Real sample and Phase 3 verification

The run used the existing local fixture at
`D:\Satquery_ai datasets\extracted\small-sample`, sample `61_39`, with no
download. Optical and SAR candidates were selected independently for the same
AOI and acquisition date. Scene IDs contain the same `61_39` sample identity.
The source hashes cover all 12 optical and both SAR source files.

The local loader materialized the canonical paired arrays on the B02 reference
grid. The paired CRS, bounds, transform, resolution, shape, finite values, and
canonical band orders were validated before Phase 1 preparation.

## 7. Phase 1 and CROMA verification

`prepare_batch` applied the named `croma_readme_patch_8bit_v1` normalization.
The official configured CROMA source/checkpoint was used on CPU. All six
representations were validated by the existing Phase 1/2 shape contract. The
15x15 spatial token convention remains row-major and no token reordering occurs.

## 8. Phase 2 verification

`bridge_to_phase2` supplied the Phase 1 prepared batch to the existing CROMA
model, computed physical features with `LocalRasterFeatureProvider`, and called
`adapt_phase1_output`. The adapter preserved sample identity, split, source
hashes, normalization, feature names/schema, channel order, and CROMA
checkpoint provenance. It produced the existing pooled vector in the existing
order: `optical_GAP, SAR_GAP, joint_GAP`.

## 9. Advanced pipeline verification

The runner instantiated the unchanged `HybridFusion(62,2304)` and produced a
finite `[1,192]` representation. No reference coverage head, new classifier,
retraining, or scientific model change was introduced.

## 10. Spatial correspondence and provenance

Optical and SAR arrays share the same validated grid metadata after the
established B02-grid materialization. The CROMA spatial outputs share the
validated 225-token structure. The final `ProcessingResult` and receipt trace
back through request, scene, acquisition, Phase 3, Phase 1, CROMA, Phase 2,
and `HybridFusion` to the exact source hashes.

## 11. Reproducibility

Two complete CPU executions with the same request ID and sample produced exact
matches for physical features, all CROMA tensors, fused features, source hashes,
and feature hashes. This is an observed exact match for the tested CPU setup;
cross-device bitwise determinism is not claimed.

## 12. Failure-path verification

The Phase 3 and Phase 3.5 suites reject wrong optical count/order, invalid
pixels, AOI coverage failures, optical/SAR CRS or grid mismatches, invalid CROMA
shapes, and invalid physical feature contracts with explicit exceptions/codes.
Phase 1/2 tests reject identity mismatch, duplicate identities, mixed splits,
invalid provenance, and feature-shape errors. B10 is excluded by the canonical
optical band contract. No silent correction is used.

## 13. Split isolation

The runner receives an explicit split and passes it into the Phase 1 identity;
it does not infer a split from file position. The three-sample fixture has no
official train/validation/test manifest, so the real verification uses an
explicit `test` assignment and reports that limitation. Existing Phase 1/2
split-leakage tests remain active.

## 14. Scientific parity

The existing physical feature formulas, CROMA configuration, pooled feature
ordering, and `HybridFusion` architecture are unchanged. The new path packages
the same raw sample and calls the same downstream components. No accuracy or
scientific capability claim is made.

## 15. Tests and receipt

Dedicated tests are in `tests/test_phase3_5_end_to_end.py`. The runner writes a
machine-readable receipt to the ignored `experiments/outputs/phase3_5/` path when
invoked with its default CLI. The receipt contains output dimensions, feature
hashes, source hashes, provenance stages, warnings, failures, and verdict.

Final validation for this checkpoint:

- Focused Phase 3.5 suite: `4 passed`.
- Full Python suite: `97 passed`.
- Frontend suite: `6 passed`.
- Python compilation: passed.
- `git diff --check`: passed.

## 16. Limitations

The local fixture is not a remote provider and has no official split manifest.
The real path uses the established local BigEarthNet loader rather than a new
remote acquisition adapter. The advanced output is an untrained representation,
not a task prediction. Temporal/change modeling, GEE retrieval, and remote
provider behavior remain unverified and intentionally out of scope.

## 17. Files changed

- `src/phase3_5_end_to_end.py`
- `tests/test_phase3_5_end_to_end.py`
- `docs/phase3_5_end_to_end_verification.md`
- `README.md`

Generated receipts and arrays are ignored and remain local only.

## 18. Final verdict and next recommendation

**FINAL VERDICT: GREEN for the verified local core path.**

The real paired sample traversed Phase 3 contracts, Phase 1, official CROMA,
Phase 2, and the existing advanced fusion pipeline with identity, spatial,
provenance, and reproducibility checks. This GREEN verdict is bounded by the
local fixture and does not certify remote providers, temporal analysis, or task
accuracy.

The next action should be checkpoint review and approval of this verification.
Only after review should a subsequent phase define a concrete remote-provider
adapter; no Phase 4 implementation is justified solely by the existence of
Phase 3 contracts.