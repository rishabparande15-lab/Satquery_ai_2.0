# Phase 2B Spatial Audit

## Verdict

Phase 2B is complete for the spatial/raster contract and task-safe leakage-policy foundation. The implementation is additive. It does not alter Pipeline 3 preprocessing, model checkpoints, CROMA inference, evidence claims, or the 65% scientific baseline.

## Existing conventions discovered

- Optical canonical runtime tensor: `[12,120,120]` in fixed band order.
- SAR canonical runtime tensor: `[2,120,120]` in `VV,VH` order.
- B02 is the common 120×120, 10 m reference grid.
- Optical bands are reprojected to B02 with bilinear resampling.
- SAR and reference maps require exact CRS, bounds, dimensions, and transform alignment.
- Reference maps remain integer categorical arrays and use nearest semantics.
- Native S2 source resolutions are 10 m, 20 m, and 60 m before common-grid loading.
- CROMA spatial output is 225×768, represented as a 15×15 row-major grid.
- Pixel/token evidence uses 8×8 blocks and half-open bounds.
- Existing evidence uses north-up affine map bounds and preserves provenance.
- Sample `61_39` uses EPSG:32633, 10 m resolution, bounds `[373200,5352000,374400,5353200]`, and transform `[10,0,373200,0,-10,5353200,0,0,1]`.

## New authoritative contract

[src/spatial_contract.py](../src/spatial_contract.py) defines `spatial_contract_v1`, explicit spaces (`PIXEL`, `ANALYSIS_GRID`, `CROMA_TOKEN`, `NORMALIZED_IMAGE`, `GEO`), `RasterMetadata`, `Point`, `BBox`, deterministic token conversion, normalized conversion, affine GEO conversion, validation statuses, and SHA-256 spatial fingerprints.

[src/leakage_policy.py](../src/leakage_policy.py) defines `leakage_policy_v1` for scene classification, image VQA, captioning, grounding, temporal change, and optical-SAR tasks.

## Transform rules

- Pixel and analysis coordinates use x=column and y=row.
- Rasters use upper-left origin and north-up orientation.
- Boxes are positive-extent half-open boundary intervals.
- Invalid values are reported as malformed, out-of-bounds, or degenerate; no clipping occurs.
- Token mapping is row-major: `index = row * 15 + col`.
- 120×120 analysis pixels map to 15×15 8×8 token blocks.
- GEO transforms require both CRS and affine metadata; absent metadata fails closed.
- Source BigEarthNet.txt unit-square geometry is preserved but remains unmapped to image/GEO coordinates.

## Real-data spatial validation

Available real project artifacts validate the following:

- `61_39` pixel-feature arrays: 120×120 maps and 225-token derived arrays.
- `61_39` spatial evidence: 225 tokens, row-major token indices, deterministic regions, and affine map bounds.
- Pipeline 3 loader conventions: optical 12 bands, SAR 2 bands, reference map 120×120, aligned CRS/bounds/transform requirements.
- BigEarthNet annotation geometry: 25,303 parsed boxes and 13,124 parsed points; source coordinate convention remains unknown by design.

The current real annotation artifact reports zero invalid geometry issues, but that does not establish geographic mapping for the source unit-square coordinates.

## Leakage findings

- Phase 2A image-level leakage: `0`.
- Duplicate image/question combinations: `0`.
- Test rows assigned to train: `0`.
- Identical QA content across splits: `2,001`; this is separately reported content overlap, not image contamination.
- Pair, temporal, region, caption, and feature-provenance policies are now explicit and testable for future task records.

## Test results

Focused Phase 2B tests:

- `python -m pytest -q tests/test_spatial_contract.py tests/test_leakage_policy.py`
- `10 passed`

The tests cover coordinate-space validation, bounds classifications, degenerate geometry, normalized round trips, token row/column conversion, 15×15 mapping, affine GEO round trips, missing CRS/transform, swapped orientation, image leakage, pair leakage, temporal grouping, content overlap, and feature provenance.

Full regression after the Phase 2B changes:

- `python -m pytest -q`
- `326 passed, 5 skipped, 0 failed`
- `361 warnings`, consisting of existing `rasterio` deprecation warnings and the known Windows pytest cache ACL warning.

## Determinism

The transform functions are pure and deterministic. Repeated token and affine round trips produce identical serialized dictionaries. `spatial_fingerprint` uses sorted JSON and SHA-256. Existing evidence serialization tests continue to cover deterministic `spatial_evidence_v1` output.

The real `61_39` validation produced the same result on repeated evaluation: 225 row-major tokens, token 0 `[0,0,8,8]`, token 224 `[112,112,120,120]`, and EPSG:32633 origin mapping `(373200,5353200)`.

## Backward compatibility

No changes were made to `preprocessing.py`, `dataset_loader.py`, `croma_adapter.py`, `evidence_schema.py`, Pipeline 3 model code, checkpoints, interpretation, API, frontend, or receipt validation. The contract is an adapter and validation layer around existing conventions.

## Known limitations

- Geographic transforms require authoritative per-artifact CRS and affine metadata.
- BigEarthNet.txt source coordinates are not converted because their raster coordinate convention is not authoritative.
- Temporal scene identity is defined as a future requirement; no temporal dataset exists in this phase.
- Caption near-duplicate detection beyond exact string overlap is deferred.
- The contract does not perform reprojection or resampling.

## Intentionally deferred

VQA, captioning, learned grounding, segmentation, temporal/CDVQA, agent planning, model training, feature generation, and Pipeline 3 scientific changes remain out of scope.
