# Phase 2D Foundation Audit

## Verdict

Phase 2D-Foundation is complete for the two requested prerequisites:

1. an authoritative 5,000-area representation availability catalog
2. an evidence-backed spatial-source mapping decision

The catalog correctly reports that no complete per-image representation store exists for the 5,000-area manifest. The spatial decision correctly keeps all BigEarthNet.txt source geometry unmapped.

## Files and components

Created:

- `src/representation_catalog.py`
- `src/region_contract.py`
- `tests/test_representation_catalog.py`
- `tests/test_region_contract.py`
- `scripts/build_representation_catalog.py`
- `docs/PHASE2D_REPRESENTATION_CATALOG.md`
- `docs/PHASE2D_SPATIAL_SOURCE_MAPPING.md`
- `docs/PHASE2D_FOUNDATION_AUDIT.md`

Modified:

- `.gitignore` to exclude the generated large catalog payload
- prior Phase 2 files remain in the existing uncommitted worktree

## Catalog results

Authoritative manifest:

- Images: `5,000`
- Train: `4,600`
- Validation: `200`
- Test: `200`
- Catalog rows by split: train `69,000`, validation `3,000`, test `3,000`

Catalog:

- Representation types enumerated: `15`
- Rows: `75,000`
- Verified represented images: `0`
- Images without verified representations: `5,000`
- Missing rows: `75,000`
- Receipt rows inspected: `35`
- External receipt rows: `35`
- External receipt status: `PROVENANCE_MISMATCH`
- Catalog SHA-256: `30913ab2da5ccc7b9ec51c849da68ea408c870844a257a402ee6597fbaef63bb`

The external `61_39` artifacts are valid evidence/probe artifacts in their own context, but `61_39` is not an exact 5,000-area manifest identity. They are not admitted into the authoritative catalog.

## Receipt validation

Receipt discovery reuses the existing file SHA-256 helper and the same immutable artifact principles as `ArtifactResolver`/`ReceiptCatalog`. No filesystem path is written into canonical catalog records. Missing checksum, missing array, checksum mismatch, and image/split mismatch are explicit statuses.

## Spatial conclusion

BigEarthNet.txt source geometry is numerically valid but semantically unresolved. The source proves unit-square ranges and paired point-box consistency only. It does not prove raster relationship, axis orientation, boundary convention, or CRS.

- Spatial records: `25,303`
- Proven source semantics: `0`
- Safely mapped records: `0`
- Deliberately unmapped: `25,303`
- Invalid/out-of-bounds/degenerate observed geometry: `0`
- Token links: `0`

No unsupported normalized-to-120×120 transform was introduced.

## Region abstraction

`region_contract_v1` supports `SOURCE_ANNOTATION_REGION` and `DETERMINISTIC_REGION`. `LEARNED_REGION` is reserved and rejected. Regions carry image ID, split, coordinate space, geometry, linked pixels/tokens, provenance, and validation status.

## Tests and determinism

Focused Phase 2D tests:

- `python -m pytest -q tests/test_representation_catalog.py tests/test_region_contract.py`
- `7 passed`

Full regression after the Phase 2D-Foundation changes:

- `python -m pytest -q`
- `339 passed, 5 skipped, 0 failed`
- `361 warnings`, consisting of existing `rasterio` deprecation warnings and the known Windows pytest cache ACL warning.

Catalog generation was repeated with representation types in different input order and produced the same canonical bytes/fingerprint. The full catalog generation is deterministic and emits UTF-8 sorted JSONL.

## Backward compatibility

No changes were made to Pipeline 3 preprocessing, CROMA inference, checkpoints, scientific calculations, evidence semantics, interpretation, API, frontend, or Phase 2A/2B behavior. No training or learned grounding was started.

## Remaining blockers

The full Region Representation phase cannot yet begin for real BigEarthNet spatial supervision. It requires:

1. a complete verified per-image representation receipt catalog for the desired images
2. authoritative source documentation or validated evidence establishing BigEarthNet.txt box/point raster semantics

The deterministic region contract and image-level catalog foundation are ready for that future input.
