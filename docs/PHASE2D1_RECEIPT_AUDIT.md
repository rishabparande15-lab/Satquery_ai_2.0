# Phase 2D.1 Receipt Audit

## Receipt schema and integrity

The catalog reuses existing artifact SHA-256 verification conventions from `src/representation_artifacts.py` and the typed receipt rules in `src/receipt_catalog.py`. Verified rows retain logical URI, artifact/shard checksum, size, shape, dtype, producer, model/checkpoint version where available, preprocessing version, dataset fingerprint, and validation status.

For shard-backed rows, `artifact_ref` records the shard checksum and `sample_index`; the checksum identifies the entire shard, not the individual sample. The `shard_ref` records the identity field (`area_ids`), shard name, and index.

## Real audit result

The authoritative manifest contains 5,000 images in the expected 4,600/200/200 split. The generated `representation_catalog_v1` contains 75,000 expected rows across 15 representation types.

- Verified representation images: `0`
- Missing representation rows: `75,000`
- Invalid rows: `0`
- Expired rows: `0`
- Split mismatch rows: `0`
- Provenance mismatch rows in catalog: `0`
- External receipt records inspected: `35`
- External receipt records rejected: `35`
- External receipt identity: `61_39`
- External receipt rejection: `PROVENANCE_MISMATCH`
- Catalog fingerprint: `30913ab2da5ccc7b9ec51c849da68ea408c870844a257a402ee6597fbaef63bb`

The external 61_39 artifacts were not mixed into the 5,000-area catalog.

## Existing representation evidence

The 5,000-area scientific report proves that physical, CROMA, and evaluated hybrid representations were used during the validated experiment, but the reusable per-image shard cache itself was not present in the available non-test artifact roots. A representation definition or historical metric is not treated as a per-image verified artifact.

The only exact 5,000-area feature-cache manifests discovered were pytest fixtures. They were intentionally excluded from catalog admission.

## Linking impact

Because the authoritative catalog has zero verified image representations, Phase 2C annotation links remain image-level but representation-missing for the full real annotation inventory. This is an accurate availability result. Spatial annotation links remain unmapped independently because BigEarthNet source coordinate semantics remain unknown.

## Determinism and performance

Catalog generation was repeated with the same authoritative manifest and produced the same `records.jsonl` fingerprint:

`30913ab2da5ccc7b9ec51c849da68ea408c870844a257a402ee6597fbaef63bb`

The missing-only catalog generation scans the 5,000-row manifest and completes without model inference or large-array materialization. Full runtime cache materialization is intentionally deferred until an existing validated cache or an explicitly approved generation run is available.

## Tests

- Phase 2D.1 catalog/region focused tests: `9 passed`
- Full regression result is recorded in the final completion report.

## Scientific safety

No Pipeline 3 source, preprocessing, CROMA adapter, checkpoint, training configuration, or scientific metric was changed. No retraining was performed. BigEarthNet spatial mappings remain disabled.
