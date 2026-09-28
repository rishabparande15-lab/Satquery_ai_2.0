# Phase 2D.1 Representation Materialization

## Scope

Phase 2D.1 adds verified catalog admission for existing per-image receipts and for the established Pipeline 3 shard format. It does not recompute CROMA, change preprocessing, retrain, or create placeholder arrays.

## Existing artifacts inspected

- `experiments/pipeline3_5000/dataset_manifest.json`
- `experiments/pipeline3_5000/split_manifest.json`
- `scripts/run_pipeline3_5000_baseline.py`
- `src/representation_artifacts.py`
- `src/receipt_catalog.py`
- `src/live_representation_persistence.py`
- `artifacts/pixel_feature_probe/61_39/receipt.json`
- existing Pipeline 3 final metrics and scientific fingerprint

The exact 5,000-area training script defines a reusable compressed shard format containing `area_ids`, `splits`, `physical`, `optical_croma`, `sar_croma`, `joint_croma`, and targets. No completed authoritative 5,000-area shard cache was present outside test fixtures at audit time.

## Admission paths

`src/representation_catalog.py` now supports:

1. receipt-backed artifacts, verified using existing SHA-256 conventions
2. shard-backed artifacts, verified by shard checksum, exact `area_ids` index, split, shape, dtype, and dataset fingerprint

Shard-backed records distinguish the whole shard checksum from the per-image `sample_index`; the shard checksum is never mislabeled as a sample checksum. Canonical logical URIs contain no machine-local paths.

CLI:

```powershell
python -m scripts.build_representation_catalog --shard-manifest <cache>/manifest.json
```

## Current real coverage

The authoritative catalog currently enumerates 15 representation types for 5,000 images:

- expected rows: `75,000`
- train rows: `69,000`
- validation rows: `3,000`
- test rows: `3,000`
- verified images: `0`
- missing rows: `75,000`

The only discovered exact 5,000 manifests were pytest fixtures and were not admitted. The existing 61_39 receipt yielded 35 inspectable representation rows, but every row was rejected as `PROVENANCE_MISMATCH` because 61_39 is outside the authoritative 5,000-area manifest.

## Core representation set

For the current evaluated predictor, the core set is:

- 62-D physical features
- 768-D joint CROMA GAP
- derived 830-D evaluated hybrid input

No image currently has a verified complete core set in the authoritative catalog. The 192-D `HybridFusion` representation remains distinct and is not substituted for the evaluated 830-D representation.

## Safety boundary

No raw dataset copies, placeholder arrays, fabricated checksums, inferred splits, or BigEarthNet spatial mappings were created. BigEarthNet spatial annotations remain deliberately unmapped because their raster semantics are unresolved.
