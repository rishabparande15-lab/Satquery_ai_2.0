# Phase 2D Foundation: Representation Catalog

## Scope

`representation_catalog_v1` is the authoritative availability inventory connecting the exact Pipeline 3 image manifest to receipt-backed representations. It reuses the existing `ArtifactResolver` SHA-256 verification boundary and does not infer artifacts from directory names.

## Canonical image key

The canonical key is the exact `area_id` from `experiments/pipeline3_5000/dataset_manifest.json`. This is the same full Sentinel-2 patch identity used by Phase 2A `image_id`. No suffix, basename, or filesystem path fallback is permitted.

## Record schema

Each catalog row contains:

- `image_id`, `split`, `representation_type`, `representation_id`
- `artifact_ref`, `logical_uri`, `checksum`, `size`, `shape`, `dtype`
- `modality`, `spatial_semantics`
- `producer`, `model`, `model_version`, `preprocessing_version`, `dataset_version`
- `provenance`
- `availability` and `validation_status`
- `validation_issues`

Statuses are `VERIFIED`, `MISSING`, `INVALID`, `EXPIRED`, `PROVENANCE_MISMATCH`, and `SPLIT_MISMATCH`.

## Receipt policy

A receipt-backed row is verified only when the receipt identifies the exact manifest image, its split agrees when supplied, its array file exists, and its SHA-256 matches. Shape/dtype metadata is retained and is not invented. Existing receipt/catalog logic remains the integrity authority; this module only creates the image-indexed audit view.

A receipt for external `61_39` was inspected but rejected from the 5,000-area catalog because `61_39` is not an exact manifest `area_id`. This produced `PROVENANCE_MISMATCH`, not a guessed image match.

## Complete 5,000-area result

The catalog enumerates 15 representation types for each of 5,000 manifest images, producing 75,000 deterministic rows:

- Manifest images: `5,000`
- Train: `4,600`
- Validation: `200`
- Test: `200`
- Verified represented manifest images: `0`
- Unrepresented manifest images: `5,000`
- Catalog rows: `75,000`
- `MISSING`: `75,000`
- Invalid/mismatched/expired manifest rows: `0`
- Receipt rows inspected: `35`
- External receipt rows: `35`, all `PROVENANCE_MISMATCH` for `61_39`

The catalog fingerprint is `30913ab2da5ccc7b9ec51c849da68ea408c870844a257a402ee6597fbaef63bb`. The generated artifact is local and ignored because it contains a large JSONL inventory; the builder is reproducible via `python -m scripts.build_representation_catalog`.

## Representation types inventoried

Raw optical, raw SAR, physical features, optical/SAR/joint CROMA tokens, optical/SAR/joint GAP, pooled CROMA, hybrid, pixel features, token features, regions, and evidence are represented as explicit catalog types. A missing row means no verified per-image receipt was available; it does not mean the scientific model lacks a representation conceptually.

## Region boundary

The minimal `region_contract_v1` supports source annotation and deterministic regions only. Learned regions are reserved and rejected. Geometry, linked pixels/tokens, split, image identity, and provenance are explicit.
