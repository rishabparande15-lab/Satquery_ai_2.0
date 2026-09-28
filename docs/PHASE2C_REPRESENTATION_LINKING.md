# Phase 2C Annotation to Representation Linking

## Scope

Phase 2C creates `representation_link_v1`, a deterministic bridge from verified Phase 2A annotations to canonical image identity, split, spatial geometry, existing representation references, and provenance. It provides geometric correspondence only. It is not learned grounding, VQA, captioning, temporal reasoning, or a model-training layer.

## Existing abstractions reused

- `annotation_schema_v1` records from `src/annotation_foundation.py`
- `RepresentationRef`, `ArtifactRef`, `SpatialReference`, and `RepresentationSet` from `src/architecture_contracts.py`
- receipt-backed representation identity from `src/receipt_catalog.py`
- `spatial_contract_v1` from `src/spatial_contract.py`
- existing `spatial_evidence_v1` and `north_up_row_major_120_to_15_v1`
- `image_language_dataset.py` task and split semantics

The new owner is [src/representation_linking.py](../src/representation_linking.py). It does not replace any existing representation or evidence schema.

## Canonical link schema

`AnnotationRepresentationLink` serializes:

- `link_id`
- `annotation_id`
- `image_id`
- `split`
- `annotation_type`
- `task_type`
- `spatial_reference`
- `representation_refs`
- `representation_availability`
- `relationship_types`
- `evidence_reference`
- `source_provenance`
- `linkage_version`
- `validation_status`
- `validation_issues`
- optional `scene_id`, `observation_id`, and `temporal_group_id`

Temporal fields remain null because authoritative temporal identities do not exist in the current single-image dataset.

## Image and split linkage

The linker requires an annotation image ID to exist in the canonical image manifest. Missing or unknown IDs become `IMAGE_UNMATCHED`. Annotation, manifest, and representation splits must agree. Missing split metadata becomes `SPLIT_UNKNOWN`; it is never treated as valid by assumption. Representation scene identity is checked against the annotation image identity, and mismatches become `PROVENANCE_MISMATCH`.

## Spatial linkage

Text-only records receive an image-level relationship with `spatial_reference = None`; no coordinates are invented. Spatial source records preserve their original `box`, `point`, and `geometry_frame`. Since Phase 2A source geometry is `source_unit_square_unmapped`, those records receive `SPATIAL_LINK_UNAVAILABLE` unless a caller supplies an independently validated canonical `BBox` or `Point`.

For canonical geometry:

- `BBox` supports `PIXEL`, `ANALYSIS_GRID`, and `NORMALIZED_IMAGE`.
- `Point` supports the explicit coordinate space supplied by the caller.
- Validation uses the Phase 2B half-open 120×120 convention.
- Bboxes map to every intersecting 8×8 token block.
- Each token relation is `FULLY_CONTAINED`, `INTERSECTS`, or `CONTAINS`.
- Both fraction of token covered and fraction of bbox covered are recorded.
- Points map deterministically to one analysis pixel/token; boundary points use the established half-open cell convention with the image edge treated as the final boundary.
- Invalid geometry is represented as an invalid link and is never clipped.

Relationship names are descriptive: `IMAGE_TO_REPRESENTATION`, `BBOX_TO_ANALYSIS_REGION`, `BBOX_TO_CROMA_TOKENS`, `POINT_TO_ANALYSIS_PIXEL`, and `POINT_TO_CROMA_TOKEN`. None means learned grounding.

## Representation availability and provenance

A link can be valid while individual representation families are unavailable. Availability is explicit and may be `available`, `REPRESENTATION_MISSING`, `SPLIT_MISMATCH`, or `PROVENANCE_MISMATCH`. Linked representation records retain reference ID, representation type, scene/image ID, split, producer, version, preprocessing version, logical artifact URI, checksum, and provenance reference where available. Absolute local paths are excluded from canonical representation records.

The current representation inventory includes raw optical/SAR, physical features, optical/SAR/joint CROMA tokens and GAP vectors, hybrid features, pixel/token features, regions, evidence, and metadata/provenance. The linker references these existing identities; it does not fabricate unavailable grounding masks, temporal vectors, or semantic labels.

## Deterministic artifact

`write_link_artifact` writes:

- `manifest.json`
- `links.jsonl`
- `validation_report.json`
- `provenance.json`

Records are sorted by deterministic `link_id`, serialized as UTF-8 canonical JSONL, and fingerprinted with SHA-256. Timestamps and machine paths are excluded from canonical link bytes.
