# Phase 2A Annotation Design

## Scope

Phase 2A establishes a split-safe, provenance-preserving annotation foundation for the pinned BigEarthNet.txt source. It does not train or infer VQA, captioning, grounding, temporal, or agentic capabilities, and it does not modify Pipeline 3.

## Existing abstractions reused

- `src/annotation_foundation.py` remains the ingestion and validation owner.
- `src/image_language_dataset.py` consumes verified annotation records into deterministic downstream pools.
- `src/region_grounding.py` remains deterministic geometry-to-token infrastructure only.
- Pipeline 3 manifests remain the authority for image identity and split ownership.
- `canonical_bytes`, SHA-256 receipts, and the existing artifact layout provide stable serialization and provenance.

## Canonical identity and schema

The annotation schema is `annotation_schema_v1`. Each JSONL record contains these stable fields:

| Field | Meaning | Null allowed |
|---|---|---|
| `annotation_id` | SHA-256-derived identity from source revision, source ID, and complete raw record | No |
| `image_id` | Canonical Pipeline 3 area ID, or null when unmatched | Yes |
| `split` | Split inherited from the canonical image manifest | Yes |
| `annotation_type` / `task_type` | Controlled type: `binary_qa`, `multiple_choice_qa`, `caption`, `text_box`, `point_box` | Yes |
| `question` | Original source input for Q&A | Yes |
| `answer` | Original source output for Q&A | Yes |
| `choices` / `options` | Source choices when present | Yes |
| `caption` | Original source output for caption records | Yes |
| `bbox` / `box` | Parsed source box, not a learned or geographic box | Yes |
| `point` | Parsed source point | Yes |
| `source_dataset` | `BigEarthNet.txt` | No |
| `source_revision` | Pinned source revision in provenance | Yes |
| `source_file` | Artifact provenance file name | Yes |
| `source_record` / `source_record_id` | Original source row identifier | Yes |
| `provenance` | Matching method, source and artifact references | No |
| `metadata` | Preserved source metadata fields such as country, season, climate zone, latitude, longitude | No |
| `raw_source` | Complete untouched source row | No |

Existing field names are retained for compatibility; aliases make the contract explicit without forcing downstream redesign. Task-specific values are null rather than fabricated.

## Image-ID normalization

`normalize_image_id` is the single lexical normalization mechanism. It trims whitespace, normalizes path separators, removes directory prefixes, and strips only observed `.SAFE` and `.zip` container suffixes. It never removes date, tile, orbit, sensor, or numeric suffix tokens. `build_image_id_index` rejects two distinct source values that normalize to one ID. The complete normalized patch identity is preserved; ambiguous or malformed IDs are quarantined or fail manifest validation.

Matching still requires the exact normalized optical and SAR pair. A basename or shortened suffix is never used as a fallback because it can collide across areas.

## Split safety

The Pipeline 3 image manifest and split manifest are authoritative. Each canonical image must occur in exactly one split, and image/split manifests must agree. Every matched annotation inherits the image split; the source annotation `split` is retained only as `source_partition`. A source `bench` annotation produces `bench_holdout` and holds out every annotation on that image. Missing images remain unmatched/quarantined and are never assigned to a split. Duplicate image or sensor identities across splits fail closed.

## Duplicate and conflict policy

Records are never silently discarded.

- `duplicate_exact`: identical complete source content repeated.
- `duplicate_conflict`: one source annotation ID used with different complete content.
- `image_level_conflict`: same image and question with different answers.
- `spatial_conflict`: same image and spatial reference with different geometry.
- `unmatched_image`: no exact normalized optical/SAR pair.
- `split_conflict`: source partition disagrees with the inherited image split or image annotations disagree at image level.

Affected records are retained with deterministic validation issues and quarantined. Reports also retain the older compatibility counters (`duplicate_source_records`, `duplicate_source_ids_with_different_content`, and `source_partition_disagreements`).

## Annotation types

The source schema is inspected rather than assumed. The observed source types are `binary`, `mcq`, `captioning`, and `bounding box`. The bounding-box categories are preserved as `text_box` for `reference` and `point_box` for `point`; other source categories remain in `source_category` and `raw_source`.

## Spatial representation

Boxes use the publisher's observed string form `[first first, second second]`; points use `<point>(first, second)</point>`. Values are parsed as finite numeric values and checked for unit-square bounds, positive box extent, and point-within-box consistency. The source coordinate convention, origin, axis direction, pixel/geographic semantics, CRS, raster mapping, and inclusive/exclusive boundary remain `UNKNOWN`. No geographic grounding or learned grounding is claimed.

## Provenance and artifact format

The source revision is `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316`; the verified source SHA-256 is `d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554`. Provenance records source file name, source checksum, source revision, source schema, manifest/split checksums, matching implementation checksum, code revision, and annotation artifact checksum. Unavailable values remain null.

Canonical output is UTF-8 JSONL with sorted JSON keys and compact separators. Records are sorted by annotation ID and source-record checksum. The artifact includes `manifest.json`, `annotations.jsonl`, `validation_report.json`, `split_audit.json`, `leakage_audit.json`, `source_schema.json`, `geometry_audit.json`, and `provenance.json`. Absolute machine paths are excluded from canonical records. SHA-256 fingerprints cover canonical bytes; generation timestamps are provenance-only.

## Downstream compatibility

`image_language_dataset.py` can query records by task family, image, and split and assigns conservative pools. VQA candidates require `VISUAL_ONLY`; metadata-dependent and spatial records remain restricted or quarantined. Future captioning and grounding consumers can use the preserved source text and geometry without changing the annotation schema. No `RepresentationRef` is fabricated.
