# Phase 2A Annotation Audit

## Verdict

Phase 2A annotation foundation is complete for the current repository scope. The implementation is additive, split-safe, deterministic, and provenance-linked. It does not implement a VQA model, captioning model, learned grounding, temporal reasoning, agent planning, or any Pipeline 3 change.

## Existing architecture reused

The foundation reuses the canonical Pipeline 3 dataset/split manifests, `src/annotation_foundation.py`, `src/image_language_dataset.py`, `src/region_grounding.py`, canonical JSON/SHA-256 helpers, and the existing artifact/receipt conventions. No second image identity system or model runtime was introduced.

## New or hardened components

- Conservative `normalize_image_id` and collision-checked `build_image_id_index`.
- Normalized identity use at manifest loading, source selection, and dual-sensor matching boundaries.
- Explicit contract aliases: `annotation_type`, `choices`, `bbox`, and `source_record`.
- Preserved source metadata in `metadata` and complete original data in `raw_source`.
- Explicit `duplicate_exact`, `duplicate_conflict`, `image_level_conflict`, and `spatial_conflict` audit counters.
- Adversarial tests for normalization collisions and answer/spatial conflicts.
- Required Phase 2A design and audit documentation.

## Exact schema

The canonical schema version is `annotation_schema_v1`. Required identity/provenance fields are `annotation_id`, `image_id`, `source_dataset`, `raw_source`, `source_record_sha256`, `provenance`, `validation_status`, and `validation_issues`. Nullable content fields are `question`, `answer`, `choices`, `caption`, `bbox`, `point`, and `metadata`. Compatibility names `task_type`/`annotation_type`, `options`/`choices`, `box`/`bbox`, and `source_record_id`/`source_record` carry the same value. `split` is inherited from the image manifest; `source_partition` is the untrusted source annotation partition; `use_partition` is the conservative effective partition.

## Real-data validation

The locally available source is the pinned BigEarthNet.txt Parquet file. It was scanned without downloading or committing the large source:

- Source rows scanned: `9,553,962`
- Selected canonical records: `101,571`
- Matched images: `4,771`
- Images without records: `229`
- Unmatched selected records: `0`
- Validated records: `101,542`
- Quarantined records: `29`
- Benchmark-held-out records: `29`
- Source SHA-256: `d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554`
- Source revision: `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316`
- Dataset fingerprint: `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`
- Split fingerprint: `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`
- Final `annotations.jsonl` SHA-256: `451cf9b86ef8ce3119fb824bad14ed65943735d78bb072550cb4edd1a3ebe47c`
- Final matching implementation SHA-256: `854a1baa67a208523cec2624778f0cdbbda318e8ccf2475a5b85587c31d96324`

Record counts by type:

- `binary_qa`: 37,518
- `multiple_choice_qa`: 33,979
- `caption`: 4,771
- `text_box`: 12,179
- `point_box`: 13,124

Dependency counts are `VISUAL_ONLY` 61,957, `VISUAL_PLUS_METADATA` 14,311, and `SPATIAL_VISUAL` 25,303. Parsed boxes: 25,303. Parsed points: 13,124. Coordinate convention remains unknown.

## Duplicate/conflict statistics

The verified local artifact reports:

- Duplicate annotation IDs: `0`
- Exact duplicate source records: `0`
- Conflicting source IDs: `0`
- Duplicate image/question combinations: `0`
- Image-level leakage: `0`
- Source/experiment split conflicts: `0`
- Test rows assigned to train: `0`
- Identical QA content across different image splits: `2,001`; this is reported as content overlap, not image leakage.

The fixture suite additionally proves that exact duplicates, conflicting source IDs, different answers for one image/question, and conflicting spatial metadata are retained and quarantined rather than silently removed.

## Split-safety validation

All matched records inherit the canonical image split. No annotation source partition is used as the experiment split. Cross-split image identity intersections are required to be zero, duplicate image/sensor identities fail manifest validation, benchmark images are isolated, and missing images remain unmatched. The real artifact has `0` cross-split image leakage and `0` records assigned to train from a test source partition.

## Determinism

Canonical serialization uses UTF-8, sorted keys, compact separators, stable record ordering, and SHA-256 fingerprints. Existing deterministic artifact checks pass for repeated fixture generation. Provenance timestamps are excluded from canonical annotation bytes and hashes.

## Test results

Focused Phase 2A validation after the changes:

- `python -m pytest -q tests/test_annotation_foundation.py`
- `13 passed`

The pre-existing annotation/image-language focused suite remains green:

- `python -m pytest -q tests/test_annotation_foundation.py tests/test_image_language_dataset.py`
- `21 passed`

Full regression after the changes:

- `python -m pytest -q`
- `316 passed, 5 skipped, 0 failed`
- `361 warnings`, consisting of existing `rasterio` deprecation warnings and the known Windows pytest cache ACL warning.

## Scientific regression

No retraining was performed. No Pipeline 3 predictor, checkpoint, scientific artifact, CROMA adapter, evidence pipeline, interpretation adapter, API, or frontend code was modified. The authoritative scientific baseline remains:

- Accuracy: `65.0%`
- MAE: `4.1034 pp`
- RMSE: `9.5873 pp`

## Known limitations

- The canonical source/experiment match is limited to the selected Pipeline 3 image inventory; the complete BigEarthNet.txt universe is counted but not exported.
- Coordinate convention and raster/geographic mapping are intentionally unknown.
- Metadata-dependent QA is not eligible for visual-only VQA.
- Identical question/answer text across images or splits is not itself image leakage; future model evaluation still needs task-specific leakage controls.
- No learned grounding or language model is present.

## Intentionally excluded from Phase 2A

VQA training, caption generation, learned grounding, temporal/CDVQA, agent planning, CROMA replacement, Pipeline 3 changes, arbitrary labels, fabricated confidence, silent conflict removal, and silent split assignment are all out of scope.

## Phase 2B blockers

There are no annotation-foundation blockers for beginning Phase 2B. Before any learned capability work, Phase 2B should define an explicit task-level leakage policy, establish authoritative geometry/raster conventions, and specify held-out evaluation partitions for each capability. The existing visual-only eligibility and benchmark holdout rules should remain enforced.
