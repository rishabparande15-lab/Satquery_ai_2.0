# Phase 6B.1 — BigEarthNet.txt annotation foundation

**Current state (Phase 6B.4):** The pinned official source is checksum-verified, real ingestion has produced 101,571 selected records on 4,771 images, and deterministic image-language pools are implemented. See [Phase 6B.2 source audit](ANNOTATION_SOURCE_AUDIT.md) and [Phase 6B.4 image-language dataset](IMAGE_LANGUAGE_DATASET.md). The 6B.1 baseline observations below describe the earlier foundation stage.

## Baseline and source audit

Implemented against clean `main` at `2f7b241e4f4687d2aae263faf4b1d5c08eae9e5e`. Pipeline 3's exact 5,000-area manifest has 4,600 train, 200 validation and 200 test areas. Its frozen scientific fingerprint is `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`. This phase makes no change to that runtime or its model, checkpoint, metrics, splits, evidence or API.

The historical Phase 3.6 reference audit inspected Ravi `Sat_Query` at `ac5541086c637ece2e9fd3c0a79cfa8bea23304d`. It describes a pinned Parquet source at `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316`, checksum `d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554`, with 9,553,962 rows. It reports dual-Sentinel matching on Ravi's distinct 1,000-area subset: 955 images, 20,453 annotations and 45 images without annotations. Its categories are 7,529 binary Q&A, 6,818 MCQ, 955 captions, 2,477 text-reference boxes and 2,674 point boxes. Those are **historical reference counts**, not a fresh Phase 6B.1 ingestion result. The Parquet source and Ravi's current checkout were unavailable in this workspace; GitHub network access failed. The live matcher/notebook/output schema could not be re-inspected. Our prior `src.phase1_foundation.link_annotations` conservatively checks `patch_id` and `s1_name`, but drops unmatched records and lacks a versioned output/audit; it was not reused as the new artifact producer.

Before changes, Python tests yielded 279 passed, 5 skipped, one unrelated HTTP upload socket-abort failure; frontend tests passed 7/7. The upload failure is a baseline observation, not caused by this phase.

## Implemented schema and matching

`annotation_schema_v1` JSONL records carry `annotation_id`, canonical `image_id`, `optical_identity`, `sar_identity`, experiment `split`, controlled `task_type`, original `source_task_type`/`source_category`, `source_dataset`, optional `source_record_id`, original `source_partition`, effective `use_partition`, match status, optional `question`, `answer`, `options`, `caption`, `box`, `point`, source geometry frame, `referenced_text`, entire untouched `raw_source`, source-record SHA-256, matching provenance, validation status and issues. Missing task-specific values stay null. Controlled mappings are `binary → binary_qa`, `mcq → multiple_choice_qa`, `captioning → caption`, `bounding box/reference → text_box`, `bounding box/point → point_box`, and `region → point_box`. An unrecognized source type is quarantined. Raw `input` and `output` strings are copied without trimming or rewriting. Derived unit-square box/point values are parsed from the publisher's observed strings; unknown raster mapping remains explicit. The source may have additional fields; these remain in `raw_source` even when no canonical interpretation is established.

Matching requires both exact source `patch_id` and `s1_name` against the canonical manifest's optical/area and SAR identifiers. The source `split` is retained as a source partition and never determines the experiment split. Every associated annotation inherits the frozen image split. Missing/ambiguous identifiers are quarantined, never guessed. The image and split manifests must agree, and duplicated canonical/Sentinel identities or cross-split image IDs abort ingestion. A source benchmark-holdout marker or a source/experiment partition conflict quarantines its record even if the image has an experiment split. Source test annotations are present for audit, but the artifact is not a training dataset; a later training exporter must explicitly exclude them.

The source table is scanned in batches. Only rows whose optical `patch_id` belongs to the selected canonical inventory enter the artifact; unrelated source images are counted as `outside_selected_image_inventory`, not mislabeled as failed matches. Rows for a selected optical identity with a missing or wrong SAR association are quarantined. This bounds artifact size to the selected experiment while retaining an explicit full-table scan count. Annotation IDs are SHA-256 over source revision, source record ID and the canonical hash of the whole source row. Thus they do not depend on Parquet/JSONL order. Duplicate IDs, identical raw records and conflicting uses of a source ID are quarantined and counted within the selected inventory. Records are sorted by ID/hash before JSONL serialization. A generation timestamp is isolated in `provenance.json`; `annotations.jsonl` and its hash stay byte-identical on repeated generation from the same input/configuration. `manifest.json`, `validation_report.json`, `split_audit.json`, `leakage_audit.json` and `provenance.json` accompany the JSONL. Provenance records source identity/revision/checksum, matching/schema version, image/split manifest checksums, code commit and UTC generation time. Only the source file's basename is stored; local paths are not embedded.

The leakage audit checks duplicate annotation IDs, raw rows and source IDs, duplicate/cross-split image IDs and sensor associations, identical question-answer strings across experiment splits, source benchmark holdout, and source-versus-experiment partition disagreement. Metadata answer exposure is now identified for country/season/climate-zone MCQ tasks; filename answer leakage remains unverified. Reference-box reuse is **unverified** because no grounding training/export exists. Source unit-square `box` (four values, positive extent) and `point` (two finite values inside the paired box) are checked. Axis origin, raster alignment, sensor grid and coordinate transform remain **UNKNOWN**. This phase explicitly treats coordinate convention and CROMA mapping as **UNKNOWN** until authoritative evidence is found, and it forbids unverified transformation assumptions.

## Reproduction

The verified official source is stored locally under ignored `data/raw/bigearthnet_txt/source/`. Run from the repository root:

```powershell
python -m scripts.ingest_bigearthnet_txt
```

The default output is `artifacts/annotations/bigearthnet_txt/`. The CLI defaults to the pinned official revision/checksum/URL and fails on a missing or mismatched file. Large source tables and generated annotation JSONL stay ignored/uncommitted; only small reports/receipts are reviewable. The generator does not download imagery or annotations.

## Implemented downstream boundary

Phase 6B.4 converts this annotation artifact into `image_language_record_v1` pools without changing the `TaskRequest → Capability → RepresentationSet → Evidence → TaskResult` contracts. Records reserve a nullable future `representation_ref` but do not invent one. VQA, captioning and grounding training/evaluation, learned image-text alignment, RAG, LLM answers, agents and temporal modeling remain **not implemented**.

## Final local validation

The Phase 6B.1 post-change suite passed 286 Python tests with 5 skips, including six annotation-focused tests; frontend passed 7/7. Python compilation and `git diff --check` passed. At that earlier stage, the source was absent and real counts were unavailable. Phase 6B.2 has superseded that limitation with the verified real source and results in [the current source audit](ANNOTATION_SOURCE_AUDIT.md). The frozen scientific fingerprint is `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`.
