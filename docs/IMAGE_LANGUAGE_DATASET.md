# Phase 6B.4 — image-language dataset

## Status and boundary

**IMPLEMENTED and VERIFIED:** deterministic image-language records, six disjoint pools, image-level split inheritance, benchmark isolation, metadata/spatial restrictions, provenance, balance summaries, duplication/leakage audits, and reproducibility checks.

**QUARANTINED:** every spatial record because coordinate semantics and raster mapping remain `UNKNOWN`; every annotation on the one benchmark image; any future invalid or unknown-dependency record.

**NOT IMPLEMENTED:** VQA/captioning/grounding training, CROMA fine-tuning, box/point-to-pixel or token mapping, free-form answer generation, an agentic controller, resampling, model selection, or benchmark scoring.

**PLANNED:** Phase 6B.5 may train the first visual-only VQA baseline using only the approved visual VQA pool and the frozen image splits.

Pipeline 3 is unchanged. Its frozen fingerprint remains `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`.

## Source and schema

The input is the checksum-verified `annotation_schema_v1` artifact generated from the author-maintained BigEarthNet.txt source at revision `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316`, source SHA-256 `d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554`. The regenerated annotation JSONL SHA-256 is `5562e7976aee140850fede7efe5d1c375934244fe5c0c4a754b76eeb1af5447b`; it includes the Phase 6B.3 dependency taxonomy.

`image_language_record_v1` contains deterministic record/source IDs; canonical image identity, frozen split, effective partition and pool; controlled task type; original question, answer/raw answer, caption and source identity; ordered derived MCQ options and deterministically resolved answer text; nullable quarantined spatial values; dependency class, eligibility and restriction reason; checksum-linked provenance; and a nullable future `representation_ref`.

Optional fields remain null where inapplicable. No image bytes, tensors, absolute paths, local drive letters, or invented CROMA references are embedded. Record IDs are SHA-256 identities over schema version and annotation ID. Source question/answer/caption strings are not rewritten. All 33,979 MCQs parsed into four ordered source options; all raw answer keys were deterministically resolvable, with distribution `a=8,553`, `b=8,566`, `c=8,297`, `d=8,563`.

## Pool policy and exact counts

Pool assignment is exclusive. Benchmark assignment takes precedence. Non-benchmark validated binary/MCQ records enter the visual pool only when dependency is `VISUAL_ONLY` and eligibility is `ELIGIBLE_VISUAL_VQA`. Metadata-dependent Q&A remains restricted. Captions remain a whole-image task family. Spatial annotations remain quarantined.

| Pool | Total | Train | Validation | Test | Images | Tasks |
|---|---:|---:|---:|---:|---:|---|
| `visual_vqa_candidate` | 57,162 | 52,504 | 2,369 | 2,289 | 4,770 | 37,503 binary; 19,659 MCQ |
| `metadata_aware` | 14,310 | 13,155 | 591 | 564 | 4,770 | 14,310 MCQ |
| `captioning_candidate` | 4,770 | 4,385 | 197 | 188 | 4,770 | 4,770 captions |
| `spatial_quarantine` | 25,300 | 23,164 | 1,018 | 1,118 | 4,011 | 12,178 text boxes; 13,122 point boxes |
| `benchmark_holdout` | 29 | 0 | 0 | 29 | 1 | 15 binary; 10 MCQ; 1 caption; 1 text box; 2 point boxes |
| `unknown_quarantine` | 0 | 0 | 0 | 0 | 0 | none |

The complete input contains 101,571 records: 37,518 binary Q&A, 33,979 MCQ, 4,771 captions, 12,179 text-reference boxes, and 13,124 point-guided boxes. Candidate Q&A totals 71,472; 57,162 records are eligible visual-only VQA candidates and 14,310 are metadata-aware restricted records. Quarantined/held-out records total 25,329. No missing annotation is converted into a negative.

The inherited taxonomy remains unchanged: `VISUAL_ONLY=61,957`, `SPATIAL_VISUAL=25,303`, and `VISUAL_PLUS_METADATA=14,311`. No selected record is currently `METADATA_ONLY` or `UNKNOWN_DEPENDENCY`; the schema and policy still fail closed for those classes.

## Split, leakage, overlap, and balance audit

Every record inherits the frozen canonical image split. No annotation-level split is created. Image intersections are zero for train/validation, train/test, and validation/test. All 29 annotations on the benchmark image are in `benchmark_holdout`; benchmark contamination is zero. Spatial records in visual VQA are zero, and metadata-dependent records in visual VQA are zero.

Duplicate record IDs, annotation IDs, source records, image/question pairs, and image/question/answer triples are all zero. There are 6,302 identical question/answer pairs used by more than one image, 5,186 question strings used by more than one image, and 2,001 distinct question/answer pairs occurring across splits. These are content overlap, not image leakage; they are reported without removal.

Binary answers are `no=19,087` and `yes=18,431`. The full task, split, dependency, eligibility, question-frequency, MCQ option-count, and answer-key distributions are in `eligibility_audit.json`. This is descriptive only. No balancing, sampling, shuffling, or weighting is performed, and no claim of dataset balance is made.

## Artifacts and reproducibility

Generated files live under `artifacts/annotations/image_language/`: six pool JSONLs plus `manifest.json`, `eligibility_audit.json`, `leakage_audit.json`, and `provenance.json`. The large JSONL files are ignored by Git; compact receipts are reviewable. Canonical JSON uses sorted keys, compact separators, stable ordering, and no generation timestamp. The timestamp appears only in provenance. Two independent generations produced identical records, IDs, ordering, pool membership, splits, and hashes.

The image-language fingerprint is `186bf42721a8eba484167989c2647ccac4a2a2495e09ec2a0f7f9fba47459fef`.

| Artifact | SHA-256 |
|---|---|
| visual VQA | `ea37673d6124baa50c12ba94d5f1d1867b966c354aa4c4d2b81fba3ac843a31d` |
| metadata-aware | `c7a6020e8410c4d6aa7833fcab7ff0eb684404e0c2aaf93d3fb2887a04429d2f` |
| captions | `289ddcdd12b49520b579007cd232de364077dbeba13fb8ec9a4595c54c4574a9` |
| spatial quarantine | `dd95334ac62273bee5cc06cd3e7866f6d74c4c8960f1546c2d26ce31f9f0c497` |
| benchmark holdout | `909bbfda8441f68f887443634282360b492f52fc0ec10dd3d490027943ff46d2` |
| unknown quarantine (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Rebuild with `python -m scripts.ingest_bigearthnet_txt` followed by `python -m scripts.build_image_language_dataset`.

## Limitations and Phase 6B.5 recommendation

Coordinate convention, raster mapping, S1/S2 correspondence, and CROMA 15×15 mapping remain unresolved. Image IDs encode acquisition/tile information and must not be predictive text features. Captions contain geographic/seasonal language and are not visual-only VQA targets. Metadata-aware records require a separately declared experiment. Repeated question templates across splits require transparent reporting.

For Phase 6B.5, train a first closed-answer baseline only on visual-pool train records; use visual-pool validation for model selection; keep test untouched until the protocol is frozen; keep all other pools out of training and tuning; resolve images only through the canonical manifest; preserve raw answers and option order; report binary/MCQ results separately and by question-frequency strata; and do not use the benchmark holdout until preprocessing, answer handling, and model selection are final.

## Verification

Final verification passed 301 Python tests with 5 skips, 7/7 frontend tests, Python compilation, `git diff --check`, and 9 focused Pipeline 3 artifact/acquisition regressions. Pipeline 3 remains at 65.0% dominant-class accuracy, 4.1034 pp MAE, 9.5873 pp RMSE, and scientific fingerprint `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`.
