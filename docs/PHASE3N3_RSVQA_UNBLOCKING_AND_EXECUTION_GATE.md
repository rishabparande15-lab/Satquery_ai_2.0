# Phase 3N.3 — RSVQA Unblocking and First Real-Evaluation Gate

## 1. Executive status

**RSVQA: BLOCKED.** No evidence added since Phase 3N.2 legitimately unblocks real RSVQA evaluation. The new RSVQA acceptance contract and readiness artifact make the missing evidence explicit; they do not represent, download, or execute the dataset.

**NO REAL RSVQA BENCHMARK SCORE WAS GENERATED.** No RSVQA data was downloaded, no inference or prediction was generated, no training occurred, and no scientific artifact changed.

Machine-readable status: [`artifacts/audits/phase3n3_rsvqa_readiness.json`](../artifacts/audits/phase3n3_rsvqa_readiness.json).

## 2. Current RSVQA blocker inventory

| Blocker ID | Category | Current state | Required evidence | Local / external state | Prevents real evaluation |
|---|---|---|---|---|---|
| `RSVQA_DATASET_LOCAL` | DATASET | `DATASET_NOT_PRESENT_LOCALLY` | Immutable image/question/answer release | No local images, questions, answers, or RSVQA-named cache/artifact; source route not established here | Yes |
| `RSVQA_IDENTITY_REVISION` | PROVENANCE | UNVERIFIED | Exact identity, version/revision, source receipt | Absent locally; externally obtainable only after authoritative source is identified | Yes |
| `RSVQA_LICENSE` | LICENSE | UNVERIFIED | Image, annotation, and permitted-use terms | Absent locally; external evidence required | Yes |
| `RSVQA_SPLITS` | SPLIT | MISSING | Official train/validation/test manifests | Absent locally | Yes |
| `RSVQA_LINKAGE` | DATASET | UNVERIFIED | Image→question and question→answer keys | Absent locally | Yes |
| `RSVQA_PROVENANCE_CHECKSUMS` | PROVENANCE | UNVERIFIED | Manifest, source revision, archive/per-file identity | Absent locally | Yes |
| `RSVQA_LEAKAGE` | LEAKAGE | UNVERIFIED | Duplicate image/question/pair and cross-split/near-duplicate audit | Cannot run without authoritative manifests | Yes |
| `RSVQA_MODALITY` | ADAPTER | UNVERIFIED | Actual image mode/bands/value scale and deterministic preprocessing | Cannot inspect without source records | Yes |
| `RSVQA_MODEL` | MODEL | PARTIAL | Held-out task-compatible model path | Qwen RGB smoke exists; RSVQA/RS-adaptation validity absent | Yes |
| `RSVQA_EVALUATOR` | EVALUATOR | UNAVAILABLE | Official versioned evaluator/schema/normalization | No local evaluator; Phase 2E hook raises `NotImplementedError` | Yes |

## 3. Dataset acceptance contract

`src/evaluation/rsvqa.py` adds `RSVQADatasetAcceptance`. Every required acceptance field has exactly one of `VERIFIED`, `UNVERIFIED`, `MISSING`, or `NOT_APPLICABLE`; unknown data is never promoted to verified. Required fields are dataset identity/revision; image/question/answer manifests; both linkage directions; all three official splits; license; provenance; artifact identity; duplicate/cross-split/near-duplicate checks; and modality definition.

The present local contract has `MISSING` image/question/answer and split manifests and `UNVERIFIED` identity, license, linkage, provenance, checksums, leakage, and modality. Its conversion to the generic `DatasetReadiness` object is therefore fail-closed.

## 4. Actual local availability

Repository search of `README.md`, `docs/`, `src/`, `tests/`, `artifacts/`, `data/`, and `experiments/` found RSVQA references only in contracts, reports, structural tests, and reserved adapter code. It found no RSVQA data files. Project-managed `data/raw` contains only `bigearthnet_txt`; `artifacts/` has no RSVQA directory. A name-filtered Hugging Face cache inspection found no RSVQA/RS-VQA entry. No download was attempted.

| Local item | Status |
|---|---|
| Images | MISSING |
| Questions | MISSING |
| Answers | MISSING |
| Official split files | MISSING |
| Dataset metadata / source receipt | MISSING |
| License/source documentation | MISSING |
| Checksums/provenance manifest | MISSING |

## 5. Model and adapter compatibility

| Compatibility question | Current state | Evidence / consequence |
|---|---|---|
| Image ingestion | UNVERIFIED for RSVQA | Qwen RGB adapter processed a controlled RGB array; RSVQA image format/bands/value semantics are not locally available. |
| Task/prompt | STRUCTURAL_ONLY | Binary/MCQ/open-ended contract fixtures validate, but no real RSVQA schema/evaluator is available. |
| Answer generation | OPERATIONAL_GENERIC_RGB_ONLY | Qwen generation is not a task-quality or benchmark-validity result. |
| Benchmark scientific validity | UNVERIFIED | No held-out benchmark execution. |
| RS adaptation | NOT_ESTABLISHED | Qwen is not claimed as an RS-adapted/fine-tuned RSVQA VLM. |

The minimum future adapter is either documented direct source compatibility or a source-specific RGB input adapter that preserves image identity, split identity, modality, deterministic preprocessing, and provenance, while rejecting undocumented channel conversion, resizing, or cropping. Current classification: **REQUIRES_SOURCE_VERIFIED_ADAPTER_OR_DOCUMENTED_DIRECT_COMPATIBILITY**. No adapter or fine-tuning is implemented here.

## 6. Evaluator compatibility

The only repository RSVQA adapter is `RSVQAAdapter` in `src/image_language_foundation.py`, whose assembly path is reserved and raises `NotImplementedError`. No official evaluator code, evaluator revision, input/output schema, binary/MCQ/open-ended normalization, deterministic result proof, or evaluator provenance record is locally available. Structural synthetic accuracy in Phase 3N.1 is not a substitute. RSVQA evaluator status remains **UNAVAILABLE**.

## 7. Split, leakage, provenance, and license status

No official membership manifest exists locally, so train/validation/test membership; duplicate images/questions/image-question pairs; exact cross-split overlap; and near-duplicate overlap cannot be calculated. No file/annotation manifest enables revision or checksum validation. No license or permitted-use evidence is present. These statuses are `MISSING` or `UNVERIFIED`, not passing checks.

## 8. Conditions for `READY_FOR_REAL_EVALUATION`

The real gate can return `READY_FOR_REAL_EVALUATION` only when all following are verified for a concrete RSVQA release:

1. Immutable identity/revision and an image/question/answer manifest with artifact hashes.
2. Explicit image, annotation, and permitted-use license evidence.
3. Official train, validation, and test split manifests with image→question→answer linkage.
4. Deterministic duplicate and cross-split leakage audit, plus near-duplicate audit where applicable.
5. Actual image modality/value/preprocessing contract and either verified direct RGB compatibility or a validated source adapter preserving all identity/provenance fields.
6. Official, versioned evaluator with documented schemas, task-specific normalization, and deterministic provenance-bearing outputs.
7. A held-out compatible model and prediction schema. `hybrid_830d` remains excluded because it is scientific-predictor-only.

Any missing item keeps the gate non-executable. A verified dataset whose model/evaluator/prediction path is still unavailable is `PARTIALLY_READY`; the present release is not data-ready and remains `BLOCKED`.

## 9. Exact next action

Obtain (or be provided with) an authoritative, license-bearing, immutable RSVQA release manifest covering images, questions, answers, official splits and record linkage. Then rerun the acceptance contract and split/leakage audit before inspecting or executing any model/evaluator. No current repository action can replace that external evidence.

## 10. Validation and preserved state

`python -m pytest -q tests/test_rsvqa_readiness.py tests/test_real_evaluation_readiness.py tests/test_evaluation_infrastructure.py tests/test_evaluation_dry_run.py tests/test_agent_controller.py tests/test_evidence_schema.py --basetemp=.pytest-phase3n3-core` — **68 passed in 7.04s**.

`python -m pytest -q tests/test_multispectral_projector.py tests/test_sar_projector.py tests/test_temporal_contracts.py tests/test_temporal_rgb_adapter.py tests/test_grounding_contracts.py --basetemp=.pytest-phase3n3-modality` — **27 passed in 9.14s**.

The Phase 3N.3 tests verify all required acceptance fields, complete synthetic readiness, missing dataset/revision/split/linkage/provenance/license/leakage behavior, model/evaluator/prediction incompatibility, deterministic decisions, and the JSON artifact schema. They are synthetic gate tests only.

VRSBench, CDVQA, SAR, optical-SAR, temporal, grounding, hidden ISRO/SAC, Phase 2F, Pipeline 3, CROMA, `hybrid_830d`, scientific splits/checkpoints/receipts and scientific metrics are unchanged.
