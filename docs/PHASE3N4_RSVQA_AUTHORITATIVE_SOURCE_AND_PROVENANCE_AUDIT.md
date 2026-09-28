# Phase 3N.4 — RSVQA Authoritative Source and Provenance Audit

## 1. Executive summary

**RSVQA: BLOCKED.** The audit establishes a stronger source fact for RSVQA-LR: the author-hosted RSVQA website links its LR dataset to Zenodo version 1.0, DOI `10.5281/zenodo.6344334`; the record lists question, answer, image, and train/validation/test files with MD5 values. This verifies an immutable metadata/release candidate, not permission to acquire or evaluate the data.

The Zenodo record does not expose a dataset license value in its Rights section, and the author code repository’s GPL-3.0 applies to code, not demonstrably to dataset images, derived annotations, or underlying imagery. The license chain is therefore unverified. **No acquisition was authorized:** 0 files, 0 bytes. No inference, prediction, score, training, or scientific-artifact modification occurred.

## 2. Sources investigated and authority classification

| Variant / source | Classification | Evidence | Current use decision |
|---|---|---|---|
| RSVQA project website | PRIMARY_AUTHORITATIVE index | Author project page links code and all three dataset variants. | Source index only. |
| RSVQA-LR Zenodo record | PRIMARY_AUTHORITATIVE release candidate | Website links record; record identifies creators, version 1.0, DOI, files and MD5 values. | Metadata verified; data acquisition blocked by license chain. |
| `syvlo/RSVQA` code | PRIMARY_AUTHOR code | Website-linked author repo; GPL-3.0 code repository. | Code provenance only; not a dataset/evaluator license. |
| RSVQA-HR Zenodo route | PRIMARY_AUTHORITATIVE release candidate | Website links `zenodo.org/record/6344367`; page fetch was rate-limited in this audit. | Revision/files/license remain unverified. |
| RSVQAxBEN Zenodo route + `syvlo/RSVQAxBEN` | DERIVED_SUBSET / author-linked construction code | Website links the route; author code says it constructs a database leveraging BigEarthNet. | Not interchangeable with LR/HR; separate BigEarthNet/source-license audit required. |
| Third-party wrappers/mirrors | SECONDARY_MIRROR or UNVERIFIED | Not used for admission. | Never substituted for the author-linked release. |

Primary source links: [project site](https://rsvqa.sylvainlobry.com/), [LR Zenodo v1.0](https://zenodo.org/records/6344334), [author code](https://github.com/syvlo/RSVQA), [RSVQAxBEN construction code](https://github.com/syvlo/RSVQAxBEN).

## 3. Variant audits

### RSVQA-LR

The LR Zenodo record is published 2022-03-10 as version 1.0 and names the project authors. It declares Sentinel-2 as a keyword and lists `all_answers.json`, `all_questions.json`, `Images_LR.zip`, and separate train/val/test images/questions/answers JSON files with MD5s. Its file listing therefore supports source identity, a release identifier, and prospective split/file receipt checks.

It does **not** establish locally available contents, image↔question↔answer linkage, file-level SHA-256, split-membership consistency, leakage, image format/dynamic range, or a rights chain for annotations/images/underlying imagery. The Zenodo Rights area is present but its license value is blank in inspected metadata. Status: `AUTHORITATIVE_SOURCE_VERIFIED_FOR_METADATA`, acquisition and execution **BLOCKED**.

### RSVQA-HR

The project site calls HR a very-high-resolution dataset and describes USGS 15.24 cm orthorectified imagery with OSM-derived question/answer information. It links a Zenodo route, but that route was rate-limited during this audit. No HR record metadata, files, checksums, split, license or source imagery redistribution evidence was acquired. Status: `UNVERIFIED`.

### RSVQAxBEN

The project site links a separate dataset route; author code describes database construction over BigEarthNet and contains only construction/training material. It is a derived BigEarthNet VQA dataset, not interchangeable with original LR/HR. Its own release revision, files, license/provenance chain, splits, evaluator and leakage evidence were not established. Status: `UNVERIFIED` / `DERIVED_SUBSET`.

## 4. License and revision analysis

| License subject | State | Evidence / limitation |
|---|---|---|
| LR annotations | UNVERIFIED | Zenodo record gives no inspected license value. |
| LR image archive | UNVERIFIED | File list/MD5 does not establish rights. |
| Underlying Sentinel-2 / OSM-derived content | UNVERIFIED | Project describes provenance but this audit did not establish a release-specific redistribution/derivative-use chain. |
| HR USGS imagery / OSM-derived annotations | UNVERIFIED | Source route/terms not fully retrieved or bound to a release. |
| Author code | GPL-3.0 observed | Scope is code only; it cannot license benchmark artifacts. |
| Evaluator | UNVERIFIED | No authoritative standalone evaluator/version was established. |

RSVQA-LR’s DOI/version/file MD5 listing is the strongest observed immutability evidence. It does not replace SHA-256 verification of locally acquired files. A mutable GitHub `main` branch is not used as a dataset revision.

## 5. Bounded acquisition, structure, split, and leakage results

The Phase 3N.4 bounded acquisition condition was not met: all artifact/image/underlying-image licensing evidence must be verified before any metadata, manifest, split file, or representative image is downloaded. The resulting acquisition manifest truthfully contains no artifacts.

- [acquisition manifest](../artifacts/audits/phase3n4_rsvqa_acquisition_manifest.json): `ACQUISITION_NOT_AUTHORIZED`, 0 files, 0 bytes.
- [split audit](../artifacts/audits/phase3n4_rsvqa_split_audit.json): LR official split files are **listed by source**, but membership parsing/validation was not run.
- [leakage audit](../artifacts/audits/phase3n4_rsvqa_leakage_audit.json): all content-based and ID-based checks are `NOT_CHECKED`/`INCONCLUSIVE`; no claim of no leakage is made.

No source defect was repaired because no source file was acquired. Image/question/answer linkage remains unverified.

## 6. Evaluator analysis

The author code repository documents database generation and model training, but this audit did not establish an authoritative, standalone evaluator revision; official prediction schema; task normalization; binary, multiple-choice, open-ended, or counting metric behavior; or deterministic evaluator receipt. The repository’s `RSVQAAdapter` remains a reserved hook and raises `NotImplementedError`. Evaluator status: **UNVERIFIED**.

## 7. Model and modality compatibility

Qwen2.5-VL-3B-Instruct RGB execution is technically validated only for a controlled RGB array. It is not evidence that it can ingest actual RSVQA-LR Sentinel-2 inputs or HR orthophotos under an authoritative preprocessing contract, generate task-valid answers, or support a scientifically valid held-out evaluation. The required state is `REQUIRES_SOURCE_VERIFIED_ADAPTER_OR_DOCUMENTED_DIRECT_COMPATIBILITY`; no adapter/fine-tuning/inference was implemented or run.

## 8. Readiness result and exact blockers

The final machine-readable record is [phase3n4_rsvqa_readiness.json](../artifacts/audits/phase3n4_rsvqa_readiness.json). `execution_permitted` is **false** by phase rule and by unresolved gates.

Remaining blockers:

1. Written, release-specific license and permitted-use evidence for annotations, packaged images and underlying imagery.
2. After licensing clears, a bounded retrieval of authoritative LR manifest/split/annotation metadata and hashes.
3. Structural linkage and official split validation; duplicate, cross-split and applicable near-duplicate leakage audit.
4. Source image modality/dimensions/dynamic-range and deterministic preprocessing admission.
5. Authoritative evaluator revision/schema/normalization verification.
6. Separately, held-out model/prediction-format validity. Technical Qwen RGB execution alone is insufficient.

## 9. What is verified and what remains unverified

**Verified:** project-site-to-LR-Zenodo authority linkage; LR v1.0 DOI; author creators; listed LR files and MD5s; LR has source-listed train/validation/test file groups; author code repository exists and declares GPL-3.0 for code.

**Unverified:** dataset and image usage rights; local files; actual annotation schema/linkage; resolved split memberships; checksums of local data; leakage; HR release metadata; RSVQAxBEN release/provenance; official evaluator; actual benchmark model compatibility.

## 10. Validation and immutability

`python -m pytest -q tests/test_rsvqa_provenance.py tests/test_rsvqa_readiness.py tests/test_real_evaluation_readiness.py tests/test_evaluation_infrastructure.py tests/test_evaluation_dry_run.py tests/test_agent_controller.py tests/test_evidence_schema.py --basetemp=.pytest-phase3n4-core` — **74 passed in 6.87s**.

`python -m pytest -q tests/test_multispectral_projector.py tests/test_sar_projector.py tests/test_temporal_contracts.py tests/test_temporal_rgb_adapter.py tests/test_grounding_contracts.py --basetemp=.pytest-phase3n4-modality` — **27 passed in 9.03s**.

New tests exercise immutable primary source evidence, mutable/unpinned rejection, annotation/image/underlying-image license separation, acceptance split/linkage/provenance/checksum/leakage states, deterministic empty bounded-acquisition receipts, and evaluator-revision non-claims. Full benchmark execution remains forbidden.

Pipeline 3, CROMA, `hybrid_830d`, scientific metrics/splits/checkpoints/receipts, S2 artifacts, SAR/temporal/grounding freezes, VRSBench/CDVQA/hidden statuses and the scientific baseline are unchanged.

**No benchmark inference ran. No benchmark score was generated.**
