# Phase 3H.2 — SAR-Language Supervision Source Audit

Status: **SAR_VQA_SUPERVISION_BLOCKED**

## 1. Objective and current blocker

This is a research and provenance audit; it neither downloads a corpus nor
trains the SAR projector. Phase 3H.1 proved the project has genuine linked
S1 imagery, but BigEarthNet.txt has no per-record declaration that its
optical/image-language questions are valid SAR-only targets. That conclusion
is unchanged.

## 2. Search method

On 2026-09-22, the audit examined official project repositories, the linked
papers/dataset pages, and author-published Hugging Face release pages. A
candidate was accepted as SAR-language evidence only where the source itself
describes language paired with SAR observations or an explicit optical–SAR
task. Code-repository licenses were not treated as image or annotation
licenses. Missing source facts below are deliberately marked **UNKNOWN**.

## 3. Candidate dataset inventory

| Dataset | Modality / task | SAR-native supervision | Image / language linkage | License | Split / provenance | SatQuery compatibility | Status |
|---|---|---|---|---|---|---|---|
| [SARLANG-1M](https://github.com/Jimmyxichen/SARLANG-1M) | SAR; captions, VQA, location/position tasks | VERIFIED | Images plus JSON/CSV text release; official project says SAR images and text are paired | UNKNOWN for dataset and constituent imagery | Text/image release exists; per-source provenance and official split audit still required | Conceptually supports VQA, captioning, and location tasks | BLOCKED pending license/release-manifest review |
| [SAR-TEXT / SAR-VQA](https://github.com/YiguoHe/SAR-TEXT) | SAR; image-text, captioning, dialogue VQA | VERIFIED | Official release distinguishes optical `RS-VQA` from SAR `SAR-VQA` | Code: Apache-2.0; data: UNKNOWN | Baidu release exists; counts, split, SAR-VQA schema, source-image provenance UNKNOWN until release inspection | Conceptually supports captioning and VQA | BLOCKED pending data/license/schema audit |
| [FSAR-Cap](https://github.com/hitjiao/FSAR-Cap) | SAR; captioning | VERIFIED | Official project says 14,480 SAR images and 72,400 image-text pairs | UNKNOWN for dataset and FAIR-CSAR imagery | Science Data Bank download exists; official split and record schema UNKNOWN | Captioning only; does not substitute for VQA | BLOCKED pending license/split/schema audit |
| [SAREval](https://github.com/Dilys2022/SAREval) | SAR; evaluation prompts, labels, captions, grounding | VERIFIED for benchmark tasks | JSON maps relative SAR image paths to targets/prompts | UNKNOWN | Full images/annotations are published, but source-image lineage and training-use terms require audit | Evaluation and structural-format compatibility; training use unverified | BLOCKED / evaluation candidate only |
| [OSVQA / TGFNet](https://arxiv.org/abs/2411.15770) | aligned optical–SAR; VQA | VERIFIED as a joint task, not SAR-only | Paper describes aligned pairs and QA annotations | UNKNOWN | Paper documents construction; a concrete official release manifest, terms, and current split files were not located | Phase 3I optical–SAR reasoning only | JOINT_OPTICAL_SAR, BLOCKED pending access/license audit |

## 4. SAR-VQA candidates

SARLANG-1M is explicitly a SAR vision-language benchmark: its official
repository reports more than one million SAR image-text pairs from more than
59 cities and VQA-style tasks including identification, classification,
counting, referring, positioning, and reasoning. It says original TIFF and
preprocessed PNG imagery plus JSON/CSV text are published through its
author-owned Hugging Face release. Its stated source imagery includes
SARDet-100K, SpaceNet6, DFC2023, and OpenEarthMap-SAR. This is strong evidence
of SAR-language association, but does not substitute for a license and
per-source split/duplicate audit. The public release lists 37.8 GB total,
with 31 GB original imagery, 6.77 GB preprocessed imagery, and 17.1 MB text;
none was downloaded.

SAR-TEXT’s official project explicitly calls `SAR-VQA_conv.json` a **SAR
Image–Text Dialogue Dataset**, while separately identifying `RS-VQA_conv.json`
as optical. That distinction passes the modality-validity rule. The currently
published evidence does not expose the SAR-VQA JSON, source image manifest,
data license, counts, sensor/polarization, or official train/validation/test
division, so it cannot yet become a reproducible training input.

SAREval is a SAR VLM benchmark with task JSON fields such as `image_path`,
`ground_truth`, `ground_truth_option`, and prompt templates. It has multiple
choice, caption, and grounding evaluation metrics. It is not treated as a
training source without terms and source-lineage confirmation.

## 5. SAR-captioning and grounding candidates

FSAR-Cap is an official 2026 SAR-captioning release built on FAIR-CSAR. The
authors report 14,480 images and 72,400 pairs, two-stage annotation with
manual verification/expert supplementation, and a Science Data Bank download.
It is a captioning candidate only; no VQA or region-grounding format is
claimed here.

SARLANG-1M’s image-description category has 45,650 text records. Its region
referring (221,450) and object-positioning (106,171) categories suggest future
grounding compatibility, but the public documentation consulted does not
establish the coordinate representation, coordinate frame, or region-label
file schema. Therefore it is not approved for Phase 2F-style grounding.

SAREval reports visual-grounding evaluation and JSON image/target mappings,
but the coordinate format and data-use license must be inspected before any
training or coordinate conversion.

## 6. Optical–SAR language candidate

OSVQA is **JOINT_OPTICAL_SAR**, not a license to transfer labels to S1 alone.
The paper describes aligned optical–SAR pairs and QA labels generated after
manual image-attribute annotation and templates. Its reported version differs
between the preprint (6,008 pairs / 1,036,694 QA) and later publisher listing
(7,108 pairs / 1,131,730 QA); the exact frozen version must be selected from a
specific official release before use. It belongs to Phase 3I, never Phase 3H
single-modal SAR training.

## 7. License, split, leakage, and provenance analysis

No audited candidate presently has all of the following verified in one
authoritative, version-pinned release: dataset/annotation/image license,
redistribution terms, frozen train/validation/test manifest, image-level and
scene/geographic duplicate policy, complete acquisition metadata, and a hash
manifest. SARLANG-1M acknowledges several upstream image datasets, so its
eventual use requires tracing each source license and checking overlap with
SatQuery’s 5,000 BigEarthNet areas rather than assuming disjoint geography.
The same check is required for SAR-TEXT/HRSID, FSAR-Cap/FAIR-CSAR, SAREval,
and OSVQA. No external sample has been merged into SatQuery and no claim of
non-overlap is made.

The target future record can preserve the required contract:
`sample_id`, `image_id`, `sar_id`, `annotation_id`, `split`, `question`,
`answer`, `caption`, `task_type`, `sensor`, `polarization`, `source_dataset`,
`source_revision`, `license`, `provenance`, and `preprocessing_version`.
No candidate’s incomplete fields may be invented.

## 8. Training readiness decision and Phase 3H.3 recommendation

**No candidate is presently cleared for training.** Four sources supply
authoritative SAR-language evidence; one supplies authoritative joint
optical–SAR VQA evidence. However, all require source-specific release,
license, split, and provenance inspection before they meet every Phase 3H.3
gate. The exact remaining requirement is a version-pinned, license-verified,
downloadable official release with image-to-annotation IDs, official splits,
and sufficient acquisition/provenance metadata. For SAR grounding, coordinate
semantics are an additional requirement.

Phase 3H.3 should select **one** approved source, ingest only its small
metadata/manifest first, validate its license and image/annotation/split
contracts, and conduct an external-overlap audit before any imagery download
or training. BigEarthNet.txt remains optical/image-language foundation data;
it was not reclassified.

## 9. Scientific boundary

No scientific predictor, CROMA component, representation, receipt, split,
S2 adapter/checkpoint, or SAR projector was modified. The locked scientific
baseline remains 65.0% accuracy, 4.1034286734 pp MAE, and 9.5873312123 pp
RMSE.
