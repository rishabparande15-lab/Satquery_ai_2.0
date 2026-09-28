# Phase 3H.3 — SAR Dataset Verification and Training Readiness Gate

Final decision: **SAR_TRAINING_BLOCKED**

## 1. Objective

This strict verification re-audits the five Phase 3H.2 candidates against the
training gates: explicit SAR-language supervision, an official source,
image-to-annotation linkage, license, immutable pin, official splits,
provenance, usable input, bounded sample validation, and projector path. No
dataset was downloaded and no model was trained.

## 2. Verification matrix

| Dataset | Version / revision | SAR-native language | Image / annotation / linkage | Official train / val / test | License | Pin / hashes | Modality and input | Final status |
|---|---|---|---|---|---|---|---|---|
| [SARLANG-1M](https://huggingface.co/datasets/YiminJimmy/SARLANG-1M/tree/cd5a4cb162ad5cbb4787a51b4372f434df773178) | HF commit `cd5a4cb162ad5cbb4787a51b4372f434df773178` VERIFIED | VERIFIED | Official docs describe SAR image folders, JSON/CSV text, and SARVQA1/2 source-specific annotations; exact record schema still requires metadata inspection | train/test VERIFIED by filename; validation UNKNOWN | **LICENSE_UNVERIFIED** for corpus and upstream images | Dataset commit VERIFIED; per-archive checksums / manifest UNKNOWN | Original TIFF and preprocessed PNG exist; sensor/polarization/channel values UNKNOWN; ADAPTER_REQUIRED | BLOCKED |
| [SAR-TEXT](https://github.com/YiguoHe/SAR-TEXT) | mutable `main`, 17 commits; SOURCE_PINNING_WEAK | VERIFIED | Official release labels `SAR-VQA_conv.json` SAR dialogue; schema/image IDs unavailable without release inspection | UNKNOWN / UNKNOWN / UNKNOWN | Repository code Apache-2.0; data/images/annotations **LICENSE_UNVERIFIED** | No tag, release checksum, or manifest verified | format/sensor/polarization/preprocessing UNKNOWN; ADAPTER_REQUIRED | BLOCKED |
| [FSAR-Cap](https://github.com/hitjiao/FSAR-Cap) | mutable `main`, 8 commits; SOURCE_PINNING_WEAK | VERIFIED, captioning only | 14,480 SAR images / 72,400 pairs claimed; exact record IDs/schema UNKNOWN | UNKNOWN / UNKNOWN / UNKNOWN | **LICENSE_UNVERIFIED** for FSAR-Cap and FAIR-CSAR imagery | No pinned data archive/hash verified | format/sensor/polarization UNKNOWN; ADAPTER_REQUIRED | BLOCKED |
| [SAREval](https://github.com/Dilys2022/SAREval) | mutable repository; no release pin verified | VERIFIED as evaluation prompts/labels | JSON `image_path` maps task images to targets/prompts VERIFIED | official train/validation/test UNKNOWN | **LICENSE_UNVERIFIED** | archive checksum / source-image provenance UNKNOWN | JPG/PNG evaluation images; raw polarizations UNKNOWN; not a verified raw VV/VH source | BLOCKED |
| [OSVQA](https://arxiv.org/abs/2411.15770) | paper reports conflicting preprint/later counts; exact release UNKNOWN | NOT APPLICABLE to SAR-only; VERIFIED joint optical–SAR VQA | aligned pairs and QA generation described; downloadable image/annotation IDs UNKNOWN | UNKNOWN / UNKNOWN / UNKNOWN | **LICENSE_UNVERIFIED** | no concrete official release/archive manifest verified | paired optical–SAR; raw bands/polarization UNKNOWN; Phase 3I only | BLOCKED; `PHASE_3I_CANDIDATE` |

## 3. License and release verification

No paper, public repository, or public download link was accepted as a data
license by implication. SAR-TEXT displays Apache-2.0 for its repository, but
that text does not establish rights for its Baidu-hosted SAR images or
annotations. FSAR-Cap links to Science Data Bank but the official repository
does not state corpus/image terms. SAREval publishes download instructions,
not a verified data license. SARLANG-1M’s official Hugging Face tree offers a
pin-able commit and the archives, but its visible release information does not
state the corpus or constituent-image license. OSVQA’s paper does not supply a
concrete, license-bearing release manifest.

Therefore every candidate is `LICENSE_UNVERIFIED` and fails the training gate.
No bounded imagery acquisition is authorized under this phase’s own rule.

## 4. Split, linkage, and provenance verification

SARLANG-1M is the only candidate for which the official repository documents
specific split naming: caption training/test files and `SARVQA1_train/test`,
`SARVQA2_train/test`. It identifies SARVQA1 as SARDet-100K-derived text from
bounding boxes, and SARVQA2 as text for SpaceNet6, DFC2023, and
OpenEarthMap-SAR images. This demonstrates intended SAR association, but there
is no documented validation subset, split manifest hash, duplicate audit, or
geographic-overlap audit. The actual text archive has not been fetched, so
image IDs, annotation IDs, linkage columns, hashes, and scene/image
intersections are unverified.

SAR-TEXT clearly distinguishes optical `RS-VQA_conv.json` from the SAR
`SAR-VQA_conv.json`, but the SAR file is in an external download rather than
the repository. Its fields, split policy, image IDs, and image-to-dialogue
relation are unknown. FSAR-Cap’s count and image-text pairing are published,
but its split and exact linkage fields are not exposed. SAREval’s published
JSON structure gives a relative `image_path`, target, option, and prompt
templates; this is sufficient structural linkage for its evaluation files but
not source acquisition/polarization provenance or a training split.

No candidate has been compared against the BigEarthNet 5,000-area population;
external image/scene/geographic overlap and near-duplicate risk remain
unknown. No external data is merged into SatQuery.

## 5. SAR modality and annotation compatibility

SatQuery’s training contract is raw canonical `[2,120,120]` VV/VH. SARLANG-1M
documents original TIFF plus optional preprocessed PNG, but not the released
per-record polarization/channel/value conventions in the evidence inspected.
It is thus **ADAPTER_REQUIRED**, not directly compatible, until a small,
licensed sample proves a safe conversion. SAR-TEXT, FSAR-Cap, and OSVQA lack
the required raw modality details in their public release documentation.
SAREval explicitly uses JPG/PNG task images; it is not a verified raw VV/VH
adapter source.

SARLANG-1M supplies captions and VQA-style records. SAR-TEXT claims SAR
dialogue VQA and image-text/caption tasks. FSAR-Cap supplies captions only.
SAREval provides benchmark prompts, target labels, captions, and grounding
evaluation files; it must not be relabeled as training supervision. OSVQA is
explicitly joint optical–SAR VQA; it does not establish a single-SAR subset.

## 6. Sample acquisition and projector compatibility

**NOT RUN.** The gate requires explicit supervision, license, and pinning
before any 3–5 image acquisition. Although SARLANG-1M has a pinned dataset
commit and explicit SAR supervision, it fails the license gate. Consequently
there is no lawful/auditable external sample to normalize or pass through
`SARProjector`; no structural result is claimed.

## 7. Remaining blockers and readiness

There are five candidates audited, zero fully verified, and five blocked.
Four have explicit SAR-language evidence; none has verified usable image and
annotation licenses. One is a joint optical–SAR Phase 3I candidate, not a
Phase 3H single-SAR source.

To lift the block, obtain for one official source: a version-pinned release
with explicit data/image/annotation terms; archive or per-file checksums; a
downloadable official split manifest including validation policy; exact image
and annotation IDs; source sensor/polarization/preprocessing metadata; and
permission to use a 3–5 sample subset. Then perform the bounded acquisition,
duplicate/overlap audit, and untrained `[B,16,2048]` projector test.

BigEarthNet.txt remains optical/image-language foundation data and was not
reclassified. Scientific artifacts, the SAR projector, S2 pathway, CROMA,
splits, and receipts remain untouched; the locked baseline is 65.0% accuracy,
4.1034286734 pp MAE, and 9.5873312123 pp RMSE.

## 8. Source evidence

- SARLANG-1M official repository documents TIFF/PNG availability, JSON/CSV
  text, caption splits, and SARVQA1/2 train/test labels.
- SAR-TEXT official repository distinguishes optical RS-VQA from its external
  SAR-VQA dialogue release and separately offers image-text/caption data.
- FSAR-Cap official repository states its image/text counts, annotation method,
  and Science Data Bank delivery.
- SAREval official repository documents its SAR image folders, JSON/TSV
  mappings, prompt/target fields, and evaluation orientation.
- OSVQA’s official paper describes a joint paired optical–SAR VQA construction,
  not SAR-only supervision.
