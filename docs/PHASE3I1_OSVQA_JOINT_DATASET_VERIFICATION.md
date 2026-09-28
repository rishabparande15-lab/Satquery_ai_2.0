# Phase 3I.1 — OSVQA Joint Optical–SAR Dataset Verification

Final status: **JOINT_OPTICAL_SAR_TRAINING_BLOCKED**

## 1. Objective

This audit determines whether OSVQA can defensibly supervise a future joint
optical–SAR language adapter. It does not train, download imagery, create a
fusion mechanism, or alter either existing modality adapter.

## 2. OSVQA source

The authoritative source located is the authors’ paper, *Text-Guided
Coarse-to-Fine Fusion Network for Robust Remote Sensing Visual Question
Answering* ([arXiv:2411.15770](https://arxiv.org/abs/2411.15770); later
[publisher abstract](https://www.sciencedirect.com/science/article/pii/S0924271625003405)).
The paper describes OSVQA as optical–SAR RSVQA and points readers to the
organization URL `https://github.com/mmic-lcl/`; no concrete official OSVQA
repository, tag, dataset card, archive, release asset, or license-bearing
download manifest was located there during this audit.

The paper identifier is immutable, but **SOURCE_PINNING_WEAK** applies to the
dataset itself. The arXiv version reports 6,008 pairs / 1,036,694 QA; the later
publisher description reports 7,108 pairs / 1,131,730 QA. Without a specific
release, neither count can be selected as the training dataset version.

## 3. Dataset structure and joint-supervision validity

**JOINT_SUPERVISION_VERIFIED at the paper level.** The authors explicitly
describe aligned optical–SAR image pairs, 16 question categories, a fusion
architecture using optical and SAR features, and a modality-quality category
whose questions assess relative optical/SAR reliability. They describe manual
image-attribute annotation followed by template-generated QA; this is an
explicit joint task design, not a geographic co-location inference.

This does not make a downloadable record training-ready. No official file was
available to inspect for pair ID, optical ID, SAR ID, annotation ID, question,
answer, split, or exact image-to-annotation foreign keys. Caption and
region/grounding annotations are NOT VERIFIED.

## 4. License, release, and split gate

| Gate | Status | Evidence / limitation |
|---|---|---|
| Joint optical–SAR supervision | VERIFIED | Paper’s aligned-pair, fusion, and modality-quality task definition |
| Optical image source/license | UNKNOWN | Source datasets and terms not bound to a release manifest |
| SAR image source/license | UNKNOWN | Polarization/image terms not exposed in an OSVQA release |
| Annotation license | UNKNOWN | No dataset license text located |
| Repository/code license | UNKNOWN | No concrete official project repository located |
| Dataset revision/checksum | BLOCKED | No tagged release, dataset revision, archive digest, or manifest |
| Train / validation / test splits | UNKNOWN | No split file located |
| Pair/image/scene duplicate audit | BLOCKED | Requires official sample manifests |
| Small reproducible sample acquisition | NOT RUN | License and release gates fail |

No split or leakage claim can be reconstructed from counts in a paper. Pair,
image, scene, geographic, optical, SAR, and cross-split overlap all remain
unknown.

## 5. Optical and SAR compatibility

OSVQA is **ADAPTER_REQUIRED** for both SatQuery branches. The paper’s public
evidence does not specify released optical band count/order, sensor, raw value
representation, preprocessing, dimensions, or an S2 product mapping. It
therefore cannot be routed to the Phase 3G raw 12-band S2 projector as-is.

Likewise it does not identify released SAR sensor, polarization, channel
count, value scale, file format, or preprocessing. It cannot be assumed to be
SatQuery’s raw `[VV,VH]`, `[2,120,120]` contract and cannot be routed to the
SAR projector without a source-specific decision based on real licensed files.

The paper calls pairs “well-aligned,” verifying intended paired spatial
correspondence. It does not publish transforms, CRS, registration residuals,
or pixel-grid metadata in the evidence inspected. Co-registration is therefore
**SPATIALLY_CORRESPONDING**, not `COREGISTERED` under SatQuery’s strict
geospatial contract.

## 6. Pair linkage, sample acquisition, and adapter test

**NOT RUN.** No verified licensed pinned release provides a bounded 3–5 pair
acquisition route. No optical/SAR bytes, hashes, annotation IDs, split labels,
or pair manifest were acquired; no S2 or SAR projector structural test is
claimed. A paper-level task description cannot substitute for a reproducible
data record.

The eventual provenance record must preserve `dataset_id`, `dataset_revision`,
`pair_id`, `optical_id`, `sar_id`, `annotation_id`, `split`, `sensor`,
`polarization`, source URLs, file and annotation hashes, preprocessing,
adapter versions, co-registration status, license, question, and answer.

## 7. SatQuery architecture implication

If a release satisfies the missing gates, the conceptual path is:

```text
OSVQA optical -> source-specific optical adapter -> S2-like tokens
OSVQA SAR     -> source-specific SAR adapter     -> SAR tokens
                                                   -> future learned fusion -> Qwen
```

No fusion option is selected. Token cross-modal fusion, representation-level
joint projection, and native multi-image Qwen remain future alternatives that
depend on verified source modality/geometry and Qwen interface evidence.
OSVQA cannot unlock single-SAR Phase 3H training.

## 8. Remaining blockers and Phase 3I.2 gate

The failed gates are: official optical/SAR source and licenses; annotation
license; concrete OSVQA release; dataset pin/checksum; official splits;
per-pair IDs/linkage; raw modality and co-registration metadata; bounded
sample acquisition; and real adapter structural validation.

Phase 3I.2 is blocked. Reopen it only after obtaining an author-published,
license-bearing, version-pinned OSVQA distribution with pair manifests,
official splits, modality/preprocessing metadata, and checksums. Then audit at
most five pairs before considering learned fusion.

BigEarthNet.txt remains optical image-language data and is not combined with
OSVQA. The scientific pipeline, CROMA, S2 adapter/checkpoints, SAR projector,
splits, receipts, and Phase 2F semantics are unchanged. The locked scientific
baseline remains 65.0% accuracy, 4.1034286734 pp MAE, and 9.5873312123 pp
RMSE.
