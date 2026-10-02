# Phase 3V.2 — Bi-temporal change description

Status: `PHASE3V2_DATA_BLOCKED`.

SatQuery now has a separate `TEMPORAL_CHANGE_DESCRIPTION` contract, controller
capability, dedicated `POST /api/v1/temporal` boundary, and a BI-TEMPORAL UI
mode. It takes an explicitly ordered RGB pair only: T1/BEFORE is pre-phase and
T2/AFTER is post-phase. It is neither the completed optical–SAR route nor the
single-image VQA route. It produces no change mask, bounding boxes, area
estimate, or Change-VQA claim.

## Authoritative basis

The official [LEVIR-CC repository](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)
documents `images/train/A`, `images/train/B`, `images/val/A`, and `images/val/B`,
with A as pre-phase and B as post-phase. The official
[Chg2Cap repository](https://github.com/ShizhenChang/Chg2Cap) provides the
dual-image captioning architecture, preprocessing, checkpoint link, and
caption evaluator. Its evaluator supports BLEU-1/2/3/4, METEOR, ROUGE-L, and
CIDEr. Its stock test script is hard-wired to TEST and was not run.

The Chg2Cap LEVIR checkpoint was downloaded from the author-provided link,
hashed, and strictly loaded into the published architecture. It requires a
501-entry vocabulary. The public preprocessing script derives that vocabulary
from a global `LevirCCcaptions.json` and creates a TEST list as part of the
same process. Opening that monolithic annotation file would violate the
development policy, so no caption is fabricated and no validation metric is
reported.

## Guardrails and result

The API/wrapper rejects missing T1/T2, corrupt or unsupported images, identical
inputs, incompatible dimensions, missing/invalid `PRE_POST` ordering, and any
`test` split. RGB `t1`/`t2` upload roles are dedicated to this new route; the
existing S1+S2 and S2 GeoTIFF upload paths are unchanged.

`TEST_ACCESS = 0`. No LEVIR test image, caption, identity, manifest, score,
or preview was opened or created.

The official repository’s easily exposed archive is monolithic. No
TRAIN/VALIDATION-only image/caption manifest was exposed, and the source
repository did not provide an explicit imagery license in the material audited.
The linked Hugging Face mirror states Apache-2.0, but that mirror label is not
treated as authoritative replacement for original imagery terms.

## Required next step

`PHASE3V2A_LEVIR_CC_DEVELOPMENT_SPLIT_ADMISSION`: obtain an authoritative
TRAIN/VALIDATION-only image and caption/vocabulary manifest (and clarify image
use terms), then run genuine pretrained Chg2Cap validation inference, temporal
sanity controls, real browser success-path verification, and resource profiling.
