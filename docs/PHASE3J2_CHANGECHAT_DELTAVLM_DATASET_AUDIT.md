# Phase 3J.2 — ChangeChat / DeltaVLM Upstream Data and Temporal Supervision Audit

Final status: **TEMPORAL_TRAINING_BLOCKED**

## 1. Objective

This is a provenance and dataset-readiness audit of ChangeChat-105k and
DeltaVLM as a potential source of bi-temporal change-language supervision for
SatQuery. It performs no acquisition of imagery, no temporal modelling, and no
change to the scientific predictor, S2 adapter, SAR projector, data splits, or
receipts.

## 2. ChangeChat / DeltaVLM identity

The authors' official [DeltaVLM repository](https://github.com/hanlinwu/DeltaVLM)
describes two distinct artifacts:

| Component | Identity and availability | Revision / license status |
|---|---|---|
| Code | DeltaVLM implementation; a bi-temporal encoder, difference-perception module, instruction-guided Q-former, and frozen Vicuna-compatible LLM path | Git `6c9bd60940286e59771e14d63012c6682504136c` observed at official `HEAD`; no repository `LICENSE` was located, so **CODE_LICENSE_UNVERIFIED** |
| Annotations | Author-published `hlwu/changechat-105k`, eight JSON files, about 87 MB; no imagery | Git revision `2e467fccf34d24dfebff544858145ecb4c809acd` observed at official dataset `HEAD`; **CC-BY-4.0** for annotations |
| Images | LEVIR-CC archive, separately hosted by the LEVIR-CC authors and linked by both official DeltaVLM and ChangeChat-105k pages | Official archive repository revision `881887bfc8a0f856f9059bcedf74c388e0d92ad7` observed at `HEAD`; dataset card declares **Apache-2.0** |
| Model | `hlwu/DeltaVLM` checkpoint, separate from code and data, and dependent on a separately obtained Vicuna-compatible base model | model card labels its license `other` and defers to project, dataset, and base-model terms: **MODEL_LICENSE_UNVERIFIED** |

The revisions above are immutable commit identifiers obtainable with `git
ls-remote`; they are not tags or releases. A future acquisition must specify
these revisions explicitly rather than use `main`.

DeltaVLM is **dataset + model + annotation framework**, not a drop-in SatQuery
component. Its released architecture targets its own CLIP/LAVIS/Vicuna-era stack
and does not make its checkpoint or temporal modules reusable with Qwen without
a separate execution and compatibility audit.

## 3. Upstream image sources

The official DeltaVLM documentation explicitly states that it does not
redistribute imagery: ChangeChat-105k paths point to the [official
LEVIR-CC repository](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset), which
links the author-published `lcybuaa/LEVIR-CC` archive. The latter contains one
2.68 GB `Levir-CC-dataset.zip`, with `images/{train,val,test}/{A,B}` and
`LevirCCcaptions.json`.

Thus the authoritative components are deliberately separate:

```text
DeltaVLM code  -> training/evaluation implementation
ChangeChat-105k -> questions, answers, instructions, relative image paths
LEVIR-CC       -> actual A/B PNG image pairs and original caption JSON
```

Real T1/T2 imagery is **AVAILABLE AS A SEPARATE FULL ARCHIVE**, not included in
the annotation repository or the code repository. The official source does not
publish a per-pair manifest or per-file checksums in the material inspected.

## 4. Temporal supervision

**TEMPORAL_SUPERVISION_VERIFIED.** The official dataset card and DeltaVLM
documentation identify every record as a question/answer over a bi-temporal
pre/post pair. The paper defines its remote-sensing image change analysis task
as instruction-guided analysis of visual differences. This is substantive
change language rather than a filename-derived pairing.

The released tasks are:

| SatQuery mapping | Evidence in ChangeChat-105k |
|---|---|
| `CHANGE_DESCRIPTION` | captions; five-reference caption test set |
| `CHANGE_VQA` | binary change, counting, open-ended QA, and dialogue prompts/answers |
| `CHANGE_GROUNDING` | coarse 3x3 location text only; not a geometric grounding label |
| Other | `changeflag` indicates unchanged (`0`) versus changed (non-zero) pairs |

The instructions are hybrid rule-based and GPT-assisted. That provenance is
documented, not hidden; it means generated language must remain identified as
such in any later quality evaluation. It does not invalidate its explicit
pair-conditioned supervision.

## 5. T1/T2 pair linkage

Linkage is explicit, not inferred from similar names. Instruction records use
an `image` array in positional order, e.g.
`["train/A/train_000001.png", "train/B/train_000001.png"]`; `<image>` markers
in the conversation refer to those two entries. Caption-test records instead
use explicit `image_A` and `image_B` fields. The upstream image tree uses the
same paths.

`A` is explicitly documented as the pre-phase view and `B` as the post-phase
view, so **temporal ordering is VERIFIED**. The per-record `id` is an
instruction-record identifier; it is not a canonical pair identifier. A stable
future pair key can be the exact ordered path tuple plus the pinned annotation
and image revisions, but it must not substitute a new source-issued pair ID.

## 6. Temporal metadata

| Field | Status |
|---|---|
| A is pre-event / B is post-event ordering | **VERIFIED** |
| Per-pair T1 date or timestamp | **UNKNOWN** |
| Per-pair T2 date or timestamp | **UNKNOWN** |
| Per-pair acquisition interval | **UNKNOWN** |
| Dataset-level temporal detail | **PARTIAL**: the corresponding LEVIR literature describes multi-temporal imagery, but no record-level dates were released in the inspected materials |

No filename, split name, or A/B ordering is treated as a date.

## 7. Spatial correspondence

The sources say A and B are pre/post views of the same tile, which supports
**SPATIALLY_CORRESPONDING**. They do not publish per-pair CRS, transforms,
registration residuals, affine alignment records, or a proof of pixel-level
co-registration. Therefore this audit does **not** label the data
`COREGISTERED`.

## 8. Sensor / modality

LEVIR-CC is optical, RGB, PNG imagery. The DeltaVLM paper reports 256 x 256
image patches at 0.5 m/pixel. It is neither Sentinel-2 multispectral raw data
nor Sentinel-1 SAR: no sensor identity, reflectance scale, S2 band order, SAR
polarization, or georeferencing is supplied in the release material. It is
therefore **ADAPTER_REQUIRED** and cannot enter SatQuery's canonical S2
`[12,120,120]` or SAR `[2,120,120]` contracts without a separately validated
image adapter. No adapter is implemented here.

## 9. License

Licenses are evaluated per component:

| Component | Finding |
|---|---|
| ChangeChat-105k annotations | **VERIFIED: CC-BY-4.0**; attribution required |
| Official LEVIR-CC archive | **VERIFIED from author-linked dataset card: Apache-2.0** |
| DeltaVLM code | **LICENSE_UNVERIFIED**; no license file was found in the inspected official code repository |
| DeltaVLM model checkpoint | **LICENSE_UNVERIFIED** (`other` on its model card; required base-model terms are separate) |

The imagery and annotations can satisfy the data-license gate only when their
respective pinned sources and required attributions are retained. Code/model
licenses are not silently inherited from the image or annotation licenses.

## 10. Release / pinning

All three data-bearing sources can be pinned by the commit IDs in section 2.
This is **SOURCE_PINNING_VERIFIED AT THE REVISION LEVEL**, but weaker than an
author-provided release archive with a published archive checksum. The image
archive has a stable filename and source-repository revision but no inspected
per-file SHA-256 manifest. A later acquisition must calculate and preserve the
archive and selected-file hashes locally.

## 11. Split / leakage

LEVIR-CC publishes image directories for 7,590 train, 1,438 validation, and
2,135 test pairs. ChangeChat-105k publishes a mixed training file and
task-specific test files, but it does not publish a ChangeChat validation JSON.
The numerous task-specific test annotations intentionally reuse image pairs;
they are not independent test populations. The localization-augmented training
file also has the same reported cardinality as training and must be treated as
an alternate annotation view, not merged as new pairs.

Without materializing and hashing the pinned JSONs and archive, this audit
cannot measure duplicate paths, T1/T2 reuse across annotations, geographic
overlap, or near duplicates. No split is created, merged, or adopted by
SatQuery. **OFFICIAL_SPLITS_PARTIAL; LEAKAGE_AUDIT_PENDING.**

## 12. Annotation structure

Instruction files are JSON arrays with `id`, ordered `image` paths,
`changeflag`, and ShareGPT-style `conversations` (`from`, `value`). Caption
evaluation uses `image_A`, `image_B`, `changeflag`, and five `captions`.
Question/answer linkage is within each conversation turn sequence. The source
does not expose an independent annotation UUID for every answer beyond the
record identity and positional conversation turn.

## 13. Grounding

ChangeChat localization answers express locations in a 3x3 textual grid. They
contain **no bbox, point, polygon, or mask geometry**. Optional upstream
LEVIR-CC change-detection masks are mentioned by the DeltaVLM documentation,
but this audit did not acquire or verify a record-level binding between those
masks and ChangeChat annotations. Phase 2F consequently remains fail-closed.

## 14. Small sample acquisition

No fixture was acquired. The candidate passes the temporal-supervision,
annotation/image license, and revision-pinning gates, but the only official
image payload located is a 2.68 GB monolithic archive. Downloading it to obtain
3–5 pairs would violate this phase's bounded, no-large-data acquisition rule.
No author-published partial archive, per-pair endpoint, or verified per-file
manifest was located.

Accordingly **SAMPLE_ACQUISITION_BLOCKED_BY_DISTRIBUTION_GRANULARITY** and no
structural image or adapter validation can be claimed.

## 15. SatQuery compatibility

A future, separately approved pathway is conceptual only:

```text
ordered LEVIR A (pre) -> validated optical temporal adapter -> T1 tokens
ordered LEVIR B (post) -> validated optical temporal adapter -> T2 tokens
T1 + T2 + explicit difference evidence -> temporal representation -> Qwen -> change language
```

This candidate is a temporal-language source, while BigEarthNet.txt remains a
single-image optical-language source. No labels or splits are cross-transferred.

## 16. Remaining blockers

1. Per-pair timestamps and acquisition intervals are not released.
2. Pixel-level co-registration evidence and geospatial metadata are not
   released.
3. There is no ChangeChat validation annotation split and no completed
   duplicate/scene/geographic leakage audit.
4. The official image distribution is one large archive, preventing the
   permitted bounded 3–5-pair fixture acquisition.
5. No archive/per-file manifest is published, so acquisition-time hashes and
   JSON-to-image structural linkage remain unverified.
6. RGB LEVIR imagery does not satisfy existing S2/S1 raw-input contracts; a
   new validated adapter contract is required.
7. Grounding is textual-grid only; no verified geometry may enter Phase 2F.

## 17. Temporal training readiness

| Required gate | Result |
|---|---|
| Real T1/T2 imagery available | PASS, separately downloadable full archive |
| Explicit change-language supervision | PASS |
| Exact ordered T1/T2 linkage | PASS |
| Annotation linkage | PASS within JSON records |
| Temporal ordering | PASS |
| Spatial correspondence | PARTIAL: spatially corresponding, not co-registered |
| License verified | PASS for annotations and author-linked imagery; code/model are separate and unverified |
| Source revision pinned | PASS at Git revision level |
| Official splits verified | PARTIAL |
| Provenance complete | PARTIAL |
| Small sample acquisition | BLOCKED |
| Structural validation | BLOCKED |

The required all-gates criterion is not met. Final decision:
**TEMPORAL_TRAINING_BLOCKED**.

## 18. Phase 3J.3 gate

Phase 3J.3 must remain closed until an authorised, bounded acquisition can
materialize 3–5 pinned pairs and verify path linkage, image hashes, RGB
structure, split lineage, duplicate policy, and temporal adapter input
contract. It must also preserve the explicit limitation that timestamps and
pixel-registration evidence are absent unless an official record-level source
is provided.

## Sources

- [Official DeltaVLM repository](https://github.com/hanlinwu/DeltaVLM)
- [Official ChangeChat-105k dataset card](https://huggingface.co/datasets/hlwu/changechat-105k)
- [Official LEVIR-CC repository](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)
- [Author-linked LEVIR-CC archive card](https://huggingface.co/datasets/lcybuaa/LEVIR-CC)
- [DeltaVLM paper](https://doi.org/10.3390/rs18040541)
