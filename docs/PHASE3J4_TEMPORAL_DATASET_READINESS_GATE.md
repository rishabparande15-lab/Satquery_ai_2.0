# Phase 3J.4 — Temporal Dataset Re-evaluation and Acquisition Gate

Final status: **TEMPORAL_TRAINING_BLOCKED**

## 1. Objective

This gate re-audits the earlier temporal candidates and searches official
releases for a source meeting every SatQuery training-readiness requirement.
No temporal model, adapter, labels, scientific artifact, split, or temporal
contract was changed.

## 2. Previous blockers

CDVQA remains missing an auditable image release and pair metadata. ChangeChat
and LEVIR-CC retain explicit pre/post pairing and valid component licenses but
lack dates, record-level registration evidence, and a bounded fixture route.
The Phase 3J.3 contract remains architecture only; it does not change data
readiness.

## 3. Existing candidate re-evaluation

| Source | Re-evaluation result |
|---|---|
| CDVQA | Official code `cc5893123dd32326de38745b65d2ffe45055937b` remains available. No author-published, pinned T1/T2 image release, image license, manifest, timestamps, sensor metadata, or registration evidence was located. |
| ChangeChat-105k / DeltaVLM | Annotations remain pinned at `2e467fccf34d24dfebff544858145ecb4c809acd`, CC-BY-4.0. DeltaVLM code is at `6c9bd60940286e59771e14d63012c6682504136c`. No ChangeChat validation annotations, dates, registration metadata, or granular image fixture route appeared. |
| LEVIR-CC | Author-linked image archive remains pinned at `881887bfc8a0f856f9059bcedf74c388e0d92ad7`, Apache-2.0. It remains a 2.68 GB monolithic archive with no per-file manifest or record dates/registration fields. |

## 4. Newly identified candidates

The official Google Research [RSRCC dataset](https://huggingface.co/datasets/google/RSRCC)
is a new, materially more complete candidate. It co-hosts images and metadata,
documents three splits and before/after paths, and provides direct streaming or
individual object paths. Its pinned dataset revision is
`7898de7bfd08bc404d9a92e1caaa9dce91b0c3ea`; the linked Google Research code
revision observed is `9b75ae0ac103995624a4f5da64554de373f55f82`.

Other official sources found do not clear the gate: [QAG-360K/VisTA](https://github.com/like413/vista)
has research-only use, an access-controlled test set, and upstream-provenance
dependencies; [RSCC](https://github.com/Bili-Sakura/RSCC) is disaster change
captioning but defers to xBD/EBD upstream terms and does not disclose a full
auditable split/temporal metadata path in the inspected release. They are not
substitutes for a ready source.

## 5. Temporal supervision and image availability

RSRCC provides explicit `CHANGE_VQA`: before/after images plus generated
question-and-answer text (yes/no and multiple choice). It is not a
`CHANGE_DESCRIPTION` caption source, nor a `CHANGE_GROUNDING` source. The
official card calls its image pairs temporally aligned and its annotations
semantic change Q&A. All image objects and per-split metadata CSVs are hosted
in the same official dataset repository; the card declares Apache-2.0.

## 6. T1/T2 linkage and temporal metadata

RSRCC metadata rows contain `before_file_name`, `after_file_name`, and `text`.
The shared UUID filename stem is an explicit pair key, with before/after as the
documented order. A row is the annotation linkage, but there is no separate
source-issued `annotation_id`; fixture records retain `annotation_id: null` and
a CSV-row locator plus annotation hash.

The release provides no T1/T2 timestamps, dates, acquisition interval, sensor
identity, or source-scene metadata. Thus its order is **VERIFIED as
before-to-after semantic ordering**, but timestamp metadata is **UNKNOWN** and
cannot support time-conditioned training claims.

## 7. Spatial correspondence and registration

The source documents temporally aligned pairs, sufficient to classify them as
**SPATIALLY_CORRESPONDING** for semantic change VQA. It does not supply CRS,
transforms, alignment residuals, or a reproducible registration procedure, so
they are not `COREGISTERED`. This is insufficient for a claim of pixel-precise
change grounding.

## 8. License, pinning, splits, and leakage

RSRCC images and annotations are co-hosted under the dataset card's
Apache-2.0 declaration; code remains a separate Google repository. The pinned
dataset revision provides immutable source selection, and local fixture hashes
bind retrieved objects to it.

The official `train`, `val`, and `test` metadata files were boundedly obtained
and audited:

| Split | Annotation rows | Unique ordered pairs |
|---|---:|---:|
| train | 87,016 | 6,417 |
| val | 17,116 | 1,260 |
| test | 21,999 | 1,826 |

Exact ordered-pair and UUID-key overlap was zero for train/val, train/test,
and val/test. This is a metadata-level audit: it does not establish geographic
or near-duplicate independence because the release omits those fields.

## 9. Bounded acquisition and structural validation

Unlike LEVIR-CC, RSRCC exposes individual PNG objects. Three validation pairs
were fetched directly—no archive download—and verified as 512x512 RGB PNGs.
Their metadata hashes, image hashes, pair keys, row locators, split audit, and
source revision are in
`artifacts/test_fixtures/temporal_change_fixture_manifest.json`.

All three were accepted by `TemporalInput` as ordered optical pairs with
`SPATIALLY_CORRESPONDING` status. This validates file/linkage/contract
structure only. It computes neither temporal features nor change labels.

## 10. S2/SAR compatibility and grounding

RSRCC is RGB optical imagery, not canonical raw Sentinel-2 `[12,120,120]` or
Sentinel-1 VV/VH `[2,120,120]`. A separately validated RGB temporal image
adapter would be required; current S2 and SAR adapters are unchanged. RSRCC
provides no masks, boxes, points, polygons, or verified region-text geometry;
Phase 2F stays fail-closed. None of the audited sources provides verified SAR
temporal language supervision.

## 11. Dataset comparison

| Dataset | Language task | Images / linkage | Dates / registration | License / pinning | Splits and bounded fixture | Status |
|---|---|---|---|---|---|---|
| CDVQA | CHANGE_VQA | Image release unavailable | unknown / unknown | image license unverified; code commit only | annotation files only | BLOCKED |
| ChangeChat + LEVIR-CC | CHANGE_VQA, CHANGE_DESCRIPTION | explicit A/B paths | dates unknown; spatially corresponding only | CC-BY annotations + Apache image archive; pinned commits | partial splits; monolithic archive blocks fixture | BLOCKED |
| RSRCC | CHANGE_VQA | co-hosted before/after paths and metadata rows | dates unknown; spatially corresponding, registration not reproducible | Apache-2.0; pinned revision | train/val/test, metadata overlap audited, 3-pair fixture succeeded | BLOCKED |
| QAG-360K | CHANGE_VQA, CHANGE_GROUNDING | upstream-dependent | not established | research-only / source chain incomplete | test requires contact | BLOCKED |
| RSCC | CHANGE_DESCRIPTION | pre/post disaster imagery claimed | per-pair metadata not established | CC-BY claim conflicts with upstream-term dependency to audit | full split protocol not established | BLOCKED |

## 12. Final training readiness

No source passes every required gate. RSRCC is the strongest newly auditable
source but fails these non-relaxable gates:

1. no T1/T2 dates or acquisition interval;
2. no sensor/source-scene metadata or per-sample upstream provenance;
3. no registration parameters, residuals, CRS, or reproducible alignment;
4. no separate source-issued annotation ID;
5. no geographic/near-duplicate leakage audit from the released metadata;
6. RGB input requires a new validated adapter, not the existing S2/S1 paths.

Therefore the only valid decision is **TEMPORAL_TRAINING_BLOCKED**.

## 13. Phase 3J.5 gate

Phase 3J.5 is **TEMPORAL BENCHMARK / DATASET DEFERRAL + CONTINUED ARCHITECTURE
HARDENING**. A training pilot becomes eligible only when a source supplies all
missing temporal, registration, provenance, and compatibility evidence.

The frozen scientific baseline remains accuracy `65.0%`, MAE
`4.1034286734 pp`, and RMSE `9.5873312123 pp`.

## Sources

- [RSRCC dataset card](https://huggingface.co/datasets/google/RSRCC)
- [Google Research remote-sensing repository](https://github.com/google-research/remote-sensing/)
- [DeltaVLM / ChangeChat-105k](https://github.com/hanlinwu/DeltaVLM)
- [Official LEVIR-CC repository](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset)
- [CDVQA repository](https://github.com/YZHJessica/CDVQA)
