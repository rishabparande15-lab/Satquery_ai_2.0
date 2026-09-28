# Phase 3H.1 — Authoritative S1 Acquisition and SAR VQA Linkage Audit

Status: **SAR_VQA_SUPERVISION_BLOCKED**

## 1. Objective

This audit separates two requirements that cannot be conflated: exact S1/S2
area correspondence and proof that a text annotation is valid SAR-only
supervision. No SAR adapter or Qwen training was performed.

## 2. Current Phase 3H blocker

Phase 3H supplied the structural adapter but had no S1 root. That acquisition
blocker is resolved. The independent supervision-applicability requirement is
not resolved, and is now the sole training blocker.

## 3. Authoritative S1 source

The recovered source is the existing project-pinned BigEarthNet v2 selected
5,000-area archive at
`E:\\SatQuery_ai_2.0\\datasets_2.0\\bigearthnet-v2-5000-20260911T162804Z-1-001.zip`.
It has SHA-256
`b3471e5650bd263367bcf4cc650405ab2981a86621579e21a1b1d08af630c595` and
contains `bigearthnet-v2-5000/metadata.parquet` plus the nested
`BigEarthNet-S1-selected.zip`. This is the source already used by Pipeline 3,
not a substitute public collection. Metadata carries exact `patch_id`,
`s1_name`, and split fields; the S1 archive carries one VV and one VH GeoTIFF
per `s1_name`.

## 4. Bounded acquisition and raw validation

Five non-test fixtures were extracted atomically from that verified nested
archive. TIFF payloads are ignored as local data; their portable receipt is
[s1_sar_vqa_fixture_manifest.json](../artifacts/test_fixtures/s1_sar_vqa_fixture_manifest.json).
All five have VV then VH, float32 single-band GeoTIFFs, 120×120 pixels, 10 m
resolution, 14,400 valid finite pixels per band, null declared nodata,
north-up affine transforms, matched VV/VH CRS/bounds/transforms, and canonical
tensor shape `[2,120,120]`. The five fixtures span EPSG:32629, EPSG:32632, and
EPSG:32635. Existing loader preprocessing remains authoritative: per-channel
mean ±2 standard deviations, clipped to `[0,1]`, nodata represented as zero.

## 5. Exact image linkage

The audit required, for every fixture:

`BigEarthNet.txt image_id == metadata.patch_id` and
`BigEarthNet.txt sar_identity == metadata.s1_name`.

It also required identical inherited splits and rejected either mismatch. The
five resulting complete identifier chains and raw/canonical checksums are in
the fixture manifest. A scan found **71,497** validated binary or multiple
choice VQA records within the 5,000-area source whose dual identities match;
this proves correspondence only, not SAR-label validity.

## 6. Annotation applicability

BigEarthNet.txt source provenance establishes paired/co-registered S1/S2
records and source question/answer text. It does not declare any question or
answer to be a SAR-only target, identify an input modality per VQA row, or
provide a SAR-specific evaluation/label-validity policy. The source questions
include visual scene, extent, adjacency, and season/object appearance claims;
whether each can be resolved from VV/VH is not established by geographic
co-location or by plausibility.

Consequently every candidate VQA row is classified **SAR_UNKNOWN**, not
`SAR_APPLICABLE`; none are silently transferred as S1 training labels. No row
is classified `SAR_NOT_APPLICABLE` merely from intuition, because the record
also lacks the modality evidence needed to make a source-authoritative
negative applicability claim.

## 7. Alternative supervision sources

The repository exposes only future adapter hooks for RSVQA, VRSBench, and
CDVQA; no existing authoritative SAR-language source, license, image linkage,
or annotation linkage is established locally. These remain candidates for a
separately approved source-specific audit, not datasets acquired or used here.

## 8. Projector structural validation

Each real raw S1 fixture was passed through canonical normalization and the
untrained Phase 3H projector. All five produced `[1,16,2048]`. This verifies
the raw-S1 → canonical VV/VH → learned-projector interface only. It does not
produce language output or a learned SAR claim.

## 9. Provenance and remaining blocker

The fixture manifest records the archive identity/checksum, nested member
source, exact identifiers/split, band filenames and SHA-256 values, geospatial
metadata, canonical and normalized tensor hashes, channel order, annotation
ID, and applicability decision. Scientific representations, CROMA, splits,
receipts, S2 adapters, and checkpoints were not read as model inputs or
modified.

**Missing requirement:** publisher or equivalent authoritative documentation
must establish SAR-only applicability for a defined subset of the exact
BigEarthNet.txt VQA records, or an approved SAR-language source must provide
its own authoritative image/annotation linkage. Until then, S1 training on
optical-derived text labels is not permitted.

## 10. Phase 3H.2 gate

| Gate | Status |
|---|---|
| Authoritative S1 source | PASS |
| 3–5 real S1 samples | PASS (5) |
| VV/VH, raster, checksum validation | PASS |
| Exact dual image linkage | PASS |
| SAR-language applicability | BLOCKED (`SAR_UNKNOWN`) |
| Valid SAR-language supervision | BLOCKED |
| Projector structural contract | PASS |
| Scientific baseline unchanged | PASS — 65.0% accuracy, 4.1034286734 pp MAE, 9.5873312123 pp RMSE |

Phase 3H.2 is **not ready for SAR VQA training**.
