# Phase 3J.5 — Real Temporal RGB Adapter and Execution Path

**Status: IMPLEMENTED / TESTED.** Temporal training remains
**TEMPORAL_TRAINING_BLOCKED**.

## 1. Objective

This phase proves only an engineering boundary: a real RSRCC before/after RGB
pair can be preserved as `TemporalInput`, packaged as two independent RGB
visual inputs, accepted by the existing Qwen processor, and yield short
language output. It does not train, fuse, register, score, or ground a
temporal model.

## 2. RSRCC fixture identity

The bounded fixture is `google/RSRCC`, revision
`7898de7bfd08bc404d9a92e1caaa9dce91b0c3ea`, Apache-2.0, split `val`.
It contains these three acquired 512×512 RGB pairs:

| Pair key | T1 | T2 | Annotation reference |
|---|---|---|---|
| `d259f34c_8693_418e_9dab_ef14b860319f` | `*_before.png` | `*_after.png` | `val/metadata.csv row 1` |
| `1432362f_5538_4803_b148_be89ea186c3f` | `*_before.png` | `*_after.png` | `val/metadata.csv row 2` |
| `e88af405_10d9_4de5_8f1f_064f20861517` | `*_before.png` | `*_after.png` | `val/metadata.csv row 3` |

There is no source-issued annotation ID; the fixture keeps `annotation_id` as
`null`, plus a row locator and annotation hash. Image and metadata hashes are
in `artifacts/test_fixtures/temporal_change_fixture_manifest.json`.

## 3. TemporalInput validation

**TESTED.** Each fixture pair has distinct T1/T2 references, modality
`optical`, documented `before → after` ordering, and
`SPATIALLY_CORRESPONDING` correspondence. The pair order is dataset semantic
order, not a dated chronology. `TemporalMetadata()` is intentionally empty:
timestamps and temporal interval are **UNKNOWN**. Registration remains
**UNKNOWN at the reproducible-registration level**; no warp, alignment, or
source-image modification was performed.

## 4. RGB temporal adapter

**IMPLEMENTED:** `src/eo_vlm/temporal_rgb_adapter.py` provides the
model-swappable `TemporalRGBAdapter` and `PreparedTemporalRGBInput`.

It validates two optical H×W×3 RGB arrays, preserves reference IDs and the
`TemporalInput` fingerprint, uses the existing `Qwen25VLRGBAdapter` RGB
preparation once per image, and emits a two-image Qwen request. It does not
duplicate RGB pixel processing, construct change features, perform learned
fusion, or route RGB data through the S2 or S1 adapters.

## 5. Qwen integration

**TESTED on one real pair.** The existing local
`Qwen/Qwen2.5-VL-3B-Instruct` checkpoint revision
`66285546d2b821cf421d4f5eb2576359d3770cd3` ran on `cuda` in `float16`, with
batch size 1 and `max_new_tokens=8`. The processor received two RGB images in
documented before/after order. The fixed prompt used “before image” and
“after image,” never unsupported dates or chronological timestamps.

## 6. Pair execution and generated output

**PASS:** `d259f34c_8693_418e_9dab_ef14b860319f` completed the execution
path. The raw model output was:

> `the red car is no longer in the`

This is unvalidated model-generated language, not a dataset answer, objective
change evidence, annotation, geometry, or benchmark result. No accuracy,
BLEU, ROUGE, CIDEr, or other metric was calculated. The other two acquired
pairs were structurally validated but not sent to Qwen; this remained within
the three-pair and non-concurrent safety limit.

## 7. Resource measurements

| Measurement | Result |
|---|---:|
| Preprocessing latency | 0.01046 s |
| Generation latency | 40.73444 s |
| Total latency | 40.74490 s |
| Generated tokens | 8 |
| CUDA allocated before inference | 0 bytes |
| CUDA reserved before inference | 0 bytes |
| Peak CUDA allocated | 7,802,813,952 bytes |
| Peak CUDA reserved | 8,019,509,248 bytes |
| Process RSS before inference | 509,673,472 bytes |
| Process RSS after inference | 2,990,092,288 bytes |

The 8 GiB GPU had narrow headroom. The single model instance and temporary
objects were released and CUDA cache was cleared after the sample. No OOM
occurred; no repeated OOM retry was attempted.

## 8. Provenance and evidence boundary

The execution receipt is
`artifacts/test_fixtures/rsrcc_execution_phase3j5.json`. Its pair input SHA-256
is `855b7416ff25ae7824569c49cbb16f2da69f1076c16550a3e7a82e4170ad0c65`, and
its `TemporalInput` fingerprint is
`69f386641ad896318255c84f0bccdd4696b700f5ca5cc56fdf5f9b4a9f26727f`.

T1 and T2 observations remain source images; the CSV answer remains an
annotation/reference answer; the quoted output is a model-language output;
dataset identity is provenance. None is promoted to a change mask, box,
point, polygon, or scientific evidence.

## 9. Limitations and separation

**UNKNOWN / BLOCKED:** RSRCC supplies no acquisition dates, intervals, sensor
or source-scene provenance, source-issued annotation ID, registration
parameters/residuals/CRS, or geographic/near-duplicate audit fields.

RSRCC temporal RGB is separate from native SatQuery S2 12-band temporal data
and separate from S1 VV/VH temporal data. The S2 adapter/checkpoint, SAR
projector/provenance, CROMA, scientific predictor, scientific split/receipts,
and Phase 2F fail-closed grounding semantics were not changed.

## 10. Implemented versus deferred

- **IMPLEMENTED:** RGB pair validation, deterministic two-image preparation,
  provenance, resource estimate, and Qwen-compatible generation delegation.
- **TESTED:** focused adapter tests and one real RSRCC Qwen generation.
- **DEFERRED:** learned temporal fusion, temporal adaptation/training,
  registration, temporal benchmark evaluation, scientific change detection,
  and grounding.
- **BLOCKED:** temporal training, due to the Phase 3J.4 data-readiness gaps.

## 11. Validation and Phase 3J.6 gate

Focused temporal RGB, temporal-contract, RGB, S2, and SAR tests passed:
**38 passed**. `compileall` and `git diff --check` passed. The frozen
scientific baseline remains **65.0% accuracy**, **4.1034286734 pp MAE**, and
**9.5873312123 pp RMSE**.

Phase 3J.6 is **TEMPORAL MODEL / SUPERVISION DECISION GATE**. It must decide
whether a fully valid temporal dataset has become available; otherwise
temporal training stays deferred while SatQuery advances.
