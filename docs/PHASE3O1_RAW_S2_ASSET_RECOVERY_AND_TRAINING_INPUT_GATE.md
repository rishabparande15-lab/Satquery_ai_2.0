# Phase 3O.1 — Raw S2 Asset Recovery and Training-Input Gate

Status: **RAW S2 VERIFIED; TRAINING INPUT READY FOR A SUBSEQUENT CONTROLLED PHASE**

## Objective and recovery result

This phase recovered no data from the network and performed no training. It
reused five real, already project-managed S2 areas from the external
Pipeline-3 S2 roots. The first Phase 2E canonical view had no direct path
overlap with those roots, so it was not silently used. Instead, linkage comes
from the existing validated `visual_vqa_candidates.jsonl` BigEarthNet.txt pool
whose records contain exact image ID, split, annotation ID, source-record
checksum, and revision.

The source authority is the project-pinned BigEarthNet v2 acquisition route
documented by Phase 3F.1: `hackelle/BigEarthNetV2-LMDB` revision
`118d1b6285c080ba8e4078414e1b8a243b18c9bd`, connected through the validated
BigEarthNet.txt annotation revision
`72d865f2146f0a85b720f7f3ca1cdbaeafc3d316`. Use is recorded as
`PROJECT_AUTHORIZED_BIGEARTHNET_TXT`; no new license assertion or download was
introduced.

## Inspection and acceptance

The bounded lookup inspected five candidate train-only, `VISUAL_ONLY` binary
or multiple-choice records, excluding relative-position and all geometric,
temporal, SAR, optical-SAR, test, and validation records. All five were
accepted; none were rejected. Zero files and zero bytes were acquired because
the files already existed outside Git in the project-managed cache.

For each accepted area the gate verified twelve readable single-band GeoTIFFs,
the canonical order `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12`, native
20/60/120-m grid dimensions, finite numeric pixels, no nodata pixels, a
combined SHA-256, CRS, affine transform, bounds, exact train split, annotation
linkage, source revision, and source-record checksum. It rejects duplicate
image IDs, duplicate combined hashes, and ambiguous or cross-split input.

The canonical cube is constructed only through the established bilinear
alignment to B02 followed by `robust_channel_scale_v1`; no RGB conversion or
channel dropping occurs.

## Projector-only smoke test

Five verified cubes were preprocessed and passed to the existing frozen-Qwen
S2 projector, without loading Qwen or modifying any weight.

| Measurement | Result |
|---|---:|
| CPU preprocessing | 0.7706017000 s |
| Projector forward | 15.8262319000 s |
| Peak CUDA allocation | 40,620,544 bytes |
| Peak process RAM | 1,110,974,464 bytes |
| Output | `[5,16,2048]`, finite |

The smoke test is `PASS`. It is a representation compatibility receipt, not a
quality measure or a model-update result.

## Gate and artifacts

`training_input_ready` is true: all raw, source, split, linkage,
normalization, duplicate, contamination, and projector conditions passed. The
Phase 3O.1 receipt still sets `execution_permitted: false`; training belongs
only to the subsequent controlled phase. No benchmark inference ran.

- [Verified asset inventory](../artifacts/training/phase3o/verified_s2_training_assets.json)
- [Deterministic training-input manifest](../artifacts/training/phase3o/phase3o_s2_training_manifest.json)
- [Machine-readable gate receipt](../artifacts/training/phase3o/phase3o1_s2_asset_gate_receipt.json)

Manifest SHA-256: `972c0f8afbeea3d9fbfc641557dc4296558a1fdf374123d7388ebb25831051aa`.

## Validation and immutability

The new contract tests cover valid assets; shape, band, NaN, Inf, and nodata
rejections; source/provenance/checksum/split/linkage rejections; forbidden
grounding/temporal/SAR records; collision and duplicate handling; deterministic
manifests; smoke schema; and both ready and blocked synthetic gates. Real
assets, not synthetic tensors, supplied the recorded smoke test.

No Pipeline-3 baseline, scientific split/metric/checkpoint, CROMA asset,
hybrid predictor, existing S2 projector checkpoint, SAR/temporal/grounding
freeze, or benchmark-readiness state changed. The scientific baseline remains
65.0% accuracy, 4.1034286734 pp MAE, and 9.5873312123 pp RMSE.

## Condition for Phase 3O.2

Use this frozen five-area manifest, re-check every current raw-file checksum,
and run an adapter-only controlled pilot with Qwen frozen. Acceptance must
record finite loss, a real projector parameter change, validation isolation,
adapter-checkpoint hash, and checkpoint reload. 
