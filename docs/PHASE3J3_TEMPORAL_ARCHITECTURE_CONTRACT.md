# Phase 3J.3 — Temporal Input and Representation Architecture

Final training status: **TEMPORAL_TRAINING_BLOCKED**

## 1. Objective

Phase 3J.3 implements a reusable, model-neutral boundary for future
bi-temporal inputs. It does not train, evaluate, or select a temporal model;
it does not compute a difference image or fabricate change features.

Implementation: `src/eo_vlm/temporal_contracts.py`.

## 2. Current temporal blockers

The Phase 3J.1 CDVQA gate and Phase 3J.2 ChangeChat/DeltaVLM gate remain
blocked. In particular, no approved temporal source currently supplies the
combination of bounded sample acquisition, complete record-level timestamps,
pixel-registration evidence, verified split/leakage audit, and a usable
per-file image manifest. This architecture does not relax those requirements.

## 3. Temporal input contract

`TemporalInput` contains two independently validated `TemporalFrame` values
and never treats a pair as temporal merely because paths look similar.

```text
TemporalInput
  t1: TemporalFrame(reference_id, modality, provenance, optional tensor_shape)
  t2: TemporalFrame(reference_id, modality, provenance, optional tensor_shape)
  temporal_order: TEMPORAL_ORDER_VERIFIED | TEMPORAL_ORDER_UNKNOWN
  spatial_status: SPATIAL_CORRESPONDENCE_VERIFIED | SPATIAL_CORRESPONDENCE_UNKNOWN
  spatial_correspondence: COREGISTERED | SPATIALLY_CORRESPONDING |
                          SCENE_CORRESPONDING | UNKNOWN
  temporal_metadata: optional ISO-8601 timestamps and interval
  provenance and validation status
```

T1 and T2 must have distinct reference identities. Metadata must be
deterministic JSON-compatible data; timestamps, when supplied, must be a pair
of ISO-8601 values in nondecreasing T1-to-T2 order. An interval is rejected
unless both timestamps exist.

`TemporalSequenceInput` is included as a future-facing, provenance-preserving
container for three or more uniquely identified frames. It performs no
sequence encoding.

## 4. T1/T2 identity

Each frame retains a source reference and independent `TemporalProvenance`
(source ID, optional revision, provenance reference, and deterministic source
attributes). The pair can also retain pair-level provenance. Frame payloads
remain outside the contract, so this metadata layer does not copy imagery or
introduce an untracked cache.

## 5. Temporal ordering

Ordering is an explicit status, never inferred from a filename. A source may
state `TEMPORAL_ORDER_VERIFIED` without timestamps, but unknown ordering must
remain `TEMPORAL_ORDER_UNKNOWN`. ISO-8601 timestamps add validation only; they
do not make missing source order trustworthy.

## 6. Spatial correspondence

The spatial-status flag and correspondence class are validated together:

| Validation status | Permitted correspondence class |
|---|---|
| `SPATIAL_CORRESPONDENCE_VERIFIED` | `COREGISTERED`, `SPATIALLY_CORRESPONDING`, or `SCENE_CORRESPONDING` |
| `SPATIAL_CORRESPONDENCE_UNKNOWN` | `UNKNOWN` only |

The contract deliberately does not upgrade a same-scene or same-tile claim to
pixel co-registration.

## 7. S2 temporal path

Frames with modality `s2` require the existing canonical shape
`[12,120,120]`. A later consumer can pass T1 and T2 independently through the
unchanged `S2MultispectralProjector`, producing two separate visual token
representations. Phase 3J.3 does not call, retrain, or modify that projector,
and it supplies no temporal fusion after those two calls.

## 8. SAR temporal path

Frames with modality `s1` require `[2,120,120]`, preserving the existing
VV/VH SAR-projector boundary. The contract can represent S1 T1 plus S1 T2,
but it makes no claim of SAR temporal language supervision, temporal SAR
reasoning, or learned SAR fusion.

Other `optical` and `sar` frames are supported as future interfaces with a
positive shape, including future optical/SAR pairs. They are not declared
compatible with raw S2/S1 preprocessors.

## 9. Temporal representation

`TemporalRepresentation` retains the `TemporalInput`, optional existing typed
`RepresentationRef` values for T1/T2, and a resource estimate. Its
`temporal_features` and `change_features` are both required to be exactly
`NOT_COMPUTED`. Any proposed value, including zeros, tensors, raw subtraction,
or a text label, is rejected.

Consequently this module cannot be mistaken for a learned change encoder or a
pixel-difference baseline. No `PIXEL_DIFFERENCE_BASELINE` is implemented.

## 10. Temporal fusion boundary

`TemporalFusionAdapter` defines a model-swappable interface:

```text
get_capabilities()
get_provenance()
estimate_resources(temporal_input)
encode_pair(temporal_input)
encode_sequence(sequence)
```

The supplied `ContractOnlyTemporalFusionAdapter` reports its non-model
capabilities and raises `TemporalFusionUnavailable` for both encoding calls.
It contains no Qwen, ChangeChat, DeltaVLM, EarthDial, or EarthGPT dependency.

## 11. Evidence model

`TemporalEvidenceItem` distinguishes five non-interchangeable categories:

- `OBSERVED_T1`
- `OBSERVED_T2`
- `TEMPORAL_MODEL_EVIDENCE`
- `MODEL_LANGUAGE_OUTPUT`
- `SOURCE_METADATA`

Each requires provenance. The contract supplies no change evidence and never
converts a model-language output or metadata item into temporal model evidence.

## 12. Grounding boundary

No temporal region, correspondence geometry, mask, box, point, or polygon is
derived. Phase 2F remains fail-closed; future T1/T2/changed-region grounding
requires separately verified coordinate semantics and an evaluated model.

## 13. Resource estimation

`TemporalResourceEstimate` always reports two images for a pair and uses
`UNKNOWN` for VRAM, sequence cost, and generation cost. Non-measured numeric
values are rejected. This gives a later planner an explicit safe signal rather
than a made-up memory estimate.

## 14. Model swappability

The boundary is intentionally:

```text
TemporalInput -> Temporal representation adapter -> Temporal fusion adapter -> language-facing VLM
```

Only the first contract and fusion interface exist now. The language-facing
model remains outside the interface, so future models can be substituted only
through a validated adapter implementation.

## 15. Structural tests

`tests/test_temporal_contracts.py` uses metadata and shape-only synthetic
fixtures, never synthetic labels. It covers S2 and S1 shape contracts,
optical/SAR interface representation, distinct T1/T2 identity, metadata/order
validation, spatial correspondence validation, typed representation references,
`NOT_COMPUTED` enforcement, deterministic fingerprints, sequence provenance,
evidence categories, unknown resource estimates, and fail-closed swappable
fusion behavior.

## 16. Implemented vs deferred

Implemented:

- Temporal input/sequence contracts and validation.
- Explicit ordering and spatial-correspondence semantics.
- Provenance, typed representation references, fingerprints, and resource
  estimates.
- A model-neutral fusion interface and evidence categories.

Deferred:

- Learned temporal encoder or learned fusion.
- Change VQA, description, grounding, CDVQA evaluation, and Qwen integration.
- Pixel-difference baseline and every type of change-feature computation.

Blocked:

- Real temporal training, pending the full authoritative dataset-readiness
  gate.

## 17. Training readiness

The architecture boundary is structurally ready, but it is not a training
implementation. **TEMPORAL_TRAINING_BLOCKED** remains the only valid data and
training status. It does not modify the frozen scientific predictor, CROMA,
existing S2/SAR projectors, Phase 2F semantics, splits, receipts, or
checkpoints.

The scientific baseline remains: accuracy `65.0%`, MAE `4.1034286734 pp`, and
RMSE `9.5873312123 pp`.

## 18. Phase 3J.4 gate

Phase 3J.4 is a temporal dataset re-evaluation/acquisition gate. Training can
be considered only after a source verifies imagery, explicit supervision,
ordered pair and annotation linkage, timestamps, correspondence, licensing,
pinning, official split/leakage controls, provenance, bounded fixture
acquisition, and structural validation.
