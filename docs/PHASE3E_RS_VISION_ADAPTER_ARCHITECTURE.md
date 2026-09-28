# PHASE 3E — Remote-Sensing Vision Adapter Architecture

**Status:** architecture contracts implemented; non-RGB modality adaptation is
designed/deferred, not claimed as implemented.

## 1. Phase Objective

Phase 3E establishes a model-swappable language-facing adapter between
SatQuery's validated inputs and Qwen2.5-VL. It preserves the existing RGB path
and makes future multispectral, SAR, optical-SAR, temporal, grounding, CROMA,
and evidence paths explicit without training a projector or changing the
scientific predictor.

## 2. Current Qwen Capability Boundary

The validated local Qwen path is:

```text
RGB image → Qwen processor → Qwen2.5-VL-3B FP16 CUDA → model text
```

Phase 3D.3 measured successful FP16 loading and one RGB generation. This does
not establish native multispectral, SAR, optical-SAR, temporal, or grounding
support. `bitsandbytes` remains unavailable; 8-bit/4-bit are not claimed.

## 3. RGB Path — IMPLEMENTED

`Qwen25VLRGBAdapter` continues to validate H×W×3 RGB-compatible arrays,
prepare a deterministic uint8 RGB image, load Qwen lazily, invoke the Qwen
processor, generate text, and preserve provenance. Existing RGB behavior and
Phase 3D smoke-test results remain unchanged.

## 4. Explicit Input Contracts

| Input | Contract status | Qwen path | Current meaning |
| --- | --- | --- | --- |
| RGB optical image | `SUPPORTED` | `DIRECT_QWEN_INPUT` | Implemented and locally smoke-tested |
| S2 `[N,12,120,120]` | `ADAPTER_REQUIRED` | `PROJECTOR_REQUIRED` | Bands/order preserved; no conversion invented |
| S1 `[N,2,120,120]` `VV,VH` | `ADAPTER_REQUIRED` | `PROJECTOR_REQUIRED` | SAR semantics preserved; no native path claimed |
| Optical + SAR | `ADAPTER_REQUIRED` | `PROJECTOR_REQUIRED` | Joint fusion contract only; fusion not implemented |
| Temporal pair | `NOT_IMPLEMENTED` | `PROJECTOR_REQUIRED` | T1/T2 contract described; no fake two-image reasoning |
| CROMA tokens `[N,225,768]` | `ADAPTER_REQUIRED` | `PROJECTOR_REQUIRED` | Requires a future learned/validated projector |
| CROMA GAP `[N,768]` | `ADAPTER_REQUIRED` | `PROJECTOR_REQUIRED` | Not directly passed to Qwen |
| `physical_62d` | `UNSUPPORTED` | `UNSUPPORTED` | Scientific physical representation |
| `hybrid_830d` | `UNSUPPORTED` | `SCIENTIFIC_PREDICTOR_ONLY` | Never injected into Qwen by default |

These statuses are exposed by `get_input_contracts()` and mirrored in
`get_capabilities()`.

## 5. Multispectral Adapter Design — DESIGNED / DEFERRED

The S2 contract preserves the exact band order:

```text
B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
```

The following are design alternatives, not implemented claims:

1. selected-band pseudo-RGB;
2. multi-channel projection;
3. grouped-band representation;
4. learned multispectral projector;
5. model-native multispectral encoder.

Phase 3E does not choose among them because Phase 3C/3D established only an
RGB Qwen path. Any future implementation must record band selection,
normalization, resolution, information loss, and learned/non-learned status.

## 6. SAR Adapter Design — DESIGNED / DEFERRED

The S1 contract is `[N,2,120,120]` in `VV,VH` order. Future options include a
two-channel projection, a clearly labelled pseudo-RGB representation, a
learned SAR projector, or a dedicated SAR encoder. None is implemented.

Pseudo-RGB conversion must never be described as native SAR understanding.

## 7. Optical-SAR Architecture — DESIGNED / FUSION NOT IMPLEMENTED

The future joint path is:

```text
OPTICAL representation ─┐
                         ├→ joint representation → Qwen input adapter → Qwen
SAR representation ────┘
```

Supported future implementation families are shared projected visual tokens,
cross-modal fusion, multi-image multimodal packing, or dedicated optical/SAR
encoders. Two independent answers are not joint reasoning, and no such fusion
is currently implemented.

## 8. Temporal Architecture — DESIGNED / NOT IMPLEMENTED

Future `TemporalInput` must preserve:

```text
T1
T2
optional timestamps/order
optional spatial correspondence
optional sensor/CRS/transform metadata
```

The output contract must retain T1 evidence, T2 evidence, temporal comparison
evidence, and language output. Prompting Qwen with two images is not equivalent
to learned temporal reasoning. No temporal model or change labels were added.

## 9. CROMA Boundary

`CROMA_TOKENS`, `CROMA_GAP`, `physical_62d`, and `hybrid_830d` are represented
as explicit contracts. The adapter reports:

- CROMA tokens/GAP: `PROJECTOR_REQUIRED`;
- physical features: `UNSUPPORTED` for Qwen input;
- hybrid: `SCIENTIFIC_PREDICTOR_ONLY`.

No CROMA feature was passed to Qwen, and no generic VLM embedding was inferred
from the scientific representation.

## 10. Grounding Boundary

Future grounding output types are `point`, `bbox`, `polygon`, `mask`, and
`region_reference`. The contract preserves source geometry, model grounding,
and evidence reference as separate fields. Current status is `NOT_VERIFIED`:

- Phase 2F source geometry remains preserved but unmapped;
- no BigEarthNet pixel/token mappings were created;
- no synthetic labels or invented coordinates were created;
- learned grounding remains unimplemented.

## 11. Resource-Aware Execution

`estimate_resources()` exposes:

- `estimated_vram_bytes`;
- `estimated_context_cost`;
- `estimated_image_count`;
- `estimated_generation_cost`;
- dtype and quantization mode;
- unmeasured peak fields.

The current estimate uses the measured FP16 post-load allocation of
7,520,955,904 bytes as its checkpoint-specific basis. It is not an exact
generation-memory prediction. Multi-image, optical-SAR, and T1/T2 requests
must require a future resource check because the current 8,151 MiB GPU leaves
limited headroom.

## 12. Provenance

Adapter results preserve model ID, checkpoint revision, adapter ID, input
modality, input hash, preprocessing identity, temporal metadata, device, dtype,
quantization, and generation configuration. The explicit
`scientific_representation_used` flag prevents model text from silently
becoming scientific evidence.

## 13. Model-Swappable Architecture

The abstract `RemoteSensingVisionAdapter` contract isolates model-specific
loading, preprocessing, generation, capabilities, provenance, and resource
estimation. Qwen is the current implementation. EarthDial, EarthGPT,
Earth-OneVision, and future EO-VLMs can provide separate implementations
without changing the planner, evidence reconciler, API, frontend, or
provenance schema.

```text
USER → QUERY UNDERSTANDING → TASK PLANNER → INPUT/GEO VALIDATOR
     → REPRESENTATION SELECTOR → RS VISION ADAPTER → MODEL
     → MODEL OUTPUT → EVIDENCE RECONCILER → ANSWER GENERATOR
```

## 14. Implemented vs Planned Capabilities

| Capability | Status |
| --- | --- |
| RGB preparation and Qwen generation | `IMPLEMENTED / PASS` |
| S2 input contract | `IMPLEMENTED CONTRACT / ADAPTER REQUIRED` |
| SAR input contract | `IMPLEMENTED CONTRACT / ADAPTER REQUIRED` |
| Optical-SAR interface | `DESIGNED / FUSION NOT IMPLEMENTED` |
| Temporal interface | `DESIGNED / NOT_IMPLEMENTED` |
| Grounding output contract | `DESIGNED / NOT VERIFIED` |
| CROMA metadata boundary | `IMPLEMENTED CONTRACT / PROJECTOR REQUIRED` |
| Hybrid scientific separation | `IMPLEMENTED` |
| Resource estimate surface | `IMPLEMENTED / ESTIMATE ONLY` |
| EarthDial/EarthGPT/Earth-OneVision adapters | `DEFERRED` |
| VLM projector training | `DEFERRED` |

## 15. Tests

Focused adapter and relevant tests pass: **34 passed**. Coverage includes RGB,
S2/SAR declarations, optical-SAR and temporal contracts, CROMA and hybrid
separation, grounding fail-closed behavior, unsupported-input rejection,
provenance, resource estimates, model-swappable inheritance, and the existing
RGB regression path. Model weights are not required by ordinary unit tests.

Compile checks passed and `git diff --check` passed.

## 16. Phase 3F Readiness

Phase 3E is complete for architecture and RGB integration. Phase 3F should
begin with one real S2 multispectral adaptation path, selected only after
choosing and validating an information-preserving adapter approach. SAR,
optical-SAR fusion, temporal reasoning, grounding, and projector training
remain separate gates.

The scientific baseline remains unchanged: 65.0% accuracy, 4.1034286734 pp
MAE, and 9.5873312123 pp RMSE. No scientific artifacts, CROMA files,
representations, receipts, datasets, splits, checkpoints, or Phase 2F files
were modified.
