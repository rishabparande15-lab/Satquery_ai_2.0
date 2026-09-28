# Phase 3L.2 — Real Controller Integration

## 1. Objective

This phase connects the Phase 3L.1 controller to validated components without modifying models, datasets, representations, checkpoints, or training.

## 2. Real Components

`FrozenScientificCacheTool` executes the frozen 830-D probe over the existing checksum-validated feature cache. `S2ProjectorTool` loads the Phase 3G.1 learned projector. `QwenRGBTool` and `TemporalRGBTool` lazily use the local pinned Qwen snapshot.

## 3. Mocked Components

Blocked tasks have no fake implementation. Lightweight fakes remain limited to controller-isolation and controlled-conflict unit tests.

## 4. Flow A — Scientific

One exact non-test area from the verified cache executed through the frozen probe. It emitted `SCIENTIFIC_PREDICTION`, retained 62D + 768D → `hybrid_830d` provenance, and made no Qwen call. This is cache-backed source-validated inference, not new scientific evaluation.

## 5. Flow B — S2 Language

The real learned S2 projector checkpoint produced `[1,16,2048]` tokens. Current local raw validated S2 imagery is unavailable, so fresh end-to-end Qwen language generation from a real S2 sample is **INCONCLUSIVE**. The controller correctly records this as `METADATA_EVIDENCE`, not generated language.

## 6. Flow C — RGB

The opt-in CUDA gate uses one local RSRCC RGB fixture and the pinned Qwen2.5-VL-3B-Instruct snapshot. It emits `MODEL_LANGUAGE_OUTPUT` and retains model/revision/device/dtype/generation/latency provenance.

## 7. Flow D — Temporal RGB

The same sequential Qwen runtime executes the RSRCC pair through `TemporalInput` and `TemporalRGBAdapter`. Provenance preserves T1/T2, verified ordering, spatial correspondence, unknown timestamps, unknown registration, and `learned_temporal_fusion=False`. Output is `MODEL_LANGUAGE_OUTPUT`, never `TEMPORAL_EVIDENCE`.

## 8. Flow E — SAR Block

SAR VQA returns `SAR_VQA_BLOCKED` and `Fallback: NONE`; no optical or Qwen substitute runs.

## 9. Flow F — Optical-SAR Block

Joint requests return `OPTICAL_SAR_REASONING_BLOCKED`; independent answers are never merged as joint reasoning.

## 10. Flow G — Temporal Change Block

“What changed?” routes to blocked `CHANGE_VQA`. The separately named `TEMPORAL_RGB_EXECUTION` route requires explicit `execution_mode=temporal_rgb` and cannot satisfy learned change reasoning.

## 11. Flow H — Grounding Block

Grounding returns `GROUNDING_BLOCKED`, with no point, box, mask, polygon, region, token mapping, or source-geometry conversion.

## 12. Evidence Reconciliation

Scientific predictions, language output, and metadata evidence remain distinct and are never promoted across types.

## 13. Conflict Handling

Different same-key claims are preserved as `CONFLICT_DETECTED`; the reconciler chooses no winner.

## 14. Confidence

Confidence remains `UNKNOWN` unless a real model, calibration, or scientific-metric source supplies it.

## 15. Provenance

Results retain execution/input/tool/representation references. Real tools add applicable model/revision, adapter/receipt, device/dtype/quantization, generation configuration, and latency.

## 16. Execution Traces

All paths preserve interpretation, validation, representation and tool selection, execution, evidence collection, reconciliation, and answer stages. Blocks are explicit.

## 17. Resource Usage

Qwen is lazy and sequential; per-call latency is retained. The existing temporal receipt measured 7,802,813,952 B peak allocated and 8,019,509,248 B reserved; these are historical receipt values, not new estimates.

## 18. Error Handling

Missing IDs/modalities and invalid payloads fail deterministically. Missing cache/checkpoints, malformed S2/pairs, and runtime exceptions remain explicit failures; unsupported branches block before tool invocation.

## 19. Real vs Mocked Boundary

Real execution covers the frozen cache probe, learned projector, and opt-in Qwen RGB/pair gate. Fresh end-to-end S2 language is inconclusive due to raw-source absence. No mock is presented as a model call.

## 20. Limitations

This validates orchestration, not benchmark performance, SAR, optical-SAR, learned temporal/change, grounding, calibrated confidence, or universal multimodal reasoning.

## 21. Phase 3L.3 Gate

API/web exposure should wait for an auditable local raw S2 fixture or acceptance of the explicit S2 limitation. The scientific baseline remains 65.0% accuracy, 4.1034286734 pp MAE, and 9.5873312123 pp RMSE.
