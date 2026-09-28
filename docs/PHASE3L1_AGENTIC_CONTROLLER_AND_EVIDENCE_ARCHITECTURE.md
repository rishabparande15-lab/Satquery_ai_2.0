# Phase 3L.1 — Agentic Controller and Evidence Architecture

## 1. Architecture

`src.agent` implements the deterministic boundary: query understanding → planning → capability check → existing-contract validation boundary → representation selection → injected tool execution → typed evidence → reconciliation → answer assembly → audit trace. It does not contain model logic or initiate training/data acquisition.

## 2. Query Types

The controller defines scientific analysis, RGB reasoning, S2 VQA/captioning, SAR, optical-SAR, temporal/change, grounding, and report task types. Rules are deterministic and retain a short rationale.

## 3. Capability Registry

Scientific and RGB are `IMPLEMENTED`; S2 language is `EXPERIMENTAL` / validated PoC; SAR, optical-SAR, learned temporal/change, and grounding are `BLOCKED`. Registration does not execute a model.

## 4. Planner

`DeterministicPlanner` maps declared modality, image count, and explainable query keywords to one task. It never recasts a blocked request as a different task.

## 5. Controller

`AgentController.analyze(AnalysisRequest)` is independent of the web UI and accepts concrete tools by dependency injection. No checkpoint is loaded by default.

## 6. Tool Layer

`Tool` / `ToolResult` supports `scientific_predictor`, `qwen_rgb`, and `qwen_s2`. Future SAR, fusion, temporal, and grounding tools remain capability entries only and cannot emit fake output.

## 7. Execution Trace

Each result includes ID, UTC timestamp, interpretation, validation, representation/tool choices, execution durations, evidence IDs, reconciliation, and answer status. Blocked and no-tool outcomes include `Fallback: NONE`.

## 8. Evidence Model

Evidence records type, claim, modality, representation, spatial/temporal references, confidence source, provenance, and validation status. Language output cannot be promoted into a scientific prediction.

## 9. Evidence Reconciliation

The reconciler is deterministic: it groups claim keys, preserves every source, and compares claims. It does not use an LLM or select a winner.

## 10. Conflict Handling

Different values for the same claim key yield `CONFLICT` / `CONFLICT_DETECTED`, retaining both evidence IDs and claims without invented resolution.

## 11. Confidence

Confidence is optional. A numeric value requires `MODEL_PROVIDED`, `CALIBRATED`, or `SCIENTIFIC_METRIC`; otherwise it is `UNKNOWN`.

## 12. Provenance

Result provenance retains execution/query/task/input IDs/tool/representation references. Tool provenance carries model, revision, adapter, preprocessing, device, dtype, quantization, and dataset fingerprints when the real tool supplies them.

## 13. Answer Assembly

Answers declare `scientific_prediction`, `model_language`, `both`, `neither`, or `conflict`; a VLM is not represented as independent scientific validation.

## 14. Blocked Capability Behavior

SAR VQA, joint optical-SAR, temporal/change, and grounding return explicit blocked statuses and call no fallback tool. The controller does not create pixels, masks, tokens, regions, or change claims.

## 15. Scientific Separation

The frozen baseline remains 65.0% accuracy, 4.1034286734 pp MAE, and 9.5873312123 pp RMSE. `hybrid_830d` is selected only for the scientific predictor and is never passed to Qwen.

## 16. Model Swappability

Model-specific code stays behind `Tool.execute`; Qwen/S2 adapters, scientific predictor, CROMA, checkpoints, and representations are unchanged.

## 17. API Boundary

The layer is callable as Python API today. A future API route may instantiate configured validated tools without changing controller behavior.

## 18. Test Strategy

Focused tests use fakes and cover parsing, registry boundaries, routing, blocked/no-fallback handling, S2/RGB/scientific paths, typed evidence, confidence, conflicts, traces, deterministic planning, malformed requests, and tool isolation. They do not require checkpoints.

## 19. Implemented Capabilities

Structural orchestration is implemented for injected scientific, RGB Qwen, and S2 Qwen tools. Real model execution remains owned by their separately validated adapters.

## 20. Deferred Capabilities

No SAR language, optical-SAR fusion, learned temporal/change reasoning, learned grounding, LLM planning, automatic conflict resolution, or confidence calibration is introduced here.

## 21. Phase 3L.2 Gate

Proceed only after a concrete capability has its validated data, model, evaluation, provenance, and fail-closed tool adapter. This phase changes none of the frozen scientific inputs, checkpoints, splits, receipts, temporal contracts/execution receipt, grounding contracts, or Phase 2F spatial protection.
