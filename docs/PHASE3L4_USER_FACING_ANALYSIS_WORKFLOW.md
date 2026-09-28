# Phase 3L.4 — User-Facing Analysis Workflow

## 1. Objective
`src.agent.workflow` turns controller results into deterministic execution, audit, evidence, and report summaries.
## 2. Query Understanding and 3. Multi-Step Planning
The existing deterministic planner retains raw query, task, modality, input count, requirements, and its eight-stage plan. It invokes only the selected tool.
## 4. Capability Registry
Scientific/RGB are supported; S2 language remains inconclusive without an auditable raw fixture; SAR, joint, learned change, and grounding remain blocked.
## 5. Evidence Summary, 6. Provenance, 7. Confidence
API summaries preserve evidence type/source/modality/representation/validation/confidence source, controller provenance, and `UNKNOWN` confidence unless supplied.
## 8. Execution Summary and 9. Audit Summary
Responses expose task, status, tools, evidence count, fallback, validation, capability, and provenance completeness.
## 10. Blocked/Inconclusive UX
Blocked answers retain their explicit reason and `Fallback: NONE`; no caption, RGB, or other substitute is generated.
## 11. Multi-Tool Reconciliation and 12. Conflict Handling
The existing deterministic reconciler preserves distinct typed claims and reports conflicts without choosing a winner.
## 13. Report Generation
`deterministic_report` formats query, plan, answer, trace, evidence, provenance, limitations, and audit without an LLM.
## 14. API Integration and 15. Frontend Integration
Controller API results now include execution/evidence/audit summaries and report. The existing renderer presents controller status, evidence, provenance, and trace without direct model access.
## 16. Real Smoke Tests
Phase 3L.2 remains authoritative for real scientific/RGB/temporal calls. S2 generation remains inconclusive.
## 17. Error Handling
Blocked capability is a typed result; malformed inputs are validation errors; tool failures remain distinct.
## 18. Limitations
No training, acquisition, SAR, joint reasoning, learned change, grounding, calibration, or model fallback was added.
## 19. Test Results
Focused workflow tests accompany controller/API regressions. Full-suite result remains inconclusive without pytest terminal summary.
## 20. Phase 3L.5 Gate
Final requirements audit may classify existing functionality only; it must not upgrade unsupported capabilities.

## Sign-off Expansion: Query-First Scene-Reference Contract

SatQuery's normal interaction is **scene/input reference + query**, not arbitrary image upload. The implemented sequence is: scene resolver → query understanding → planner → capability check → validation → representation selection → tool execution → typed evidence → reconciliation → answer → execution/audit summary.

`POST /api/analyze` accepts `controller: true`, `scene`, and `query`. `scene_resolver.py` resolves only exact `pipeline3_5000` scene IDs and existing `google/RSRCC` fixture pair IDs. Invalid/missing fields are `SCENE_REFERENCE_INVALID` or `INPUT_VALIDATION_FAILED`; unknown/unsupported IDs are `SCENE_NOT_FOUND`. No other scene is substituted.

The frontend presents scene/reference information, query, response, status, evidence, provenance, execution summary, and audit summary. Its boundary is frontend → API → controller; it never calls Qwen, the scientific predictor, adapters, checkpoints, or contracts directly.

Current states: scientific and RGB are SUPPORTED; S2 learned language is EXPERIMENTAL and INCONCLUSIVE/NOT_VERIFIED for complete auditable generation; temporal RGB is a supported execution boundary only; SAR, joint optical-SAR, learned change, and grounding are BLOCKED. Blocks remain structured with `Fallback: NONE` and never invoke unrelated tools.

The API summary exposes execution ID, task, status, answer, capability state, evidence, provenance, trace, audit, execution summary, and deterministic report. Evidence records source/modality/representation/validation/confidence source. `MODEL_LANGUAGE_OUTPUT` never becomes scientific, temporal, pixel, or grounding evidence. Confidence is only UNKNOWN, MODEL_PROVIDED, or CALIBRATED; numerical confidence is never inferred. Provenance follows scene reference → resolved source → representation → adapter/tool → model → execution → evidence → answer and filters paths/tokens/secrets.

An explicit multi-tool request may execute multiple available tools; evidence remains separate and conflicting claims are `CONFLICT_DETECTED` without selecting a winner. `deterministic_report` needs no secondary LLM and contains query, plan, answer, trace, evidence, provenance, limitations, execution, and audit summaries.

Phase 3L.2 remains authoritative for real frozen scientific, RGB, and RSRCC temporal smoke execution. Temporal language is not authoritative change evidence. S2 projection exists, but complete controller-level S2-to-language generation remains inconclusive without a current auditable fixture. The scientific baseline remains unchanged: 65.0% accuracy, 4.1034286734 pp MAE, and 9.5873312123 pp RMSE. Full suite status is **FULL_SUITE_INCONCLUSIVE** because no terminal pytest summary was returned.

**Phase 3L.4 sign-off: COMPLETE.** Phase 3L.5 is a requirements classification audit only; it must add no capabilities.
