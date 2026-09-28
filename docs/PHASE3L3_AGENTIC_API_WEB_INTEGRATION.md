# Phase 3L.3 — Agentic API and Web Integration

## 1. Objective

The existing `POST /api/analyze` now accepts `controller: true` requests and routes them through the fail-closed controller facade.

## 2. Existing API

The legacy Pipeline 3 route remains unchanged. Controller requests use the same loopback endpoint and do not expose model paths or secrets.

## 3. Controller Integration

`src.agent.api_integration.analyze_controller_request` translates query/input metadata to `AnalysisRequest`, serializes results safely, and retains controller-only execution.

## 4. Request Contract

`{controller:true, query, inputs:[{id, modality, source_reference}], temporal}`. Raw model implementation fields are not accepted.

## 5. Response Contract

Responses contain execution ID, status, task type, answer, capability status, evidence, provenance, execution trace, audit summary, and blocked reason where applicable.

## 6. Capability Exposure

RGB and scientific are capability entries; S2 generation remains `NOT_VERIFIED` / inconclusive without an auditable raw fixture. SAR, joint, change, and grounding are blocked.

## 7. Real Supported Flows

The controller tools remain the only real-execution route. The web layer never imports or calls Qwen, the probe, or adapters.

## 8. Inconclusive S2 Flow

The API reports no configured generation tool rather than inventing an S2 answer.

## 9. Blocked Flows

SAR, optical-SAR, learned change, and grounding return `BLOCKED` with `Fallback: NONE`.

## 10. Evidence

Evidence serialization preserves type; language is not scientific, temporal, pixel, or grounding evidence.

## 11. Provenance

Execution and input IDs, task/tool/representation references are returned after secret/path filtering.

## 12. Execution Trace

All eight controller stages are returned for UI rendering.

## 13. Error Model

Malformed controller payloads are input errors; blocked requests are responses, not 500 errors. Internal tool detail is not returned.

## 14. Frontend Integration

The existing renderer recognizes controller responses and displays final status, typed evidence, provenance, trace, and explicit blocked/no-fallback messaging without a visual redesign.

## 15. Real API Smoke Tests

Structural API tests cover blocked paths and S2 inconclusive routing. Real-model smoke remains owned by Phase 3L.2’s opt-in execution gate to avoid checkpoint loads in ordinary API tests.

## 16. Security

The facade filters keys containing `path` or `token`; stack traces and local model/cache locations are not serialized.

## 17. Performance

No Qwen concurrency was introduced.

## 18. Test Results

Focused API/controller tests pass; full-suite status remains inconclusive unless an actual terminal pytest summary is produced.

## 19. Limitations

This does not add training, S2 generation, SAR, fusion, learned change, grounding, calibrated confidence, or a frontend model bypass.

## 20. Phase 3L.4 Gate

Proceed only with capability-aware user workflows that preserve these same evidence and fail-closed boundaries.
