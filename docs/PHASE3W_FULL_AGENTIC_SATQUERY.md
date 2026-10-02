# Phase 3W — Full Agentic SatQuery Integration

Status: `PHASE3W_INTEGRATION_PARTIAL`

Phase 3W adds `SatQueryAgent`, a deterministic, fail-closed orchestration layer over the existing specialists. It does not train or alter any model, checkpoint, projector, vocabulary, or dataset policy.

## Agent behavior

The agent normalizes declared inputs, classifies their configuration before interpreting the query, classifies a bounded query intent, selects one capability, validates compatibility, delegates to the existing specialist, and returns a common response contract. Responses contain route-specific provenance, limitations, confidence marked `NOT_AVAILABLE`, and an operational execution trace. The trace records components and actions only; it does not expose private reasoning.

The unified API is `POST /api/v1/query`. Existing specialist endpoints remain intact for compatibility and focused testing. The existing Single Image, Optical + SAR, and Bi-temporal UI panels now call the unified endpoint.

Each unified result panel exposes a client-side JSON download containing the query, input summary, route, answer, warnings, provenance, and operational execution trace. It contains no private reasoning.

## Capability policy

- `SINGLE_IMAGE_VQA`: available with limitations; no benchmark-generalization claim.
- `OPTICAL_SAR_ANALYSIS`: available with limitations; no demonstrated gain over the optical-only baseline.
- `TEMPORAL_CHANGE_DESCRIPTION`: available; language description only, with no masks, boxes, or area estimates.
- `SINGLE_IMAGE_GROUNDING`: blocked; never falls back to VQA.
- `SINGLE_IMAGE_CAPTION`: experimental and unavailable for reliable use.

## Verification

The real Chrome → unified frontend → `/api/v1/query` → `SatQueryAgent` → Chg2Cap path passed with validation-safe `val_000001.png` (A/PRE and B/POST), producing a genuine nonempty description with zero console and page errors.

The current local application did not expose approved non-test S2 or S1+S2 development samples. Therefore the real-browser single-image, optical-SAR, and grounding UI validations were not run; no substitute data was downloaded or used. This is an integration-evidence limitation, not a specialist or routing failure.

All TEST protections remain in force: BigEarthNet has no new test use; LEVIR-CC test content, labels, inference, and metrics remain zero; CDVQA test access remains zero.
