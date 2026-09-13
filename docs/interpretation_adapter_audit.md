# Pass 5E Interpretation and Hallucination Audit

Audit scope: `constrained_interpretation_v1` over `spatial_evidence_v1`, including real sample `61_39`, incomplete evidence, malformed semantic claims, and unsupported questions.

| Check | Result | Evidence |
|---|---|---|
| Every sentence grounded | PASS | Structured claims include feature, token/region, artifact, provenance, and exact emitted text |
| Epistemic categories separated | PASS | Output states inferred-from-derived; no prediction or measured-value language |
| Unsupported claims rejected | PASS | Non-allowlisted IDs and altered contract text fail closed |
| Spatial language supported | PASS | Positions derive from token `grid_position`; missing region references remove location wording |
| Optical/SAR/joint semantics | PASS | Sensor contracts enforced; complementary disclaimer; fake joint claim rejected |
| Confidence truthful | PASS | Only descriptive strength plus explicit uncalibrated wording |
| Provenance preserved | PASS | Evidence provenance is copied into the structured answer |
| Deterministic output | PASS | Exact repeated dictionaries and semantic hashes match |
| Real sample readable | PASS | Generated simple and technical artifacts contain three supported claims |
| API/UI exposure | PASS | Real loopback `POST /api/analyze` returned evidence, three claims, provenance, and traceability; visible result sections render the fields |
| Existing predictor unchanged | PASS | No task-model, CROMA, Pass 3, or Pass 5D code/result modification |

Adversarial tests cover houses/counts, vehicles, roads, company ownership, exact identity/address, fabricated forest/river/prediction wording, unsupported confidence, fake joint sensor attribution, unsupported claim IDs, and tampered output. None becomes user-facing scientific text. Missing optical evidence yields SAR-only interpretation; missing SAR evidence yields optical-only interpretation; no evidence yields `UNAVAILABLE`; missing spatial references yields a scene-only sentence.

Independent-review verdict: the adapter is ready as a constrained evidence presentation layer. It is not ready for arbitrary questions or semantics beyond its allowlist.

The final-code real API run completed the existing joint pipeline for `61_39`, built `spatial_evidence_v1`, preserved projected region extents, and returned `ANSWERED` with vegetation, water, and SAR claims. Added latency was 0.1385 s evidence construction plus 0.00171 s interpretation, 0.1402 s total, compared with the substantially heavier CROMA/inference path. Timing is machine/run specific.
