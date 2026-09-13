# Independent spatial-evidence audit

## Results

| Check | Result | Evidence |
|---|---|---|
| Token mapping/extents | PASS | Synthetic coordinate tests and 0–224 row-major validation |
| Physical aggregation | PASS | 64 valid pixels per token on `61_39`; formula/local tests |
| CROMA semantics | PASS | Real optical/SAR/joint `[225,768]` arrays referenced as latent only |
| Prediction typing | PASS | No `61_39` task output; explicitly `UNAVAILABLE`, never inferred from labels |
| Claim traceability | PASS | Feature, values, tokens, regions, artifacts and source hashes retained |
| Region determinism | PASS | Four-connected, sorted grouping; exact repeat hash |
| Sensor separation | PASS | Optical and SAR claims separate; no unsupported joint claim |
| Missing data | PASS | Optical-only/SAR-only tests; unavailable values not fabricated |
| Provenance | PASS | Dataset, ID, receipt/hash, preprocessing, feature/model/mapping versions, timestamp |
| Consumer readiness | PASS with scope | Stable JSON plus query API; no natural-language or VQA model |
| Temporal extensibility | PASS as schema | Separate timestamped scene objects can be compared later; no temporal model |

The complete object was built twice. Canonical SHA-256 matched exactly: `5906f4de2dbcb3d541590b57c802eba14e8a5d8585f591d88bc438eb3e6e3d8c`. Numerical tolerance was exact equality on this NumPy/runtime. Cross-platform floating-point reproduction should use a documented tolerance if later required.

The source artifact is the verified Phase 3.5/final-audit bundle. It contains real raw optical/SAR data and actual CROMA token arrays. The existing Pass 3 prediction models were not rerun and their outputs do not include this small fixture; therefore prediction evidence is honestly absent. Scene class summaries remain unavailable rather than substituting reference labels or untrained fusion values.

Final status: **PARTIALLY READY** for an interpretation layer. Structured retrieval, spatial claims, provenance and missing-data behavior are ready. Learned VQA/captioning, calibrated confidence, learned grounding and temporal prediction remain out of scope and unvalidated.
