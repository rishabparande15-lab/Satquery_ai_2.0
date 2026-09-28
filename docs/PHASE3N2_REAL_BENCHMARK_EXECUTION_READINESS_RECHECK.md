# Phase 3N.2 — Real Benchmark Execution Readiness Recheck

## 1. Executive status

**Phase 3N.2 result: COMPLETE AS A READINESS AUDIT; no benchmark is authorized to run.** The deterministic real-execution gate was added and tested. Current classifications are:

| Target | Status |
|---|---|
| RSVQA binary / multiple-choice / open-ended | BLOCKED |
| VRSBench grounding | BLOCKED |
| CDVQA change VQA | BLOCKED |
| Hidden ISRO/SAC final | UNVERIFIED |

**NO REAL BENCHMARK SCORE WAS GENERATED.** No benchmark data was downloaded, no benchmark inference was run, and no scientific artifact changed.

Machine-readable matrix: [`artifacts/audits/phase3n2_real_benchmark_readiness.json`](../artifacts/audits/phase3n2_real_benchmark_readiness.json).

## 2. Authoritative readiness matrix

The JSON matrix records all required identity, revision, availability, acquisition, license, split, linkage, modality, spatial/temporal semantics, provenance/checksum, duplicate/leakage, prediction/model/evaluator compatibility, bounded acquisition, reproducibility, blockers, and evidence-reference fields. Its results are summarized here.

| Target | Dataset files | License / revision | Split / linkage / provenance | Semantics | Model / evaluator | Execution decision |
|---|---|---|---|---|---|---|
| RSVQA | UNAVAILABLE | UNVERIFIED | UNVERIFIED | RGB task contract only | Qwen RGB smoke is partial; evaluator unavailable | BLOCKED |
| VRSBench | UNAVAILABLE | PARTIAL / not admission-ready | UNVERIFIED | Grounding geometry mapping blocked | learned grounder/evaluator unavailable | BLOCKED |
| CDVQA | UNAVAILABLE (annotation-level audit only) | PARTIAL | PARTIAL/UNVERIFIED | temporal ordering/registration evidence blocked | temporal model/evaluator unavailable | BLOCKED |
| Hidden ISRO/SAC | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |

## 3. RSVQA assessment

The repository has no verified real RSVQA images, questions, answers, authoritative split, immutable release receipt, license evidence, duplicate/leakage audit, evaluator, or reproducible benchmark manifest. Phase 2E exposes RSVQA adapter hooks but does not implement an evaluator; Phase 3N/3N.1 adds only structural contracts and synthetic dry runs. Qwen RGB was validated only as a bounded controlled-image CUDA generation, and the S2 adapter is a POC; neither supplies an RSVQA benchmark claim. All three RSVQA task variants are therefore **BLOCKED**.

**Single most important unblock:** an immutable, license-bearing RSVQA image–question–answer release with official partitions and record linkage; the evaluator/model/provenance/leakage gate must then also pass.

## 4. VRSBench assessment

VRSBench remains **BLOCKED** under two independent gates. The dataset audit does not establish a locally available, fully linked, license-cleared, immutable image/geometry release with authoritative coordinate semantics, dimensions, official split, or leakage evidence. Independently, SatQuery grounding is a fail-closed contract-only boundary; no learned grounding model can produce genuine boxes, points, masks, or polygons. Phase 2F remains unchanged: BigEarthNet source geometry is preserved and `MAPPING_UNVERIFIED`, never converted to grounding evidence.

**Single most important unblock:** a fully auditable geometry release with authoritative coordinate/image semantics and linkage. It does not by itself unlock execution; a genuine grounding model and evaluator must follow.

## 5. CDVQA assessment

CDVQA remains **BLOCKED**. The audit identifies change-QA supervision only at a partial annotation/design level; current repository evidence does not establish an immutable, usable image-pair release, complete image/question/answer linkage, source revision/checksums, timestamp/interval semantics, reproducible registration, license, or complete leakage audit. The temporal branch is frozen at an RGB execution boundary. A real RSRCC RGB pair run demonstrates neither learned temporal/change reasoning nor CDVQA compatibility. No temporal model or official evaluator is available.

**Single most important unblock:** a verified CDVQA (or specification-compatible) immutable T1/T2 image-pair/question/answer release with ordered IDs, registration and provenance evidence. A compatible temporal model/evaluator remains separately required.

## 6. Hidden ISRO/SAC assessment

Hidden ISRO/SAC remains **UNVERIFIED**. No repository artifact establishes authorized access to Cartosat-2S optical, RISAT SAR, pre-georeferenced/co-registered data, pair identity, metadata, hidden-set membership, license, split ownership, checksums, evaluation protocol, prediction schema, or result-submission procedure. Project descriptions cannot establish any of these facts.

**Single most important unblock:** documented authorization plus the owner-provided hidden-evaluation data/protocol contract. Hidden data must remain excluded from model selection, training, calibration, and design decisions.

## 7. Model compatibility matrix

| Current path | RSVQA | VRSBench | CDVQA | Reason |
|---|---|---|---|---|
| Qwen2.5-VL-3B RGB | PARTIAL: controlled RGB smoke only | BLOCKED | BLOCKED | No official benchmark compatibility/result; no grounding or learned temporal capability. |
| S2 learned projector | UNVERIFIED | BLOCKED | BLOCKED | POC language-loss adapter, no held-out benchmark execution. |
| SAR projector | BLOCKED | BLOCKED | BLOCKED | SAR language supervision/training blocked. |
| Optical-SAR fusion | BLOCKED | BLOCKED | BLOCKED | No verified joint data or trained fusion. |
| Temporal RGB adapter | BLOCKED | N/A | BLOCKED | RGB pair execution only; learned change reasoning frozen. |
| Grounding contract | N/A | BLOCKED | N/A | Contract-only; no learned grounding prediction. |
| `hybrid_830d` scientific predictor | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | Scientific-predictor-only; never a VLM benchmark substitute. |

## 8. Execution-gate logic

`real_evaluation_gate()` in `src/evaluation/readiness.py` is the single deterministic gate. It first evaluates every mandatory data condition through the existing readiness check: source/revisions, dataset availability, image/annotation/linkage, license, provenance, leakage, modality, split, duplicate and train/test overlap. Any failure returns **BLOCKED** with every exact reason.

Only after the data gate passes does it evaluate model, evaluator, and prediction-format compatibility. A verified dataset with one of those execution prerequisites unavailable is **PARTIALLY_READY**, never executable. All pass only yields **READY_FOR_REAL_EVALUATION**. The hidden specification with no authorized source returns **UNVERIFIED**. Every other unknown is fail-closed.

## 9. Exact blockers

- **RSVQA:** immutable image/question/answer release, license, official split/linkage, provenance/checksums, duplicate/leakage audit, evaluator, and held-out compatible model path.
- **VRSBench:** image/geometry release, authoritative coordinate/image semantics, linkage, license/pin/split/leakage evidence, learned grounding model, evaluator.
- **CDVQA:** image pairs, T1/T2/question/answer linkage, temporal order/timestamps/registration, license/pin/split/leakage evidence, temporal model, evaluator.
- **ISRO/SAC:** explicit owner authorization and complete hidden-evaluation protocol/data contract.

## 10. What changed from Phase 3N.1

The repository now has the high-level `RealEvaluationStatus`/`RealEvaluationDecision` and `real_evaluation_gate()`, plus focused readiness tests. This gate is additive and does not route or execute a benchmark. The deterministic JSON matrix records the present audit result.

## 11. What did not change

No benchmark conclusions were upgraded. No dataset, evaluator, model, adapter, controller production behavior, scientific input/split/checkpoint/receipt, S2 artifact, SAR status, temporal freeze, grounding freeze, Phase 2F semantics, or scientific baseline was changed. No real data acquisition, inference, result artifact, or score occurred.

## 12. Validation

`python -m pytest -q tests/test_real_evaluation_readiness.py tests/test_evaluation_infrastructure.py tests/test_evaluation_dry_run.py tests/test_agent_controller.py tests/test_evidence_schema.py --basetemp=.pytest-phase3n2-final` — **62 passed in 9.28s**.

`python -m pytest -q tests/test_multispectral_projector.py tests/test_sar_projector.py tests/test_temporal_contracts.py tests/test_temporal_rgb_adapter.py tests/test_grounding_contracts.py --basetemp=.pytest-phase3n2-modality` — **27 passed in 13.04s**.

Full-suite status remains `FULL_SUITE_INCONCLUSIVE`; no full-suite pass is claimed.

## 13. Recommended next phase

Remain at **REAL BENCHMARK EXECUTION READINESS RECHECK** until an external source gate changes. The next action must be triggered by evidence, not preference: receive a verified benchmark release, SAR/joint/temporal/grounding dataset, or authorized ISRO/SAC protocol; then run the corresponding Phase 3M source-readiness gate before any controlled benchmark execution.

Scientific baseline verification: **65.0% accuracy, 4.1034286734 pp MAE, 9.5873312123 pp RMSE**. Current HEAD at audit start: `2903f93172f4fb030728bc87195b60321c2c35b7`. No commit or push was made.
