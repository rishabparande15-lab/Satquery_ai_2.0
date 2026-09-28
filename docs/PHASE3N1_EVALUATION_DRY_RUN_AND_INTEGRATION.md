# Phase 3N.1 — Evaluation Dry Run and Evaluator Integration Gate

## 1. Objective

This phase proves the evaluation infrastructure structurally using tiny synthetic in-memory fixtures. **NO REAL BENCHMARK WAS EXECUTED. NO BENCHMARK SCORE IS CLAIMED.** No external dataset, model, scientific artifact, checkpoint, split, or production controller behavior was used or changed.

## 2. Dry-run scope

`src/evaluation/dry_run.py` is explicitly synthetic-only. Every fixture and every serialized dry-run artifact carries these exact markers:

```text
DRY_RUN
SYNTHETIC_FIXTURE
NOT_BENCHMARK_RESULT
```

It returns artifacts as deterministic in-memory bytes for structural testing. It does not create benchmark-result directories or persistent benchmark artifacts.

## 3. Synthetic fixture definitions

Five minimum-field fixtures are defined:

| Fixture | Benchmark definition | Structural fields |
|---|---|---|
| `fixture-rsvqa-binary` | `RSVQA_BINARY` | synthetic RGB reference, question, binary target |
| `fixture-rsvqa-mcq` | `RSVQA_MULTIPLE_CHOICE` | synthetic RGB reference, question, choices, target |
| `fixture-rsvqa-open` | `RSVQA_OPEN_ENDED` | synthetic RGB reference, question, text target |
| `fixture-vrsbench` | `VRSBENCH_GROUNDING` | synthetic RGB reference, query, dimensions, bbox target |
| `fixture-cdvqa` | `CDVQA_CHANGE_VQA` | synthetic T1, T2, order, question, answer target |

These identifiers and values are not sourced from RSVQA, VRSBench, CDVQA, BigEarthNet.txt, RSRCC, or any other dataset.

## 4. Dataset readiness tests

The generic readiness gate was exercised with synthetic metadata. A complete synthetic declaration is `EVALUATION_READY` only when source/revisions, images, annotations, linkage, license, provenance, leakage, modality compatibility, split, non-overlapping IDs, explicit compatible model, and evaluator are all verified.

The following cases return `EVALUATION_BLOCKED` and preserve exact reason strings:

| Case | Reason |
|---|---|
| Missing dataset revision | `DATASET_REVISION_UNKNOWN` |
| Missing split manifest | `SPLIT_UNKNOWN_OR_MISSING` |
| Missing provenance | `PROVENANCE_STATUS_UNVERIFIED` |
| Same synthetic ID in train/test | `TRAIN_TEST_OVERLAP` |
| Modality mismatch/unverified | `MODALITY_COMPATIBILITY_UNVERIFIED` |
| License unknown | `LICENSE_STATUS_UNVERIFIED` |

No selected record is silently dropped or reassigned.

## 5. Prediction validation

Dry-run predictions use the normal `BenchmarkPrediction` contract and an authoritative synthetic target reference (`fixture:<fixture_id>:target`). The validator rejects benchmark mismatch, fixture/sample mismatch, split mismatch, unverified/missing dry-run provenance, malformed prediction, and invalid target linkage. The constructor independently rejects missing sample ID and model revision. Invalid records are rejected, never repaired.

## 6. RSVQA dry run

Binary, multiple-choice, and open-ended RSVQA specifications all validate structurally. The dry-run evaluator calculates only the metric explicitly declared by each specification: binary and multiple-choice synthetic accuracy. The hand-checked binary example is `yes == yes`, thus 1.0; the multiple-choice example is `b != a`, thus 0.0. Open-ended has no declared metric and produces no metric result. These structural values are not RSVQA scores.

## 7. VRSBench grounding dry run

The grounding fixture is synthetic only and uses no CROMA, BigEarthNet mapping, or real image. Point, bbox, and mask-reference geometry validate against synthetic 10×10 dimensions. A reversed bbox is invalid; a bbox extending past the synthetic dimensions returns `OUT_OF_BOUNDS`. The VRSBench fixture itself validates as bbox grounding. This confirms contract handling only, not a grounding prediction/evaluation.

## 8. CDVQA temporal dry run

The CDVQA fixture validates synthetic T1/T2 identities, non-unknown temporal order, question and target. Missing T1, missing T2, and `UNKNOWN` order each fail closed with an exact reason. The specification has no current metric contract, so structural evaluation produces no metric. This is not temporal reasoning, change detection, or CDVQA execution.

## 9. Metric validation

`DryRunEvaluator` is model-independent and versioned `phase3n1`. It validates fixture and prediction before returning only a declared synthetic `MetricResult`: name, deterministic value, `test` split, sample count 1, evaluator version, and dry-run provenance. It neither calculates confidence intervals nor statistical significance.

## 10. Evaluator integration

The exercised path is:

```text
synthetic fixture -> specification validation -> prediction validation
-> DryRunEvaluator -> permitted synthetic metric(s) -> marked artifact bytes
```

The path is run twice for every fixture; fixture validation and metric outputs are identical. It deliberately does not call Qwen, S2, SAR, temporal, grounding, scientific, or external evaluators.

## 11. Provenance and determinism

Each dry-run prediction/artifact contains benchmark ID, fixture ID, split, synthetic model ID/revision, evaluator version, execution ID, validation/provenance status, and dry-run markers. Two artifact builds from identical inputs are byte-identical. There is no timestamp in this structural harness; therefore no intentionally dynamic execution metadata needs normalization.

## 12. Leakage protection and compatibility

Synthetic same-ID train/test overlap is rejected. The explicit dry-run compatibility declaration admits only `QWEN_RGB_DRY_RUN` for RSVQA structural fixtures. Synthetic S2 learned, SAR, temporal, and grounding declarations are blocked. This does not change actual capability registry status or declare a real-model compatibility.

## 13. Controller integration and blocked behavior

For a real CDVQA specification with its real dataset status unavailable and no model/evaluator, the existing evaluation decision boundary returns:

```text
status: EVALUATION_BLOCKED
fallback: NONE
```

No ordinary VLM, scientific predictor, or S2 tool is selected as a substitute.

## 14. Result artifact validation

The dry-run artifact set is:

`evaluation_config.json`, `dry_run_predictions.jsonl`, `dry_run_metrics.json`, `dry_run_provenance.json`, and `dry_run_report.md`.

Tests assert that every artifact contains `DRY_RUN` and `NOT_BENCHMARK_RESULT`, and that two serializations are identical. They remain in memory/temporary test scope rather than benchmark-result locations.

## 15. Real benchmark status

| Benchmark | Structural dry run | Real dataset/execution/result |
|---|---|---|
| RSVQA | PASS | BLOCKED / no result |
| VRSBench grounding | PASS | BLOCKED / no result |
| CDVQA | PASS | DATASET_UNAVAILABLE / no result |
| Hidden ISRO/SAC | Placeholder status preserved | DATASET_UNVERIFIED / no result |

BigEarthNet.txt remains separate and was not used as a benchmark fixture.

## 16. Limitations

This gate validates contracts and fail-closed wiring only. It does not validate official evaluator semantics, dataset parsing, label normalization, real model outputs, benchmark performance, temporal registration, grounding geometry semantics, licenses, source provenance, or hidden-evaluation access. All Phase 3L.5 and Phase 3M blockers remain in force.

## 17. Test results

- `python -m pytest -q tests/test_evaluation_infrastructure.py tests/test_evaluation_dry_run.py --basetemp=.pytest-phase3n1-core` — **30 passed in 4.17s**.
- `python -m pytest -q tests/test_agent_controller.py tests/test_evidence_schema.py tests/test_multispectral_projector.py tests/test_sar_projector.py tests/test_temporal_contracts.py tests/test_temporal_rgb_adapter.py tests/test_grounding_contracts.py --basetemp=.pytest-phase3n1-regression` — **48 passed in 20.82s**.
- A larger combined selection emitted progress without a final pytest summary; it is treated as **INCONCLUSIVE**, not a pass.

## 18. Phase 3N.2 gate

Phase 3N.1 is complete: all defined benchmark specifications execute structurally; valid and invalid readiness/prediction paths are covered; permitted structural metrics are deterministic; evaluator/provenance/artifact contracts work; grounding and temporal structures fail closed; blocked real-benchmark behavior has `Fallback: NONE`; no benchmark score exists.

Phase 3N.2 is a **real benchmark execution readiness recheck**, not execution. It can move to controlled real evaluation only when the applicable dataset, split, model compatibility, evaluator, provenance, and license gates are verified under Phase 3M.

Scientific baseline remains **65.0% accuracy**, **4.1034286734 pp MAE**, and **9.5873312123 pp RMSE**. No commit or push was made.
