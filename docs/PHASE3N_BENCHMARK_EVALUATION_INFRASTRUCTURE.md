# Phase 3N — Benchmark and Evaluation Infrastructure Hardening

## 1. Objective

This phase implements model-neutral, fail-closed evaluation infrastructure only. It introduces no dataset acquisition, benchmark execution, inference, training, generated benchmark predictions, metrics, or result artifacts. The frozen scientific baseline and all previously blocked/unverified capability statuses are unchanged.

## 2. Current benchmark blockers

Phase 3L.5 remains authoritative: RSVQA, VRSBench, and CDVQA have no complete local dataset–model–evaluator–result chain. SAR, optical-SAR, temporal/change, and learned grounding data gates remain blocked. Hidden ISRO/SAC access/protocol remains unverified. BigEarthNet.txt remains an optical image-language foundation and is not a benchmark substitute.

## 3. Benchmark registry

`src/evaluation/benchmark_specs.py` provides independent conservative definitions:

| Benchmark definition | Task | Modality | Dataset status | Notes |
|---|---|---|---|---|
| `RSVQA_BINARY` | single-image binary VQA | optical RGB | DATASET_UNVERIFIED | No SAR compatibility is inferred. |
| `RSVQA_MULTIPLE_CHOICE` | single-image multiple-choice VQA | optical RGB | DATASET_UNVERIFIED | Independent answer contract. |
| `RSVQA_OPEN_ENDED` | single-image open-ended VQA | UNKNOWN | DATASET_UNVERIFIED | Task fields and evaluator remain UNKNOWN. |
| `VRSBENCH_GROUNDING` | visual grounding | optical RGB | DATASET_UNVERIFIED | Bbox coordinate semantics and dimensions are mandatory. No VRSBench captioning task is claimed. |
| `CDVQA_CHANGE_VQA` | temporal change VQA | temporal UNKNOWN | DATASET_UNAVAILABLE | T1/T2 order and registration status are required. |
| `ISRO_SAC_HIDDEN_FINAL` | hidden final evaluation | UNKNOWN | DATASET_UNVERIFIED | Placeholder only; no access/data/protocol claim. |

## 4. Benchmark specifications

Each `BenchmarkSpec` records benchmark ID, task, modality, dataset and benchmark revision, supported splits, expected input/target/output, evaluator ID, metrics, provenance, license and leakage requirements. Unknown metadata is literally `UNKNOWN`. The registry has no BigEarthNet.txt entry.

## 5. Dataset readiness

`check_dataset_readiness()` in `src/evaluation/readiness.py` validates source and revisions; image and annotation availability; linkage; license; provenance; leakage; modality compatibility; requested split; selected sample IDs; train/test overlap; model compatibility; and evaluator availability. It returns `EVALUATION_READY` only if every applicable condition is verified.

Otherwise it returns `EVALUATION_BLOCKED` with all exact reasons, including `DATASET_UNAVAILABLE`, `SPLIT_UNKNOWN_OR_MISSING`, `DUPLICATE_SAMPLE_ID`, `TRAIN_TEST_OVERLAP`, `BENCHMARK_REVISION_MISMATCH`, invalid linkage, unavailable model, or unavailable evaluator. It never silently discards samples.

## 6. Prediction contract

`BenchmarkPrediction` records benchmark/sample IDs, prediction, split, execution ID, model and adapter IDs/revisions, provenance, and validation status. A target is optional. If one is included, an authoritative `target_reference` is mandatory; this prevents a record from presenting an unreferenced value as a benchmark target.

## 7. Evaluation and metric contracts

`BenchmarkEvaluator` is an abstract interface with `validate_predictions`, `validate_targets`, `compute_metrics`, `summarize`, `get_provenance`, and a composed `evaluate` method. It is not bound to Qwen, an EO model, or any tokenizer.

`MetricResult` records metric name/value/unit, split, sample count, evaluator version, optional confidence interval, and provenance. Confidence intervals and significance are never calculated automatically.

## 8. Provenance and result artifacts

Future real runs must retain benchmark/dataset revision, split, sample manifest, model/adapter/preprocessing/evaluator revisions, seed, configuration, and prediction/result hashes. `src/evaluation/provenance.py` declares—without creating—the deterministic artifact names:

`benchmark_manifest.json`, `evaluation_config.json`, `predictions.jsonl`, `metrics.json`, `provenance.json`, and `evaluation_report.md`.

Canonical serialization and hashing are reused from the existing annotation foundation rather than creating a second canonicalization framework.

## 9. Leakage protection and model compatibility

The readiness gate rejects missing/unknown splits, duplicate selected IDs, selected/training overlap, revision mismatch, unresolved provenance, invalid linkage, and any non-verified data readiness status. `ModelCompatibility` requires an explicit capability-to-benchmark compatibility result. The existence of Qwen RGB, an S2 projector, a SAR projector, a temporal adapter, or a grounding contract does not mark any blocked branch compatible.

## 10. Controller integration

`src/evaluation/controller.py` is an integration boundary for a future controller benchmark task. It maps a non-ready gate to `EVALUATION_BLOCKED`, exact reasons, and `Fallback: NONE`; ready is only a decision status and executes no evaluator. It cannot fall back to ordinary VLM generation.

## 11. Hidden ISRO/SAC

The `ISRO_SAC_HIDDEN_FINAL` placeholder preserves `DATASET_UNVERIFIED` and unknown contract/output fields. No synthetic hidden test data, evaluator, predictions, metrics, or result is created.

## 12. Implemented versus deferred

**Implemented:** registry, independent specifications, typed prediction/metric/provenance contracts, abstract evaluator interface, readiness and leakage rejection, compatibility contract, deterministic artifact schemas, and controller decision boundary.

**Deferred/blocked:** data admission, model execution, concrete official evaluators, actual benchmark prediction files, metrics, confidence intervals, results, benchmark controller routing, and all benchmark-dependent SAR/joint/temporal/grounding work. These retain their Phase 3L.5/3M statuses.

## 13. Test results

Focused structural/regression validation:

`python -m pytest -q tests/test_evaluation_infrastructure.py tests/test_image_language_foundation.py tests/test_agent_controller.py --basetemp=.pytest-phase3n`

Result: **32 passed in 0.34s**. The tests cover all registry definitions, BigEarthNet separation, readiness success and rejection, split/revision/linkage/leakage rejection, model/evaluator unavailability, prediction and metric contracts, evaluator versioning, deterministic serialization, controller block/no-fallback behavior, and hidden ISRO/SAC status. These are structural tests and do not generate benchmark scores.

## 14. Phase 3N.1 gate

Phase 3N.1 must be chosen only when an external remediation dependency changes:

- a benchmark dataset/release clears its established readiness gate → controlled benchmark execution;
- a fully verified SAR-language release appears → SAR source validation, then separately authorized adaptation;
- a verified paired optical-SAR release appears → pair validation, then fusion decision;
- a verified temporal release appears → pair/metadata validation, then temporal branch reconsideration;
- a verified grounding release appears → geometry validation, then grounding branch reconsideration;
- otherwise → reproducibility/application hardening with no benchmark claim.

Scientific baseline remains **65.0% accuracy**, **4.1034286734 pp MAE**, and **9.5873312123 pp RMSE**. No benchmark score was fabricated, and no commit or push was made.
