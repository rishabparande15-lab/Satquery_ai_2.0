from dataclasses import replace

import pytest

from src.eo_vlm.grounding_contracts import ImageDimensions, validate_geometry
from src.spatial_contract import GeometryStatus
from src.evaluation.benchmark_specs import get_benchmark_spec
from src.evaluation.controller import decide_benchmark_execution
from src.evaluation.contracts import (BenchmarkPrediction, BenchmarkStatus, DatasetReadiness,
                                      VerificationStatus)
from src.evaluation.dry_run import (DRY_RUN_MARKERS, DryRunEvaluator, dry_run_artifacts,
                                    dry_run_model_compatibility, synthetic_fixtures,
                                    synthetic_grounding_geometries, validate_fixture, validate_prediction)
from src.evaluation.readiness import check_dataset_readiness


def complete_dataset(**changes):
    value = DatasetReadiness("synthetic", "dataset-r1", "benchmark-r1", *(VerificationStatus.VERIFIED,) * 7, {"test": ("fixture",)})
    return replace(value, **changes)


def ready_spec():
    return replace(get_benchmark_spec("RSVQA_BINARY"), benchmark_revision="benchmark-r1")


def dry_prediction(fixture, **changes):
    value = BenchmarkPrediction(fixture.benchmark_id, fixture.fixture_id, fixture.payload["target"], fixture.split, "dry-execution-1", "synthetic-model", "dry-run", "synthetic-adapter", "dry-run", {**DRY_RUN_MARKERS, "fixture_id": fixture.fixture_id}, target=fixture.payload["target"], target_reference=f"fixture:{fixture.fixture_id}:target")
    return replace(value, **changes)


def test_all_declared_benchmarks_execute_structurally_twice():
    evaluator = DryRunEvaluator()
    for benchmark_id, fixture in synthetic_fixtures().items():
        spec = get_benchmark_spec(benchmark_id)
        prediction = dry_prediction(fixture)
        first = evaluator.evaluate(prediction, spec, fixture)
        second = evaluator.evaluate(prediction, spec, fixture)
        assert first == second
    assert DryRunEvaluator.evaluator_version == "phase3n1"


def test_readiness_complete_metadata_is_ready_and_each_required_failure_is_preserved():
    ready = check_dataset_readiness(ready_spec(), complete_dataset(), split="test", sample_ids=("fixture",), model=dry_run_model_compatibility("QWEN_RGB_DRY_RUN", "RSVQA_BINARY"), evaluator_available=True)
    assert ready.status is BenchmarkStatus.EVALUATION_READY
    cases = [
        (complete_dataset(dataset_revision="UNKNOWN"), "DATASET_REVISION_UNKNOWN"),
        (complete_dataset(split_manifest={}), "SPLIT_UNKNOWN_OR_MISSING"),
        (complete_dataset(provenance_status=VerificationStatus.UNVERIFIED), "PROVENANCE_STATUS_UNVERIFIED"),
        (complete_dataset(), "TRAIN_TEST_OVERLAP"),
        (complete_dataset(modality_compatibility=VerificationStatus.UNVERIFIED), "MODALITY_COMPATIBILITY_UNVERIFIED"),
        (complete_dataset(license_status=VerificationStatus.UNVERIFIED), "LICENSE_STATUS_UNVERIFIED"),
    ]
    for dataset, reason in cases:
        result = check_dataset_readiness(ready_spec(), dataset, split="test", sample_ids=("fixture",), train_sample_ids=("fixture",) if reason == "TRAIN_TEST_OVERLAP" else (), model=dry_run_model_compatibility("QWEN_RGB_DRY_RUN", "RSVQA_BINARY"), evaluator_available=True)
        assert result.status is BenchmarkStatus.EVALUATION_BLOCKED and reason in result.reasons


@pytest.mark.parametrize("changes,reason", [
    ({"sample_id": "other"}, "SAMPLE_ID_MISMATCH"),
    ({"split": "train"}, "SPLIT_MISMATCH"),
    ({"benchmark_id": "RSVQA_MULTIPLE_CHOICE"}, "BENCHMARK_MISMATCH"),
    ({"provenance": {}}, "PROVENANCE_INVALID"),
    ({"prediction": None}, "PREDICTION_MALFORMED"),
])
def test_prediction_validator_fails_closed(changes, reason):
    fixture = synthetic_fixtures()["RSVQA_BINARY"]
    with pytest.raises(ValueError, match=reason):
        validate_prediction(dry_prediction(fixture, **changes), get_benchmark_spec("RSVQA_BINARY"), fixture)


def test_prediction_constructor_rejects_missing_identity_or_model_revision():
    fixture = synthetic_fixtures()["RSVQA_BINARY"]
    with pytest.raises(ValueError, match="sample_id"):
        dry_prediction(fixture, sample_id="")
    with pytest.raises(ValueError, match="model_revision"):
        dry_prediction(fixture, model_revision="")


def test_rsvqa_task_specific_binary_mcq_and_open_ended_dry_runs():
    fixtures = synthetic_fixtures(); evaluator = DryRunEvaluator()
    binary = evaluator.evaluate(dry_prediction(fixtures["RSVQA_BINARY"]), get_benchmark_spec("RSVQA_BINARY"), fixtures["RSVQA_BINARY"])
    mcq = evaluator.evaluate(dry_prediction(fixtures["RSVQA_MULTIPLE_CHOICE"], prediction="b"), get_benchmark_spec("RSVQA_MULTIPLE_CHOICE"), fixtures["RSVQA_MULTIPLE_CHOICE"])
    assert binary[0].value == 1.0 and binary[0].sample_count == 1 and binary[0].split == "test"
    assert mcq[0].value == 0.0  # hand-calculated: b != synthetic target a
    assert evaluator.evaluate(dry_prediction(fixtures["RSVQA_OPEN_ENDED"]), get_benchmark_spec("RSVQA_OPEN_ENDED"), fixtures["RSVQA_OPEN_ENDED"]) == ()


def test_grounding_structural_geometry_and_coordinate_failures():
    geometries = synthetic_grounding_geometries(); dimensions = ImageDimensions(10, 10)
    assert validate_geometry(geometries["point"], dimensions) is GeometryStatus.VALID
    assert validate_geometry(geometries["bbox"], dimensions) is GeometryStatus.VALID
    assert validate_geometry(geometries["mask"], dimensions) is GeometryStatus.VALID
    assert validate_geometry(geometries["invalid"], dimensions) is not GeometryStatus.VALID
    assert validate_geometry(geometries["coordinate_mismatch"], dimensions) is GeometryStatus.OUT_OF_BOUNDS
    fixture = synthetic_fixtures()["VRSBENCH_GROUNDING"]
    validate_fixture(fixture, get_benchmark_spec("VRSBENCH_GROUNDING"))
    with pytest.raises(ValueError, match="GROUNDING_GEOMETRY_INVALID"):
        validate_fixture(replace(fixture, payload={**fixture.payload, "target": geometries["coordinate_mismatch"]}), get_benchmark_spec("VRSBENCH_GROUNDING"))


@pytest.mark.parametrize("changes,reason", [
    ({"t1": ""}, "TEMPORAL_T1_MISSING"), ({"t2": ""}, "TEMPORAL_T2_MISSING"), ({"temporal_order": "UNKNOWN"}, "TEMPORAL_ORDER_UNKNOWN"),
])
def test_temporal_fixture_fails_closed(changes, reason):
    fixture = synthetic_fixtures()["CDVQA_CHANGE_VQA"]
    with pytest.raises(ValueError, match=reason):
        validate_fixture(replace(fixture, payload={**fixture.payload, **changes}), get_benchmark_spec("CDVQA_CHANGE_VQA"))


def test_dry_run_artifacts_are_marked_and_deterministic():
    fixture = synthetic_fixtures()["RSVQA_BINARY"]; prediction = dry_prediction(fixture)
    metrics = DryRunEvaluator().evaluate(prediction, get_benchmark_spec("RSVQA_BINARY"), fixture)
    first, second = dry_run_artifacts(fixture, prediction, metrics), dry_run_artifacts(fixture, prediction, metrics)
    assert first == second
    assert set(first) == {"evaluation_config.json", "dry_run_predictions.jsonl", "dry_run_metrics.json", "dry_run_provenance.json", "dry_run_report.md"}
    assert all(b"DRY_RUN" in value and b"NOT_BENCHMARK_RESULT" in value for value in first.values())


def test_only_explicit_dry_run_compatibility_passes_and_real_benchmark_controller_blocks():
    assert dry_run_model_compatibility("QWEN_RGB_DRY_RUN", "RSVQA_BINARY").status.value == "COMPATIBLE"
    for capability in ("S2_LEARNED", "SAR", "TEMPORAL", "GROUNDING"):
        assert dry_run_model_compatibility(capability, "RSVQA_BINARY").status.value == "BLOCKED"
    real = check_dataset_readiness(get_benchmark_spec("CDVQA_CHANGE_VQA"), DatasetReadiness(), split="test", sample_ids=("x",), model=None, evaluator_available=False)
    decision = decide_benchmark_execution(real)
    assert decision.status == "EVALUATION_BLOCKED" and decision.fallback == "NONE"
