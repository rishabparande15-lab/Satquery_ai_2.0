from dataclasses import replace

import pytest

from src.evaluation.benchmark_specs import BENCHMARK_REGISTRY, get_benchmark_spec
from src.evaluation.controller import decide_benchmark_execution
from src.evaluation.contracts import (BenchmarkPrediction, BenchmarkStatus, CompatibilityStatus,
                                      DatasetReadiness, MetricResult, ModelCompatibility,
                                      VerificationStatus)
from src.evaluation.evaluator import BenchmarkEvaluator
from src.evaluation.provenance import RESULT_ARTIFACTS, result_artifact_hash, result_artifact_schema
from src.evaluation.readiness import check_dataset_readiness


def spec():
    return replace(get_benchmark_spec("RSVQA_BINARY"), benchmark_revision="benchmark-r1", dataset_revision="dataset-r1")


def ready_dataset(**changes):
    value = DatasetReadiness(
        source="official", dataset_revision="dataset-r1", benchmark_revision="benchmark-r1",
        image_availability=VerificationStatus.VERIFIED, annotation_availability=VerificationStatus.VERIFIED,
        sample_linkage=VerificationStatus.VERIFIED, license_status=VerificationStatus.VERIFIED,
        provenance_status=VerificationStatus.VERIFIED, leakage_status=VerificationStatus.VERIFIED,
        modality_compatibility=VerificationStatus.VERIFIED, split_manifest={"test": ("s1", "s2")},
    )
    return replace(value, **changes)


def compatible():
    return ModelCompatibility("qwen_rgb", "RSVQA_BINARY", CompatibilityStatus.COMPATIBLE, "validated task input", "model", "rev", "adapter", "rev")


def prediction(**changes):
    value = BenchmarkPrediction("RSVQA_BINARY", "s1", "yes", "test", "execution-1", "model", "rev", "adapter", "rev", {"dataset_revision": "dataset-r1"})
    return replace(value, **changes)


def test_registry_has_independent_specs_and_bigearthnet_is_separate():
    assert {"RSVQA_BINARY", "RSVQA_MULTIPLE_CHOICE", "RSVQA_OPEN_ENDED", "VRSBENCH_GROUNDING", "CDVQA_CHANGE_VQA"}.issubset(BENCHMARK_REGISTRY)
    assert "BigEarthNet.txt" not in BENCHMARK_REGISTRY
    assert get_benchmark_spec("CDVQA_CHANGE_VQA").dataset_status is BenchmarkStatus.DATASET_UNAVAILABLE


def test_rsvqa_variants_vrsbench_and_cdvqa_contracts_are_not_collapsed():
    assert get_benchmark_spec("RSVQA_BINARY").task_type != get_benchmark_spec("RSVQA_MULTIPLE_CHOICE").task_type
    assert get_benchmark_spec("VRSBENCH_GROUNDING").expected_target["geometry"] == "bbox"
    assert get_benchmark_spec("CDVQA_CHANGE_VQA").expected_input["image_count"] == 2


def test_readiness_accepts_only_complete_verified_inputs():
    result = check_dataset_readiness(spec(), ready_dataset(), split="test", sample_ids=("s1", "s2"), model=compatible(), evaluator_available=True)
    assert result.ready and result.status is BenchmarkStatus.EVALUATION_READY


@pytest.mark.parametrize("dataset,kwargs,reason", [
    (ready_dataset(), {"sample_ids": ("s1", "s1")}, "DUPLICATE_SAMPLE_ID"),
    (ready_dataset(), {"sample_ids": ("s1",), "train_sample_ids": ("s1",)}, "TRAIN_TEST_OVERLAP"),
    (ready_dataset(benchmark_revision="different"), {"sample_ids": ("s1",)}, "BENCHMARK_REVISION_MISMATCH"),
    (ready_dataset(sample_linkage=VerificationStatus.UNVERIFIED), {"sample_ids": ("s1",)}, "SAMPLE_LINKAGE_UNVERIFIED"),
    (ready_dataset(), {"split": "UNKNOWN", "sample_ids": ("s1",)}, "SPLIT_UNKNOWN_OR_MISSING"),
])
def test_readiness_rejects_leakage_revision_linkage_and_unknown_split(dataset, kwargs, reason):
    values = {"split": "test", "sample_ids": ("s1",), "model": compatible(), "evaluator_available": True}
    values.update(kwargs)
    result = check_dataset_readiness(spec(), dataset, **values)
    assert result.status is BenchmarkStatus.EVALUATION_BLOCKED and reason in result.reasons


def test_readiness_rejects_unavailable_model_and_evaluator():
    result = check_dataset_readiness(spec(), ready_dataset(), split="test", sample_ids=("s1",), model=ModelCompatibility("sar", "RSVQA_BINARY", CompatibilityStatus.BLOCKED, "SAR blocked"), evaluator_available=False)
    assert "MODEL_UNAVAILABLE_OR_INCOMPATIBLE" in result.reasons
    assert "EVALUATOR_UNAVAILABLE" in result.reasons


def test_prediction_requires_authoritative_target_reference_and_preserves_unknown_target():
    assert prediction().target is None
    with pytest.raises(ValueError, match="target_reference"):
        prediction(target="yes")
    assert prediction(target="yes", target_reference="annotation:s1").to_dict()["target_reference"] == "annotation:s1"


def test_metric_contract_never_invents_confidence_interval():
    metric = MetricResult("accuracy", 1.0, "test", 2, "v1", {"execution_id": "x"})
    assert metric.confidence_interval is None
    with pytest.raises(ValueError, match="sample_count"):
        MetricResult("accuracy", 1.0, "test", -1, "v1", {})


class StubEvaluator(BenchmarkEvaluator):
    evaluator_id = "stub"
    evaluator_version = "1.0"
    def validate_predictions(self, predictions): assert all(item.prediction is not None for item in predictions)
    def validate_targets(self, predictions): assert all(item.target_reference for item in predictions)
    def compute_metrics(self, predictions): return (MetricResult("exact", 1.0, "test", len(tuple(predictions)), self.evaluator_version, {"evaluator": self.evaluator_id}),)
    def summarize(self, metrics): return {"count": len(tuple(metrics))}
    def get_provenance(self): return {"evaluator_id": self.evaluator_id, "evaluator_version": self.evaluator_version}


def test_evaluator_interface_is_model_independent_and_versioned():
    evaluator = StubEvaluator()
    assert evaluator.get_provenance()["evaluator_version"] == "1.0"
    assert evaluator.evaluate((prediction(target="yes", target_reference="annotation:s1"),))[0].metric_name == "exact"


def test_result_artifacts_and_serialization_are_deterministic():
    assert set(RESULT_ARTIFACTS) == set(result_artifact_schema())
    value = {"benchmark_id": "RSVQA_BINARY", "split": "test"}
    assert result_artifact_hash(value) == result_artifact_hash(dict(reversed(tuple(value.items()))))


def test_blocked_controller_boundary_has_no_fallback():
    readiness = check_dataset_readiness(spec(), ready_dataset(), split="test", sample_ids=("s1",), model=None, evaluator_available=False)
    decision = decide_benchmark_execution(readiness)
    assert decision.status == "EVALUATION_BLOCKED" and decision.fallback == "NONE"


def test_hidden_isro_sac_stays_unverified():
    hidden = get_benchmark_spec("ISRO_SAC_HIDDEN_FINAL")
    assert hidden.dataset_status is BenchmarkStatus.DATASET_UNVERIFIED
    assert hidden.expected_input["contract"] == "UNKNOWN"
