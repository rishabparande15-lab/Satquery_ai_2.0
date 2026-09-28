from dataclasses import replace

import pytest

from src.evaluation.benchmark_specs import get_benchmark_spec
from src.evaluation.contracts import (CompatibilityStatus, DatasetReadiness, ModelCompatibility,
                                      RealEvaluationStatus, VerificationStatus)
from src.evaluation.readiness import real_evaluation_gate


def spec():
    return replace(get_benchmark_spec("RSVQA_BINARY"), benchmark_revision="benchmark-r1")


def dataset(**changes):
    value = DatasetReadiness("official", "dataset-r1", "benchmark-r1", *(VerificationStatus.VERIFIED,) * 7, {"test": ("s1",)})
    return replace(value, **changes)


def model(status=CompatibilityStatus.COMPATIBLE):
    return ModelCompatibility("rgb", "RSVQA_BINARY", status, "test")


def decision(data=None, **changes):
    values = {"split": "test", "sample_ids": ("s1",), "model": model(), "evaluator_available": True, "prediction_compatible": True}
    values.update(changes)
    return real_evaluation_gate(spec(), data or dataset(), **values)


def test_complete_synthetic_contract_is_ready_and_repeatable():
    first, second = decision(), decision()
    assert first == second
    assert first.status is RealEvaluationStatus.READY_FOR_REAL_EVALUATION


@pytest.mark.parametrize("data,kwargs,reason", [
    (dataset(dataset_revision="UNKNOWN"), {}, "DATASET_REVISION_UNKNOWN"),
    (dataset(split_manifest={}), {}, "SPLIT_UNKNOWN_OR_MISSING"),
    (dataset(provenance_status=VerificationStatus.UNVERIFIED), {}, "PROVENANCE_STATUS_UNVERIFIED"),
    (dataset(license_status=VerificationStatus.UNVERIFIED), {}, "LICENSE_STATUS_UNVERIFIED"),
    (dataset(leakage_status=VerificationStatus.UNVERIFIED), {}, "LEAKAGE_STATUS_UNVERIFIED"),
    (dataset(modality_compatibility=VerificationStatus.UNVERIFIED), {}, "MODALITY_COMPATIBILITY_UNVERIFIED"),
    (dataset(), {"train_sample_ids": ("s1",)}, "TRAIN_TEST_OVERLAP"),
])
def test_data_gate_fail_closed(data, kwargs, reason):
    result = decision(data, **kwargs)
    assert result.status is RealEvaluationStatus.BLOCKED and reason in result.reasons


def test_dataset_ready_but_model_evaluator_or_prediction_incompatible_is_partial():
    assert decision(model=model(CompatibilityStatus.BLOCKED)).status is RealEvaluationStatus.PARTIALLY_READY
    assert decision(evaluator_available=False).status is RealEvaluationStatus.PARTIALLY_READY
    assert decision(prediction_compatible=False).status is RealEvaluationStatus.PARTIALLY_READY


def test_hidden_dataset_is_unverified():
    hidden = get_benchmark_spec("ISRO_SAC_HIDDEN_FINAL")
    result = real_evaluation_gate(hidden, DatasetReadiness(), split="test", sample_ids=("hidden",))
    assert result.status is RealEvaluationStatus.UNVERIFIED
    assert result.reasons == ("HIDDEN_DATASET_UNVERIFIED",)


def test_current_real_specs_cannot_execute_without_real_metadata_or_models():
    for name in ("RSVQA_BINARY", "RSVQA_MULTIPLE_CHOICE", "RSVQA_OPEN_ENDED", "VRSBENCH_GROUNDING", "CDVQA_CHANGE_VQA"):
        result = real_evaluation_gate(get_benchmark_spec(name), DatasetReadiness(), split="test", sample_ids=("missing",))
        assert result.status is RealEvaluationStatus.BLOCKED
