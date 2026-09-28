from dataclasses import replace
import json
from pathlib import Path

from src.evaluation.benchmark_specs import get_benchmark_spec
from src.evaluation.contracts import CompatibilityStatus, ModelCompatibility, RealEvaluationStatus
from src.evaluation.readiness import real_evaluation_gate
from src.evaluation.rsvqa import AcceptanceState, RSVQADatasetAcceptance, rsvqa_acceptance_schema


def complete_acceptance(**changes):
    values = {name: AcceptanceState.VERIFIED for name in rsvqa_acceptance_schema()}
    values.update(changes)
    return RSVQADatasetAcceptance(**values)


def model(status=CompatibilityStatus.COMPATIBLE):
    return ModelCompatibility("rsvqa_rgb", "RSVQA_BINARY", status, "synthetic gate test")


def ready_decision(acceptance=None, **changes):
    spec = replace(get_benchmark_spec("RSVQA_BINARY"), benchmark_revision="RSVQA_VERIFIED_BENCHMARK_REVISION")
    values = {"split": "test", "sample_ids": ("synthetic",), "model": model(), "evaluator_available": True, "prediction_compatible": True}
    values.update(changes)
    return real_evaluation_gate(spec, (acceptance or complete_acceptance()).to_dataset_readiness(), **values)


def test_acceptance_schema_has_every_required_field():
    assert rsvqa_acceptance_schema() == (
        "dataset_identity", "dataset_version_or_revision", "image_manifest", "question_manifest", "answer_manifest", "image_question_linkage", "question_answer_linkage", "official_train_split", "official_validation_split", "official_test_split", "licensing_evidence", "provenance", "checksums_or_equivalent_artifact_identity", "duplicate_check", "cross_split_leakage_check", "near_duplicate_check_if_applicable", "modality_definition",
    )


def test_complete_synthetic_acceptance_allows_only_synthetic_gate_readiness_and_is_deterministic():
    first, second = ready_decision(), ready_decision()
    assert first == second and first.status is RealEvaluationStatus.READY_FOR_REAL_EVALUATION


def test_default_local_acceptance_is_blocked_and_reports_local_missing_fields():
    blockers = RSVQADatasetAcceptance().blocking_reasons()
    assert "RSVQA_IMAGE_MANIFEST_MISSING" in blockers
    assert "RSVQA_OFFICIAL_TEST_SPLIT_MISSING" in blockers
    assert "RSVQA_LICENSING_EVIDENCE_UNVERIFIED" in blockers
    assert ready_decision(RSVQADatasetAcceptance()).status is RealEvaluationStatus.BLOCKED


def test_each_dataset_gate_category_fails_closed():
    for field in ("dataset_version_or_revision", "official_test_split", "image_question_linkage", "provenance", "licensing_evidence", "cross_split_leakage_check"):
        acceptance = complete_acceptance(**{field: AcceptanceState.MISSING})
        assert acceptance.blocking_reasons()
        assert ready_decision(acceptance).status is RealEvaluationStatus.BLOCKED


def test_model_evaluator_prediction_and_rs_adaptation_gaps_do_not_allow_execution():
    assert ready_decision(model=model(CompatibilityStatus.BLOCKED)).status is RealEvaluationStatus.PARTIALLY_READY
    assert ready_decision(evaluator_available=False).status is RealEvaluationStatus.PARTIALLY_READY
    assert ready_decision(prediction_compatible=False).status is RealEvaluationStatus.PARTIALLY_READY
    # Current actual source acceptance is missing, so even an incompatible model is blocked at the data gate.
    assert ready_decision(RSVQADatasetAcceptance(), model=model(CompatibilityStatus.BLOCKED)).status is RealEvaluationStatus.BLOCKED


def test_phase3n3_artifact_has_required_schema_and_is_not_executable():
    path = Path("artifacts/audits/phase3n3_rsvqa_readiness.json")
    if not path.exists():
        # The report artifact is created in this same phase; keeping this guard produces a clear structural failure when absent.
        raise AssertionError("Phase 3N.3 readiness artifact missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    assert {"benchmark", "status", "dataset", "model", "evaluator", "splits", "leakage", "provenance", "license", "blocking_reasons", "execution_permitted"}.issubset(value)
    assert value["benchmark"] == "RSVQA" and value["execution_permitted"] is False
