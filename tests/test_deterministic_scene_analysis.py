from __future__ import annotations

import copy

import pytest

from src.architecture_contracts import CapabilityRegistry, SceneBundle, TaskRequest
from src.deterministic_scene_analysis import (
    CAPABILITY_NAME,
    CAPABILITY_REGISTRY,
    TASK_TYPE,
    DeterministicSceneAnalysisCapability,
    Pipeline3AnalysisAdapter,
    build_capability_registry,
    run_deterministic_scene_analysis,
)


def _legacy_result():
    evidence = {
        "status": "AVAILABLE",
        "schema_version": "spatial_evidence_v1",
        "scene": {"sample_id": "61_39", "crs": "EPSG:32633", "image_shape": [120, 120]},
        "sensor_views": {"optical": {"status": "AVAILABLE"}, "sar": {"status": "AVAILABLE"}},
        "claims": [{"claim_id": "c1"}, {"claim_id": "c2"}, {"claim_id": "c3"}],
        "regions": [{"region_id": f"r{i}"} for i in range(17)],
        "provenance": {"analysis_id": "run-1", "method": "existing deterministic evidence"},
    }
    return {
        "schema_version": "2.0",
        "analysis_id": "run-1",
        "timestamp": "2026-09-14T00:00:00+00:00",
        "status": "completed",
        "query_interpretation": {"task": "joint_optical_sar_analysis", "modalities": ["optical", "sar"]},
        "data_retrieval": {"source": "local development provider", "assets": {"sample_id": "61_39"}},
        "validation": {"valid": True, "checks": {"optical_sar_compatibility": {"crs": True}}},
        "data_cube": {"modalities": ["optical", "sar"], "spatial_dimensions": [120, 120], "metadata": {}},
        "features": {
            "spectral": {"status": "computed", "schema": "physical_v1", "dimension": 62, "statistics": [1.0]},
            "deep": {"status": "computed", "source": "official pretrained CROMA base", "pooled_dimension": 2304, "values": [2.0]},
            "hybrid": {"status": "computed", "dimension": 192, "values": [3.0]},
        },
        "model_results": {"status": "pending training", "prediction": None},
        "interpretation": {"status": "ANSWERED", "answer": "Existing deterministic answer", "claims": evidence["claims"]},
        "spatial_evidence": evidence,
        "evidence": [{"kind": "feature statistics"}],
        "confidence": {"model_confidence": None, "calibration_status": "uncalibrated"},
        "preprocessing": [{"operation": "existing preprocessing", "status": "completed"}],
        "execution_trace": [{"step": 1}],
        "warnings": [],
        "error": None,
        "temporal": {"status": "not requested"},
        "llm_explanation": "Existing deterministic explanation",
    }


class RecordingRunner:
    def __init__(self):
        self.calls = []
        self.result = _legacy_result()

    def __call__(self, request):
        self.calls.append(copy.deepcopy(request))
        return self.result


def _request(**changes):
    values = {
        "task_type": TASK_TYPE,
        "scene_id": "61_39",
        "query": "Analyze joint optical and SAR evidence",
        "requested_modalities": ("optical", "sar"),
    }
    values.update(changes)
    return TaskRequest(**values)


def test_only_deterministic_scene_analysis_is_registered_with_honest_metadata():
    registry = build_capability_registry()
    assert [capability.name for capability in registry.list_capabilities()] == [CAPABILITY_NAME]
    capability = registry.require(CAPABILITY_NAME)
    assert registry.require_task_type(TASK_TYPE) is capability
    assert capability.to_dict() == {
        "name": "deterministic_scene_analysis",
        "task_type": "deterministic_scene_analysis",
        "version": "1.0.0",
        "supported_modalities": ["optical", "sar"],
        "supported_input_types": ["satellite_scene"],
        "supports_temporal": False,
        "supports_spatial_evidence": True,
        "supports_confidence": False,
        "input_contract": "TaskRequest",
        "output_contract": "TaskResult",
    }
    with pytest.raises(LookupError, match="unknown capability"):
        registry.require("vqa")
    with pytest.raises(LookupError, match="unknown task type"):
        registry.require_task_type("captioning")


def test_global_registry_contains_the_executable_capability():
    assert isinstance(CAPABILITY_REGISTRY.require(CAPABILITY_NAME), DeterministicSceneAnalysisCapability)


def test_canonical_entry_uses_registry_and_returns_task_result(monkeypatch):
    runner = RecordingRunner()
    registry = CapabilityRegistry()
    registry.register(DeterministicSceneAnalysisCapability(Pipeline3AnalysisAdapter(runner)))
    monkeypatch.setattr("src.deterministic_scene_analysis.CAPABILITY_REGISTRY", registry)
    result = run_deterministic_scene_analysis(_request())
    assert result.status == "success"
    assert len(runner.calls) == 1


def test_invalid_task_type_returns_structured_failure_without_execution():
    runner = RecordingRunner()
    capability = DeterministicSceneAnalysisCapability(Pipeline3AnalysisAdapter(runner))
    result = capability.execute(_request(task_type="vqa"))
    assert result.status == "failure"
    assert result.diagnostics["code"] == "unsupported_task_type"
    assert runner.calls == []


def test_missing_scene_input_returns_structured_failure():
    capability = DeterministicSceneAnalysisCapability(Pipeline3AnalysisAdapter(RecordingRunner()))
    request = TaskRequest(task_type=TASK_TYPE, scene_reference={}, query="Analyze scene")
    result = capability.execute(request)
    assert result.status == "failure"
    assert result.diagnostics["code"] == "missing_scene_input"


@pytest.mark.parametrize("modalities", [("thermal",), ("optical", "thermal")])
def test_unsupported_modalities_return_structured_unavailable(modalities):
    capability = DeterministicSceneAnalysisCapability(Pipeline3AnalysisAdapter(RecordingRunner()))
    result = capability.execute(_request(requested_modalities=modalities))
    assert result.status == "unavailable"
    assert result.diagnostics["code"] == "unsupported_modalities"


def test_temporal_request_is_explicitly_unavailable():
    capability = DeterministicSceneAnalysisCapability(Pipeline3AnalysisAdapter(RecordingRunner()))
    result = capability.execute(_request(temporal_context={"before": "a", "after": "b"}))
    assert result.status == "unavailable"
    assert result.diagnostics["code"] == "temporal_not_supported"


def test_scene_bundle_modality_validation_happens_before_execution():
    runner = RecordingRunner()
    capability = DeterministicSceneAnalysisCapability(Pipeline3AnalysisAdapter(runner))
    scene = SceneBundle(scene_id="61_39", available_modalities=("optical",))
    result = capability.execute(_request(), scene)
    assert result.status == "unavailable"
    assert result.diagnostics["code"] == "scene_modalities_unavailable"
    assert runner.calls == []


def test_execution_maps_existing_output_without_mutation_or_scientific_changes():
    runner = RecordingRunner()
    before = copy.deepcopy(runner.result)
    capability = DeterministicSceneAnalysisCapability(Pipeline3AnalysisAdapter(runner))

    result = capability.execute(_request())

    assert runner.calls == [{
        "sample_id": "61_39",
        "query": "Analyze joint optical and SAR evidence",
        "analysis_type": "joint_optical_sar_analysis",
    }]
    assert runner.result == before
    assert result.status == "success"
    assert result.task_type == TASK_TYPE
    assert result.output["analysis"] is runner.result
    assert result.output["features"] is runner.result["features"]
    assert result.output["interpretation"] is runner.result["interpretation"]
    assert result.evidence is runner.result["spatial_evidence"]
    assert len(result.evidence["claims"]) == len(runner.result["spatial_evidence"]["claims"]) == 3
    assert len(result.evidence["regions"]) == len(runner.result["spatial_evidence"]["regions"]) == 17
    assert result.provenance["run_id"] == runner.result["analysis_id"]
    assert result.provenance["preprocessing"] is runner.result["preprocessing"]
    assert result.provenance["execution"]["trace"] is runner.result["execution_trace"]
    assert result.confidence is None
    assert result.confidence_calibrated is False


def test_scene_reference_is_copied_and_request_modalities_select_the_fixed_pipeline_mode():
    runner = RecordingRunner()
    reference = {"request": {"files": {"sar": "upload-token"}}, "input_type": "satellite_scene"}
    before = copy.deepcopy(reference)
    request = TaskRequest(
        task_type=TASK_TYPE,
        scene_reference=reference,
        query="Analyze radar evidence",
        requested_modalities=("sar",),
    )
    capability = DeterministicSceneAnalysisCapability(Pipeline3AnalysisAdapter(runner))
    result = capability.execute(request)
    assert result.status == "success"
    assert reference == before
    assert runner.calls == [{
        "files": {"sar": "upload-token"},
        "query": "Analyze radar evidence",
        "analysis_type": "sar_analysis",
    }]
