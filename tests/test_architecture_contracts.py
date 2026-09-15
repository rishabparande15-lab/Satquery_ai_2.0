from __future__ import annotations

import copy

import pytest

from src.analysis_engine import run_analysis
from src.architecture_contracts import (
    Capability,
    CapabilityRegistry,
    RepresentationSet,
    SceneBundle,
    TaskRequest,
    TaskResult,
    adapt_run_analysis_output,
)


def test_minimal_scene_allows_future_fields_to_be_absent():
    scene = SceneBundle(scene_id="scene-1")
    assert scene.available_modalities == ()
    assert scene.optical_available is False
    assert scene.sar_available is False
    assert scene.temporal_context is None
    assert scene.representations.spatial_tokens is None


@pytest.mark.parametrize(("modality", "optical", "sar"), [
    ("optical", True, False),
    ("sar", False, True),
])
def test_single_modality_scenes(modality, optical, sar):
    scene = SceneBundle(scene_id=f"{modality}-scene", available_modalities=(modality,))
    assert scene.optical_available is optical
    assert scene.sar_available is sar


@pytest.mark.parametrize("kwargs", [
    {"scene_id": ""},
    {"scene_id": "x", "bounds": (2, 0, 1, 1)},
    {"scene_id": "x", "resolution": (10, 0)},
    {"scene_id": "x", "dimensions": (120, -1)},
    {"scene_id": "x", "available_modalities": ("optical",), "optical_available": False},
])
def test_invalid_scene_metadata_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SceneBundle(**kwargs)


def test_valid_task_request_with_optional_parameters():
    request = TaskRequest(
        task_type="classification",
        scene_id="scene-1",
        requested_modalities=("optical",),
        parameters={"top_k": 3},
        constraints={"latency_ms": 500},
    )
    assert request.parameters["top_k"] == 3
    assert request.query is None


@pytest.mark.parametrize("kwargs", [
    {"task_type": "", "scene_id": "scene-1"},
    {"task_type": "classification"},
    {"task_type": "vqa", "scene_id": "scene-1"},
    {"task_type": "classification", "scene_id": "scene-1", "query": ""},
])
def test_malformed_task_request_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TaskRequest(**kwargs)


def test_structurally_impossible_modality_combinations_are_rejected():
    with pytest.raises(ValueError, match="optical, sar"):
        TaskRequest(task_type="optical_sar_analysis", scene_id="scene-1", requested_modalities=("optical",))
    with pytest.raises(ValueError, match="temporal_context"):
        TaskRequest(task_type="temporal_change", scene_id="scene-1", requested_modalities=("optical",))


def test_success_and_failure_results_keep_evidence_optional():
    success = TaskResult(status="success", task_type="classification", output={"label": "forest"})
    failure = TaskResult(status="failure", task_type="classification", diagnostics={"error": "not available"})
    assert success.evidence is None
    assert failure.output is None


def test_confidence_is_unavailable_unless_explicitly_calibrated():
    assert TaskResult(status="success", task_type="classification").confidence is None
    with pytest.raises(ValueError, match="calibrated"):
        TaskResult(status="success", task_type="classification", confidence=.9)
    assert TaskResult(status="success", task_type="classification", confidence=.9, confidence_calibrated=True).confidence == .9


def test_result_provenance_propagates_without_reformatting():
    provenance = {"run_id": "run-1", "representation": {"version": "croma-v1"}}
    result = TaskResult(status="partial", task_type="classification", provenance=provenance)
    assert result.provenance is provenance


def test_capability_registration_lookup_listing_and_matching():
    registry = CapabilityRegistry()
    sar = Capability(name="sar-classifier", task_type="classification", version="1.0", supported_modalities=frozenset({"sar"}))
    optical = Capability(name="optical-classifier", task_type="classification", version="1.0", supported_modalities=frozenset({"optical"}))
    registry.register(sar)
    registry.register(optical)
    assert registry.lookup("sar-classifier") is sar
    assert [item.name for item in registry.list_capabilities()] == ["optical-classifier", "sar-classifier"]
    request = TaskRequest(task_type="classification", scene_id="s", requested_modalities=("optical",))
    scene = SceneBundle(scene_id="s", available_modalities=("optical",))
    assert registry.supports(request, scene)
    assert registry.matching(request, scene) == (optical,)
    assert optical.to_dict()["supported_modalities"] == ["optical"]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(optical)


def test_capability_rejects_scene_without_requested_modality():
    capability = Capability(name="classifier", task_type="classification", version="1", supported_modalities=frozenset({"optical", "sar"}))
    request = TaskRequest(task_type="classification", scene_id="s", requested_modalities=("sar",))
    assert not capability.supports(request, SceneBundle(scene_id="s", available_modalities=("optical",)))


def _legacy_result():
    evidence = {
        "status": "AVAILABLE",
        "schema_version": "spatial_evidence_v1",
        "scene": {"sample_id": "61_39", "crs": "EPSG:32633", "image_shape": [120, 120]},
        "sensor_views": {},
        "claims": [{"claim_id": "vegetation", "strength": "MODERATE"}],
        "regions": [{"region_id": "vegetation-region-000", "token_indices": [1, 2]}],
        "provenance": {"analysis_id": "11111111-1111-1111-1111-111111111111"},
    }
    return {
        "schema_version": "2.0",
        "analysis_id": "11111111-1111-1111-1111-111111111111",
        "timestamp": "2026-09-14T00:00:00+00:00",
        "status": "completed",
        "query_interpretation": {"task": "joint_optical_sar_analysis", "modalities": ["optical", "sar"]},
        "data_retrieval": {"source": "local development provider", "assets": {"sample_id": "61_39"}},
        "validation": {"valid": True, "checks": {"optical_sar_compatibility": {"crs": True, "resolution": True, "extent": True, "transform": True, "shape": True}}},
        "data_cube": {"modalities": ["optical", "sar"], "spatial_dimensions": [120, 120], "metadata": {"optical": {"crs": "EPSG:32633", "bounds": [0, 0, 1200, 1200], "resolution": [10, 10], "transform": [10, 0, 0, 0, -10, 1200, 0, 0, 1]}}},
        "features": {"spectral": {"schema": "physical_v1", "statistics": [1.25, 2.5]}, "deep": {"source": "official pretrained CROMA base", "values": [3.5]}, "hybrid": {"values": [4.5]}},
        "model_results": {"prediction": None},
        "interpretation": {"status": "ANSWERED", "answer": "Existing answer", "claims": evidence["claims"]},
        "spatial_evidence": evidence,
        "evidence": [{"kind": "feature statistics", "reference": "features.spectral.statistics"}],
        "confidence": {"model_confidence": None, "calibration_status": "uncalibrated"},
        "preprocessing": [{"operation": "alignment", "status": "completed"}],
        "execution_trace": [{"step": 1}],
        "warnings": [],
        "error": None,
        "temporal": {"status": "not requested"},
        "llm_explanation": "Existing deterministic explanation",
    }


def test_compatibility_adapter_preserves_science_evidence_and_interpretation():
    legacy = _legacy_result()
    before = copy.deepcopy(legacy)
    adapted = adapt_run_analysis_output(legacy, {"sample_id": "61_39"})
    assert legacy == before
    assert adapted.legacy_result is legacy
    assert adapted.scene.representations.physical_features is legacy["features"]["spectral"]
    assert adapted.scene.representations.croma_scene is legacy["features"]["deep"]
    assert adapted.scene.representations.joint is legacy["features"]["hybrid"]
    assert adapted.task_result.evidence is legacy["spatial_evidence"]
    assert adapted.task_result.output["interpretation"] is legacy["interpretation"]
    assert adapted.task_result.confidence is None
    assert adapted.scene.coregistration_status == "aligned"


def test_current_run_analysis_output_converts_successfully_without_api_changes():
    request = {"query": "optical", "files": {}}
    legacy = run_analysis(request)
    adapted = adapt_run_analysis_output(legacy, request)
    assert adapted.legacy_result is legacy
    assert adapted.task_result.status == "failure"
    assert adapted.scene.analysis_id == legacy["analysis_id"]
    assert "contracts" not in legacy


def test_representation_set_serializes_as_part_of_scene():
    scene = SceneBundle(scene_id="s", representations=RepresentationSet(physical_features={"value": 1}))
    assert scene.to_dict()["representations"]["physical_features"] == {"value": 1}
