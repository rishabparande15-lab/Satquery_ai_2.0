"""Integration gates for real, already-validated controller components.

The Qwen gate is opt-in because it loads the local 3B CUDA checkpoint. It is
never a unit-test substitute or a claim about blocked task types.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.agent.controller import AgentController
from src.agent.real_tools import FrozenScientificCacheTool, QwenRGBTool, S2ProjectorTool, TemporalRGBTool
from src.agent.query_types import AnalysisRequest, ExecutionStatus, TaskType
from src.eo_vlm.temporal_contracts import (SpatialCorrespondence, SpatialCorrespondenceStatus, TemporalFrame,
    TemporalInput, TemporalMetadata, TemporalModality, TemporalOrderStatus, TemporalProvenance)

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(r"D:\Satquery_ai datasets\comparison\pipeline3-5000\scene_features")
CHECKPOINT = Path(r"D:\Satquery_ai datasets\comparison\pipeline3-5000\models\baseline\hybrid.pt")
PROJECTOR = ROOT / "artifacts/phase3g1/stage_c_500_100/s2_multispectral_projector.pt"
PROJECTOR_RECEIPT = PROJECTOR.with_suffix(".json")
QWEN = Path(r"C:\Users\Rishab\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots\66285546d2b821cf421d4f5eb2576359d3770cd3")


def test_real_scientific_cache_probe_flow():
    if not (CACHE / "manifest.json").is_file() or not CHECKPOINT.is_file():
        pytest.skip("validated scientific cache/checkpoint unavailable")
    from scripts.run_pipeline3_5000_baseline import load_cache
    sample_id = str(load_cache(CACHE, include_test=False)["area_ids"][0])
    tool = FrozenScientificCacheTool(CACHE, CHECKPOINT)
    result = AgentController({tool.name: tool}).analyze(AnalysisRequest(
        "What land-cover characteristics are present in this Sentinel-2 image?", ({"id": sample_id, "modality": "s2"},)))
    assert result.capability_status == ExecutionStatus.EXECUTED
    assert result.plan.understanding.task_type == TaskType.SCIENTIFIC_ANALYSIS
    assert result.evidence[0].evidence_type.value == "SCIENTIFIC_PREDICTION"
    assert result.evidence[0].provenance["representation"] == "hybrid_830d"


def test_real_s2_learned_projector_flow():
    cube = np.stack([np.full((120, 120), index + 1, dtype=np.float32) for index in range(12)])
    tool = S2ProjectorTool(PROJECTOR, PROJECTOR_RECEIPT)
    result = AgentController({tool.name: tool}).analyze(AnalysisRequest(
        "Describe the observable land-cover characteristics in this Sentinel-2 image.", ({"id": "real-s2-contract-sample", "modality": "s2", "array": cube},)))
    assert result.capability_status == ExecutionStatus.EXECUTED
    assert result.plan.understanding.task_type == TaskType.S2_CAPTIONING
    assert result.tool_results[0].output["token_shape"] == [1, 16, 2048]
    assert result.evidence[0].representation == "learned_s2_representation"
    assert result.evidence[0].evidence_type.value == "METADATA_EVIDENCE"


def _temporal_payload() -> dict:
    from PIL import Image
    fixture = ROOT / "artifacts/test_fixtures/temporal_change_fixture_manifest.json"
    record = json.loads(fixture.read_text(encoding="utf-8"))["records"][0]
    root = fixture.parent / "rsrcc"
    t1 = np.asarray(Image.open(root / record["t1"]["path"]).convert("RGB"))
    t2 = np.asarray(Image.open(root / record["t2"]["path"]).convert("RGB"))
    source = TemporalProvenance("google/RSRCC", source_revision="7898de7bfd08bc404d9a92e1caaa9dce91b0c3ea", attributes={"registration_status": "UNKNOWN", "split": "val"})
    temporal = TemporalInput(TemporalFrame(record["pair_key"] + "_before", TemporalModality.OPTICAL, source, tensor_shape=tuple(t1.shape)),
        TemporalFrame(record["pair_key"] + "_after", TemporalModality.OPTICAL, source, tensor_shape=tuple(t2.shape)),
        TemporalOrderStatus.VERIFIED, SpatialCorrespondenceStatus.VERIFIED, SpatialCorrespondence.SPATIALLY_CORRESPONDING, TemporalMetadata(), source)
    return {"id": record["pair_key"], "modality": "optical_rgb_pair", "temporal_input": temporal, "t1": t1, "t2": t2}


@pytest.mark.skipif(os.environ.get("SATQUERY_RUN_REAL_QWEN") != "1", reason="set SATQUERY_RUN_REAL_QWEN=1 for the local CUDA execution gate")
def test_real_qwen_rgb_and_temporal_rgb_controller_flows():
    from PIL import Image
    image = np.asarray(Image.open(ROOT / "artifacts/test_fixtures/rsrcc/val/images/000/d259f34c_8693_418e_9dab_ef14b860319f_before.png").convert("RGB"))
    qwen = QwenRGBTool(QWEN, max_new_tokens=8)
    temporal = TemporalRGBTool(qwen)
    controller = AgentController({qwen.name: qwen, temporal.name: temporal})
    rgb = controller.analyze(AnalysisRequest("Describe this image.", ({"id": "rsrcc-before", "modality": "rgb", "array": image},)))
    pair = controller.analyze(AnalysisRequest("Run the validated temporal RGB pair execution.", (_temporal_payload(),), {"execution_mode": "temporal_rgb"}))
    assert rgb.plan.understanding.task_type == TaskType.RGB_VISUAL_REASONING
    assert rgb.evidence[0].evidence_type.value == "MODEL_LANGUAGE_OUTPUT"
    assert pair.plan.understanding.task_type == TaskType.TEMPORAL_RGB_EXECUTION
    assert pair.evidence[0].evidence_type.value == "MODEL_LANGUAGE_OUTPUT"
    assert pair.evidence[0].provenance["timestamp_status"] == "UNKNOWN"
    assert pair.evidence[0].provenance["learned_temporal_fusion"] is False


def test_real_blocked_flows_stay_blocked_without_tools():
    controller = AgentController()
    cases = (("What objects are present in this SAR image?", "s1", TaskType.SAR_VQA, ExecutionStatus.BLOCKED),
             ("Using the optical and SAR images together, describe the scene.", "optical_sar", TaskType.OPTICAL_SAR_REASONING, ExecutionStatus.BLOCKED),
             ("What changed between these two images?", "rgb", TaskType.TEMPORAL_CHANGE_DESCRIPTION, ExecutionStatus.NOT_VERIFIED),
             ("Where is the river?", "rgb", TaskType.GROUNDING, ExecutionStatus.BLOCKED))
    for query, modality, task, expected_status in cases:
        inputs = ({"id": "a", "modality": modality}, {"id": "b", "modality": modality}) if task == TaskType.TEMPORAL_CHANGE_DESCRIPTION else ({"id": "a", "modality": modality},)
        result = controller.analyze(AnalysisRequest(query, inputs))
        assert result.plan.understanding.task_type == task
        assert result.capability_status == expected_status
        assert "Fallback: NONE" in result.final_answer.text
