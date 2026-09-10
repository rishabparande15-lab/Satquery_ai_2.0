"""Availability-aware routing for SatQuery analysis tasks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSelection:
    selected_tools: list[str]
    required_inputs: list[str]
    required_models: list[str]
    unavailable_tools: list[dict[str, str]]
    execution_order: list[str]
    expected_outputs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_tools(plan: dict[str, Any], capabilities: dict[str, bool] | None = None) -> ToolSelection:
    capabilities = capabilities or {}
    modalities = set(plan.get("modalities", []))
    temporal = bool(plan.get("requires_temporal_pair"))
    selected = ["input_validation", "preprocessing", "multimodal_cube", "gee_physical_features"]
    required_inputs = list(modalities)
    models: list[str] = []
    unavailable: list[dict[str, str]] = []
    if modalities and not temporal:
        if capabilities.get("croma", False):
            selected += ["croma_" + "_".join(sorted(modalities))]
            models.append("official CROMA base")
            if len(modalities) == 2:
                selected.append("hybrid_fusion")
                models.append("untrained hybrid fusion representation")
        else:
            unavailable.append({"tool": "CROMA joint representation", "status": "unavailable", "reason": "Official CROMA source/checkpoint is not configured."})
    if temporal:
        selected = ["input_validation", "temporal_validation"]
        required_inputs += ["before image", "after image"]
    task = plan.get("task", "")
    if task in {"land_cover_classification", "scene_description", "single_image_vqa", "segmentation", "object_detection"}:
        unavailable.append({"tool": task, "status": "pending training", "reason": "No validated task head is installed; only input features are available."})
    if "water" in task:
        unavailable.append({"tool": "water_detection", "status": "pending training", "reason": "No trained water mask head is installed; no area prediction will be fabricated."})
    if task in {"change_detection", "water_change_analysis"}:
        unavailable.append({"tool": "change_detection", "status": "requires valid input", "reason": "A matching before-and-after pair is required."})
    selected += ["evidence", "confidence_assessment", "explanation", "report_generation"]
    return ToolSelection(selected, required_inputs, models, unavailable, selected, list(plan.get("expected_outputs", [])))
