"""Task-specific leakage policies and deterministic audit helpers.

This module reports risks and raises only when an authoritative grouping is
structurally split. It never removes records or silently repairs assignments.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "leakage_policy_v1"


class TaskFamily(str, Enum):
    SCENE_CLASSIFICATION = "scene_classification"
    IMAGE_VQA = "image_vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    TEMPORAL_CHANGE = "temporal_change"
    OPTICAL_SAR = "optical_sar"


@dataclass(frozen=True)
class LeakagePolicy:
    task: TaskFamily
    grouping_unit: str
    required_groups: tuple[str, ...]
    content_overlap_reported: tuple[str, ...]
    feature_provenance_required: bool = True


POLICIES = {
    TaskFamily.SCENE_CLASSIFICATION: LeakagePolicy(
        TaskFamily.SCENE_CLASSIFICATION, "image", ("image_id",), (), True),
    TaskFamily.IMAGE_VQA: LeakagePolicy(
        TaskFamily.IMAGE_VQA, "image", ("image_id",), ("question", "question_answer"), True),
    TaskFamily.CAPTIONING: LeakagePolicy(
        TaskFamily.CAPTIONING, "image", ("image_id",), ("caption",), True),
    TaskFamily.GROUNDING: LeakagePolicy(
        TaskFamily.GROUNDING, "image", ("image_id",), ("image_id",), True),
    TaskFamily.TEMPORAL_CHANGE: LeakagePolicy(
        TaskFamily.TEMPORAL_CHANGE, "temporal_scene", ("temporal_scene_id", "image_id"), (), True),
    TaskFamily.OPTICAL_SAR: LeakagePolicy(
        TaskFamily.OPTICAL_SAR, "optical_sar_pair", ("optical_identity", "sar_identity"), (), True),
}


def policy_for(task: TaskFamily | str) -> LeakagePolicy:
    try:
        return POLICIES[TaskFamily(task)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unsupported leakage-policy task: {task!r}") from error


def audit_leakage(records: Iterable[Mapping[str, Any]], task: TaskFamily | str) -> dict[str, Any]:
    """Audit split grouping and content overlap without changing records."""
    policy = policy_for(task)
    rows = [dict(record) for record in records]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": policy.task.value,
        "grouping_unit": policy.grouping_unit,
        "record_count": len(rows),
        "split_group_conflicts": {},
        "image_level_leakage": 0,
        "pair_level_leakage": 0,
        "temporal_scene_leakage": 0,
        "region_grouping_is_image_level": policy.task in {TaskFamily.GROUNDING, TaskFamily.IMAGE_VQA, TaskFamily.CAPTIONING},
        "missing_feature_provenance": 0,
        "content_overlap": {},
    }
    for field in policy.required_groups:
        grouped: dict[Any, set[str]] = defaultdict(set)
        for row in rows:
            value = row.get(field)
            if value is not None:
                grouped[value].add(str(row.get("split")))
        conflicts = {str(key): sorted(splits) for key, splits in grouped.items() if len(splits - {"None"}) > 1}
        result["split_group_conflicts"][field] = conflicts
    result["image_level_leakage"] = len(result["split_group_conflicts"].get("image_id", {}))
    result["pair_level_leakage"] = max(
        len(result["split_group_conflicts"].get("optical_identity", {})),
        len(result["split_group_conflicts"].get("sar_identity", {})),
    ) if policy.task is TaskFamily.OPTICAL_SAR else 0
    result["temporal_scene_leakage"] = len(result["split_group_conflicts"].get("temporal_scene_id", {}))
    if policy.feature_provenance_required:
        result["missing_feature_provenance"] = sum(
            row.get("feature_provenance") is None and row.get("features") is not None for row in rows)
    if "question" in policy.content_overlap_reported:
        result["content_overlap"]["question"] = _overlap_count(rows, lambda row: row.get("question"))
    if "question_answer" in policy.content_overlap_reported:
        result["content_overlap"]["question_answer"] = _overlap_count(
            rows, lambda row: (row.get("question"), row.get("answer")))
    if "caption" in policy.content_overlap_reported:
        result["content_overlap"]["caption"] = _overlap_count(rows, lambda row: row.get("caption"))
    return result


def assert_split_safe(records: Iterable[Mapping[str, Any]], task: TaskFamily | str) -> dict[str, Any]:
    result = audit_leakage(records, task)
    conflicts = {field: values for field, values in result["split_group_conflicts"].items() if values}
    if conflicts:
        raise ValueError(f"Split leakage detected for {policy_for(task).task.value}: {conflicts}")
    return result


def _overlap_count(records: list[dict[str, Any]], key_fn) -> int:
    splits_by_value: dict[Any, set[str]] = defaultdict(set)
    for record in records:
        value = key_fn(record)
        if value not in (None, "", (None, None)):
            splits_by_value[value].add(str(record.get("split")))
    return sum(len(splits) > 1 for splits in splits_by_value.values())
