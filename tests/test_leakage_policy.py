from __future__ import annotations

import pytest

from src.leakage_policy import TaskFamily, assert_split_safe, audit_leakage, policy_for


def test_task_policies_use_task_specific_grouping():
    assert policy_for(TaskFamily.IMAGE_VQA).grouping_unit == "image"
    assert policy_for(TaskFamily.TEMPORAL_CHANGE).grouping_unit == "temporal_scene"
    assert policy_for(TaskFamily.OPTICAL_SAR).grouping_unit == "optical_sar_pair"


def test_vqa_reports_text_overlap_without_calling_it_image_leakage():
    records = [
        {"image_id": "a", "split": "train", "question": "Is water present?", "answer": "yes"},
        {"image_id": "b", "split": "test", "question": "Is water present?", "answer": "yes"},
    ]
    audit = audit_leakage(records, TaskFamily.IMAGE_VQA)
    assert audit["image_level_leakage"] == 0
    assert audit["content_overlap"]["question_answer"] == 1
    assert audit["content_overlap"]["question"] == 1


def test_image_and_pair_split_conflicts_fail_closed():
    records = [
        {"image_id": "same", "split": "train", "optical_identity": "o", "sar_identity": "s"},
        {"image_id": "same", "split": "test", "optical_identity": "o", "sar_identity": "s"},
    ]
    image_audit = audit_leakage(records, TaskFamily.GROUNDING)
    assert image_audit["image_level_leakage"] == 1
    with pytest.raises(ValueError, match="Split leakage"):
        assert_split_safe(records, TaskFamily.GROUNDING)
    pair_audit = audit_leakage(records, TaskFamily.OPTICAL_SAR)
    assert pair_audit["pair_level_leakage"] == 1


def test_temporal_scene_grouping_and_feature_provenance():
    records = [
        {"image_id": "a", "temporal_scene_id": "scene-1", "split": "train", "features": {}, "feature_provenance": None},
        {"image_id": "b", "temporal_scene_id": "scene-1", "split": "test", "features": {}, "feature_provenance": None},
    ]
    audit = audit_leakage(records, TaskFamily.TEMPORAL_CHANGE)
    assert audit["temporal_scene_leakage"] == 1
    assert audit["missing_feature_provenance"] == 2
