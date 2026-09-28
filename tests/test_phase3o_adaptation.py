import pytest

from src.eo_vlm.adaptation import (MODE_RGB, MODE_S2_PROJECTED, TrainingRecord, checkpoint_manifest,
                                   load_safe_records, record_from_sample, validate_record_partitions)


def sample(split="train", task="VQA_BINARY"):
    return {"sample_id": "ils:a", "image_id": "area-a", "split": split, "task_type": task, "validation_status": "VALID",
            "question": None if task == "CAPTIONING" else "Is this present?", "answer": None if task == "CAPTIONING" else "yes",
            "caption": "A scene." if task == "CAPTIONING" else None, "source_provenance": {"source_revision": "revision-1"}}


def raw(): return {"raw_source_sha256": "a" * 64}


def test_s2_record_contains_required_traceability():
    record = record_from_sample(sample(), mode=MODE_S2_PROJECTED, raw_representation=raw())
    assert record.provenance["scientific_representation_used"] is False
    assert record.to_dict()["input_representation"] == MODE_S2_PROJECTED


def test_rgb_record_has_explicit_boundary():
    assert record_from_sample(sample(), mode=MODE_RGB, raw_representation=raw()).provenance["representation"] == "rgb_image_to_qwen"


@pytest.mark.parametrize("task", ["GROUNDING_POINT", "GROUNDING_TEXT_BOX", "TEMPORAL_VQA"])
def test_unsafe_tasks_are_rejected(task):
    with pytest.raises(ValueError): record_from_sample(sample(task=task), mode=MODE_S2_PROJECTED, raw_representation=raw())


def test_test_split_is_rejected():
    with pytest.raises(ValueError): record_from_sample(sample(split="test"), mode=MODE_S2_PROJECTED, raw_representation=raw())


def test_caption_has_caption_only_supervision():
    record = record_from_sample(sample(task="CAPTIONING"), mode=MODE_S2_PROJECTED, raw_representation=raw())
    assert record.caption == "A scene."


def test_scientific_representation_is_rejected():
    record = record_from_sample(sample(), mode=MODE_S2_PROJECTED, raw_representation=raw())
    object.__setattr__(record, "provenance", {**record.provenance, "scientific_representation_used": True})
    with pytest.raises(ValueError): record.validate()


def test_partition_overlap_is_rejected():
    train = record_from_sample(sample(), mode=MODE_S2_PROJECTED, raw_representation=raw())
    validation = record_from_sample(sample(split="validation"), mode=MODE_S2_PROJECTED, raw_representation=raw())
    with pytest.raises(ValueError, match="leakage"): validate_record_partitions({"train": [train], "validation": [validation]})


def test_load_requires_local_assets(tmp_path):
    path = tmp_path / "samples.jsonl"; path.write_text(__import__("json").dumps(sample()) + "\n")
    with pytest.raises(RuntimeError, match="INSUFFICIENT"): load_safe_records(path, mode=MODE_S2_PROJECTED, raw_sources={}, limits={"train": 1, "validation": 1})


def test_checkpoint_manifest_never_claims_full_model(tmp_path):
    receipt = checkpoint_manifest(checkpoint_path=None, provenance={"model": "qwen"}, status="BLOCKED")
    assert receipt["base_model_weights_saved"] is False and receipt["checkpoint_sha256"] is None


def test_invalid_license_is_rejected():
    record = record_from_sample(sample(), mode=MODE_S2_PROJECTED, raw_representation=raw())
    object.__setattr__(record, "license_status", "UNKNOWN")
    with pytest.raises(ValueError): record.validate()


def test_invalid_augmentation_is_rejected():
    record = record_from_sample(sample(), mode=MODE_S2_PROJECTED, raw_representation=raw())
    object.__setattr__(record, "augmentation_policy", "random")
    with pytest.raises(ValueError): record.validate()


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError): record_from_sample(sample(), mode="MODE_SAR", raw_representation=raw())


def test_bad_s2_band_order_is_rejected():
    record = record_from_sample(sample(), mode=MODE_S2_PROJECTED, raw_representation=raw())
    object.__setattr__(record, "provenance", {**record.provenance, "canonical_band_order": []})
    with pytest.raises(ValueError, match="band order"): record.validate()


def test_caption_cannot_carry_vqa_fields():
    record = record_from_sample(sample(task="CAPTIONING"), mode=MODE_S2_PROJECTED, raw_representation=raw())
    object.__setattr__(record, "answer", "invented")
    with pytest.raises(ValueError, match="caption"): record.validate()


def test_vqa_cannot_carry_caption_field():
    record = record_from_sample(sample(), mode=MODE_S2_PROJECTED, raw_representation=raw())
    object.__setattr__(record, "caption", "invented")
    with pytest.raises(ValueError, match="VQA"): record.validate()


def test_empty_identity_is_rejected():
    record = record_from_sample(sample(), mode=MODE_S2_PROJECTED, raw_representation=raw())
    object.__setattr__(record, "sample_id", "")
    with pytest.raises(ValueError, match="sample_id"): record.validate()
