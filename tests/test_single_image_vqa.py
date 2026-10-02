import numpy as np
import pytest

from src.single_image_vqa import (classify_single_image_route, parse_vqa_response,
                                  validate_single_image_input)


def metadata():
    return {"shape": [120, 120], "crs": "EPSG:32633", "resolution": [10, 10], "bounds": [0, 0, 1200, 1200]}


def test_single_image_contract_accepts_one_canonical_s2_image():
    validate_single_image_input(
        image_id="patch-1", split="train", optical=np.zeros((12, 120, 120), dtype=np.float32),
        metadata=metadata(), question="Is there water?", task_type="binary_qa",
    )


@pytest.mark.parametrize("split", ["test", "unknown"])
def test_single_image_contract_rejects_test_or_unknown_split(split):
    with pytest.raises(ValueError, match="external inference inputs"):
        validate_single_image_input(
            image_id="patch-1", split=split, optical=np.zeros((12, 120, 120), dtype=np.float32),
            metadata=metadata(), question="Is there water?", task_type="binary_qa",
        )


@pytest.mark.parametrize("shape", [(2, 120, 120), (12, 119, 120)])
def test_single_image_contract_rejects_noncanonical_shape(shape):
    with pytest.raises(ValueError, match="canonical shape"):
        validate_single_image_input(
            image_id="patch-1", split="train", optical=np.zeros(shape, dtype=np.float32),
            metadata=metadata(), question="Is there water?", task_type="binary_qa",
        )


def test_single_image_contract_rejects_missing_spatial_metadata():
    with pytest.raises(ValueError, match="CRS"):
        validate_single_image_input(
            image_id="patch-1", split="train", optical=np.zeros((12, 120, 120), dtype=np.float32),
            metadata={"shape": [120, 120]}, question="Is there water?", task_type="binary_qa",
        )


def test_vqa_parsers_match_supported_answer_formats():
    assert parse_vqa_response("binary_qa", " Yes, visible") == "yes"
    assert parse_vqa_response("binary_qa", "No") == "no"
    assert parse_vqa_response("multiple_choice_qa", " (B).", [{"key": "a"}, {"key": "b"}]) == "b"
    assert parse_vqa_response("multiple_choice_qa", "D", [{"key": "a"}, {"key": "b"}]) is None
    assert parse_vqa_response("binary_qa", "maybe") is None


def test_representative_query_routes_are_explicit_and_temporal_is_blocked():
    assert classify_single_image_route("What land cover is visible?") == "SINGLE_IMAGE_VQA"
    assert classify_single_image_route("Highlight the water body") == "SINGLE_IMAGE_GROUNDING"
    assert classify_single_image_route("Use the SAR and optical image together") == "OPTICAL_SAR_ANALYSIS"
    assert classify_single_image_route("What changed before and after?") == "TEMPORAL_ROUTE_NOT_IMPLEMENTED"