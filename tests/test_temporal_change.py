from pathlib import Path

import pytest
from PIL import Image

from src.temporal_change import (TEMPORAL_ORDER, TemporalChangeDescriptionController,
                                 TemporalChangeInputError, validate_temporal_change_input)


def image(path: Path, color, size=(32, 32)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_pre_post_contract_and_generated_caption(tmp_path):
    t1, t2 = image(tmp_path / "t1.png", "black"), image(tmp_path / "t2.png", "white")
    controller = TemporalChangeDescriptionController(lambda before, after: "a building was added", model_provenance={"checkpoint": "fixture"})
    result = controller.run(t1_path=t1, t2_path=t2, query="What changed?",
                            metadata={"temporal_order": TEMPORAL_ORDER, "split": "validation", "pair_id": "val-1", "source_dataset": "LEVIR_CC"})
    assert result["route"] == "TEMPORAL_CHANGE_DESCRIPTION"
    assert result["change_description"] == "a building was added"
    assert result["temporal_order"] == "PRE_POST"
    assert result["provenance"]["source_dataset"] == "LEVIR_CC"


@pytest.mark.parametrize(("t1", "t2", "order", "code"), [
    (None, "b", "PRE_POST", "TEMPORAL_T1_MISSING"),
    ("a", None, "PRE_POST", "TEMPORAL_T2_MISSING"),
    ("a", "b", "POST_PRE", "INVALID_TEMPORAL_ORDER"),
])
def test_temporal_contract_fails_closed(tmp_path, t1, t2, order, code):
    first = image(tmp_path / "a.png", "black")
    second = image(tmp_path / "b.png", "white")
    paths = {"a": first, "b": second, None: None}
    with pytest.raises(TemporalChangeInputError) as exc:
        validate_temporal_change_input(t1_path=paths[t1], t2_path=paths[t2], temporal_order=order)
    assert exc.value.code == code


def test_identical_corrupt_and_mismatched_pairs_are_rejected(tmp_path):
    good = image(tmp_path / "good.png", "black")
    other_size = image(tmp_path / "other.png", "white", (16, 16))
    bad = tmp_path / "bad.png"; bad.write_bytes(b"not an image")
    for first, second, code in ((good, good, "IDENTICAL_TEMPORAL_INPUTS"), (good, other_size, "INCOMPATIBLE_TEMPORAL_DIMENSIONS"), (bad, good, "CORRUPT_TEMPORAL_T1")):
        with pytest.raises(TemporalChangeInputError) as exc:
            validate_temporal_change_input(t1_path=first, t2_path=second, temporal_order="PRE_POST")
        assert exc.value.code == code


def test_test_split_is_never_allowed(tmp_path):
    first, second = image(tmp_path / "a.png", "black"), image(tmp_path / "b.png", "white")
    with pytest.raises(TemporalChangeInputError, match="Only TRAIN"):
        validate_temporal_change_input(t1_path=first, t2_path=second, temporal_order="PRE_POST", split="test")
