import pytest

from src.region_grounding import Box, box_to_token_overlaps, box_to_token_indices


def test_full_image_box_covers_all_tokens():
    overlaps = box_to_token_overlaps(Box(0, 0, 120, 120))
    assert list(overlaps) == list(range(225))
    assert all(value == pytest.approx(1.0) for value in overlaps.values())


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        (Box(0, 0, 8, 8), [0]),
        (Box(112, 112, 120, 120), [224]),
        (Box(56, 56, 64, 64), [112]),
        (Box(1, 1, 2, 2), [0]),
        (Box(8, 0, 16, 8), [1]),
        (Box(-4, -4, 4, 4), [0]),
    ],
)
def test_box_to_token_indices(box, expected):
    assert box_to_token_indices(box) == expected


def test_partial_box_reports_area_overlap_and_row_major_order():
    overlaps = box_to_token_overlaps(Box(4, 4, 12, 12))
    assert list(overlaps) == [0, 1, 15, 16]
    assert overlaps == {0: pytest.approx(0.25), 1: pytest.approx(0.25), 15: pytest.approx(0.25), 16: pytest.approx(0.25)}


@pytest.mark.parametrize("box", [Box(5, 5, 5, 6), Box(1, 2, 0, 3), Box(121, 0, 122, 1)])
def test_invalid_or_nonoverlapping_boxes_are_rejected(box):
    with pytest.raises(ValueError):
        box_to_token_indices(box)
