from PIL import Image

import pytest

from src.api import build_temporal_change_request
from src.temporal_change import TemporalChangeInputError


def make_pair(tmp_path):
    t1, t2 = tmp_path / "before.png", tmp_path / "after.png"
    Image.new("RGB", (20, 20), "black").save(t1); Image.new("RGB", (20, 20), "white").save(t2)
    return t1, t2


def test_temporal_api_builder_preserves_pre_post_identity(tmp_path):
    t1, t2 = make_pair(tmp_path)
    result = build_temporal_change_request({"query": "Describe changes", "temporal_order": "PRE_POST", "split": "validation", "pair_id": "lev-1"}, t1_path=t1, t2_path=t2)
    assert result["route"] == "TEMPORAL_CHANGE_DESCRIPTION"
    assert result["metadata"]["temporal_order"] == "PRE_POST"
    assert result["t1_identity"]["filename"] == "before.png"


def test_temporal_api_builder_rejects_test_and_order(tmp_path):
    t1, t2 = make_pair(tmp_path)
    for payload in ({"temporal_order": "POST_PRE"}, {"temporal_order": "PRE_POST", "split": "test"}):
        with pytest.raises(TemporalChangeInputError):
            build_temporal_change_request(payload, t1_path=t1, t2_path=t2)
