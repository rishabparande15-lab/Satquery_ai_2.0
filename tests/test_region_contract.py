from __future__ import annotations

import pytest

from src.region_contract import DETERMINISTIC_REGION, SOURCE_ANNOTATION_REGION, Region, source_region
from src.spatial_contract import BBox, CoordinateSpace


def test_source_region_preserves_geometry_and_provenance():
    region = source_region("r1", "area-a", "train", BBox(0, 0, 8, 8, CoordinateSpace.ANALYSIS_GRID), provenance={"source": "annotation"})
    assert region.source == SOURCE_ANNOTATION_REGION
    assert region.to_dict()["geometry"]["coordinate_space"] == "ANALYSIS_GRID"


def test_deterministic_region_is_allowed_and_learned_region_is_not():
    region = Region("r2", "area-a", "train", CoordinateSpace.CROMA_TOKEN,
                    {"token_index": 0}, DETERMINISTIC_REGION, (0,), (), {"mapping": "v1"})
    assert region.linked_tokens == (0,)
    with pytest.raises(ValueError, match="reserved"):
        Region("r3", "area-a", "train", CoordinateSpace.CROMA_TOKEN,
               {"token_index": 0}, "LEARNED_REGION", (0,), (), {})
