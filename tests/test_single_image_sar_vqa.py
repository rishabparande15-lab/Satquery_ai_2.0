import numpy as np
import pytest

from src.satquery_agent import classify_input_configuration, select_route
from src.single_image_sar_vqa import validate_single_sar_input


def test_explicit_declared_s1_sar_selects_the_dedicated_route():
    config = classify_input_configuration([{"input_id": "s1", "role": "SINGLE", "sensor": "Sentinel-1", "modality": "SAR"}])
    assert select_route(config, "VQA")[0] == "SINGLE_IMAGE_SAR_VQA"


def test_sar_contract_rejects_wrong_order_and_nonfinite_values():
    base = dict(image_id="s1", split="validation", metadata={"crs": "EPSG:32632", "sar_band_order": ["VV", "VH"]}, question="Is water present?", task_type="binary_qa")
    validate_single_sar_input(sar=np.zeros((2, 120, 120), dtype=np.float32), **base)
    with pytest.raises(ValueError, match="non-finite"):
        validate_single_sar_input(sar=np.full((2, 120, 120), np.nan, dtype=np.float32), **base)
    with pytest.raises(ValueError, match="VV/VH"):
        validate_single_sar_input(sar=np.zeros((2, 120, 120), dtype=np.float32), **{**base, "metadata": {"crs": "EPSG:32632", "sar_band_order": ["VH", "VV"]}})
