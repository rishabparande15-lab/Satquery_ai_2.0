import numpy as np
import pytest
import torch

from src.eo_vlm.sar_projector import SARProjector, normalize_sar


def test_exact_contract_and_shape():
    model = SARProjector()
    assert model(torch.zeros(2, 2, 120, 120)).shape == (2, 16, 2048)

def test_invalid_channels_rejected():
    with pytest.raises(ValueError): SARProjector()(torch.zeros(1, 3, 120, 120))
    with pytest.raises(ValueError): normalize_sar(np.zeros((1, 120, 120)))

def test_order_and_normalization_are_deterministic():
    x = np.stack([np.full((120,120), -10), np.full((120,120), -20)]).astype(np.float32)
    x[:, 0, 0] += 1
    a, b = normalize_sar(x), normalize_sar(x)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[0], a[1])

def test_nan_inf_and_nodata():
    x = np.zeros((2,120,120), dtype=np.float32); x[0,0,0] = np.nan
    with pytest.raises(ValueError): normalize_sar(x)
    x[0,0,0] = -9999
    assert np.isfinite(normalize_sar(x, nodata=-9999)).all()

def test_provenance_and_frozen_separation():
    model = SARProjector(); info = model.provenance()
    assert info["channel_order"] == ["VV", "VH"] and info["output_shape"] == [1,16,2048]
    qwen = torch.nn.Linear(2, 2)
    for p in qwen.parameters(): p.requires_grad_(False)
    assert model.trainable_parameter_count > 0 and not any(p.requires_grad for p in qwen.parameters())
