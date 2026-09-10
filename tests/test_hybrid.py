import numpy as np
import pytest
import torch

from src.gee_features import FeatureExtractionError, extract_gee_features
from src.hybrid_fusion import ChangeAnalysisHead, HybridFusion, LandCoverClassificationHead
from src.temporal import TemporalPairError, compare_representations, validate_temporal_pair


BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def arrays():
    optical = np.arange(12 * 4 * 4, dtype=np.float32).reshape(12, 4, 4) + 1
    sar = np.stack([np.ones((4, 4), dtype=np.float32) * 2, np.ones((4, 4), dtype=np.float32)], axis=0)
    return optical, sar


def test_gee_features_and_indices():
    optical, sar = arrays()
    vector, report = extract_gee_features(optical, sar, BANDS, ["VV", "VH"])
    assert vector.shape == (62,)
    assert report["dimension"] == 62
    assert np.isfinite(vector).all()
    assert report["valid_pixel_counts"]["index_NDVI_mean"] == 16


def test_fusion_shape():
    fusion = HybridFusion(62, 2304)
    result = fusion(torch.randn(2, 62), torch.randn(2, 2304))
    assert result.shape == (2, 192)


def test_invalid_grid_rejected():
    optical, sar = arrays()
    with pytest.raises(FeatureExtractionError):
        extract_gee_features(optical[:, :3], sar, BANDS, ["VV", "VH"])


def test_missing_and_duplicate_bands_rejected():
    optical, sar = arrays()
    with pytest.raises(FeatureExtractionError):
        extract_gee_features(optical, sar, BANDS[:-1], ["VV", "VH"])
    with pytest.raises(FeatureExtractionError):
        extract_gee_features(optical, sar, BANDS[:-1] + ["B03"], ["VV", "VH"])


def test_fusion_is_deterministic_with_seed_and_dynamic_output():
    torch.manual_seed(7)
    first = HybridFusion(62, 2304, output_dim=80)
    torch.manual_seed(7)
    second = HybridFusion(62, 2304, output_dim=80)
    inputs = (torch.randn(2, 62), torch.randn(2, 2304))
    assert torch.equal(first(*inputs), second(*inputs))
    assert first.output_dim == 80


def test_training_and_change_interfaces():
    head = LandCoverClassificationHead(192, 6)
    logits = head(torch.randn(2, 192))
    assert logits.shape == (2, 6)
    assert torch.all((head.probabilities(torch.randn(2, 192)) >= 0) & (head.probabilities(torch.randn(2, 192)) <= 1))
    assert LandCoverClassificationHead.loss(logits, torch.zeros(2, 6)).ndim == 0
    change = ChangeAnalysisHead()(torch.zeros(1, 4), torch.ones(1, 4))
    assert torch.equal(change, torch.ones(1, 4))


def test_temporal_pair_rejects_unrelated_samples():
    with pytest.raises(TemporalPairError):
        validate_temporal_pair({"sample_id": "61_39"}, {"sample_id": "61_40"})
    result = compare_representations(np.zeros(4), np.ones(4))
    assert result["dimension"] == 4
