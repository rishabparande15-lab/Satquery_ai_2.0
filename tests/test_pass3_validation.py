import numpy as np
import torch

from src.pass3_validation import _independent_metrics, _scene_target


def test_scene_target_reduces_tokens_but_preserves_class_distribution():
    counts = torch.zeros(225, 19, dtype=torch.int64)
    counts[:, 0] = 2
    counts[:, 1] = 1

    target = _scene_target(counts)

    assert target.shape == (19,)
    assert np.isclose(target[0], 2 / 3)
    assert np.isclose(target[1], 1 / 3)
    assert np.isclose(target.sum(), 1)


def test_independent_metrics_match_exact_predictions():
    truth = np.eye(19, dtype=np.float32)[:2]
    predictions = truth.copy()

    metrics = _independent_metrics(predictions, truth)

    assert metrics == {"mae_pp": 0.0, "rmse_pp": 0.0, "bias_pp": 0.0, "dominant_class_accuracy": 1.0}