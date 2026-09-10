import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.multilabel_metrics import compute_multilabel_metrics
from src.hybrid_fusion import FeatureModeFusion
from src.training_data import BIGEARTHNET_CLASSES, encode_labels, load_manifest, make_splits, save_manifest


def test_label_encoding_and_unknown_rejection():
    encoded = encode_labels(["Arable land", "Urban fabric"], BIGEARTHNET_CLASSES)
    assert encoded.shape == (19,)
    assert encoded.sum() == 2
    with pytest.raises(ValueError):
        encode_labels(["not-a-class"], BIGEARTHNET_CLASSES)


def test_reproducible_leak_free_splits(tmp_path: Path):
    ids = [f"sample_{i}" for i in range(20)]
    first = make_splits(ids, seed=11)
    second = make_splits(ids, seed=11)
    assert first.splits == second.splits
    assert not (set(first.splits["train"]) & set(first.splits["validation"]))
    assert set().union(*first.splits.values()) == set(ids)
    path = tmp_path / "splits.json"
    save_manifest(first, path)
    assert load_manifest(path).splits == first.splits


def test_multilabel_metrics():
    targets = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    probabilities = np.array([[0.9, 0.1, 0.8], [0.1, 0.8, 0.2]], dtype=np.float32)
    metrics = compute_multilabel_metrics(targets, probabilities)
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["per_class"]) == 3


def test_training_manifest_is_json_serializable():
    manifest = make_splits(["a", "b", "c"], seed=1)
    json.dumps(manifest.to_dict())


def test_three_sample_manifest_has_all_splits():
    manifest = make_splits(["a", "b", "c"], seed=1)
    assert all(len(manifest.splits[name]) == 1 for name in ("train", "validation", "test"))


def test_all_feature_modes_share_output_contract():
    gee = torch.randn(2, 62)
    croma = torch.randn(2, 2304)
    for mode in ("gee_only", "croma_only", "hybrid"):
        output = FeatureModeFusion(mode, 62, 2304)(gee, croma)
        assert output.shape == (2, 192)
