import numpy as np
import pytest
import torch

from src.phase1_foundation import SampleIdentity, prepare_batch
from src.phase2_integration import adapt_phase1_output


def make_identity(name="sample", split="train"):
    return SampleIdentity(name, f"s1_{name}", f"s2_{name}", split, {"date": "2026-01-01"},
                          "EPSG:32632", (0, 0, 1200, 1200), (10, 10), {"source": "test"})


def make_features(count):
    spatial = torch.arange(count * 225 * 768, dtype=torch.float32).reshape(count, 225, 768)
    pooled = spatial[:, 0, :].clone()
    return {"optical_encodings": spatial, "SAR_encodings": spatial + 1, "joint_encodings": spatial + 2,
            "optical_GAP": pooled, "SAR_GAP": pooled + 1, "joint_GAP": pooled + 2}


def make_input(count=1, split="train"):
    identities = [make_identity(f"sample-{index}", split) for index in range(count)]
    batch = prepare_batch(torch.ones(count, 12, 120, 120), torch.ones(count, 2, 120, 120), identities)
    reports = [{"sample_id": identity.patch_id, "schema": "legacy_joint_62", "dimension": 2, "feature_names": ["x", "y"]} for identity in identities]
    return batch, adapt_phase1_output(batch, make_features(count), np.ones((count, 2), dtype=np.float32), reports,
                                     croma_provenance={"checkpoint": "croma.pt", "checkpoint_sha256": "abc"},
                                     experiment_configuration={"seed": 42})


def test_adapter_preserves_modalities_order_and_existing_pooled_contract():
    batch, adapted = make_input(2)
    assert adapted.optical_encodings.shape == (2, 225, 768)
    assert adapted.sar_encodings.shape == (2, 225, 768)
    assert adapted.joint_encodings.shape == (2, 225, 768)
    assert adapted.pooled_croma_features.shape == (2, 2304)
    assert adapted.pooled_croma_features[0, 768] == 1
    assert adapted.identities == batch.identities
    assert adapted.provenance[0]["identity"]["patch_id"] == "sample-0"
    assert adapted.provenance[0]["channel_order"]["sar"] == ["VV", "VH"]


def test_adapter_rejects_identity_mismatch_and_split_mixing():
    batch, _ = make_input()
    mismatched = make_features(1)
    with pytest.raises(ValueError, match="align"):
        adapt_phase1_output(batch, mismatched, np.ones((1, 2), dtype=np.float32),
                            [{"dimension": 2, "feature_names": ["x", "y"]}] * 2,
                            croma_provenance={"checkpoint": "x", "checkpoint_sha256": "y"})
    mixed = prepare_batch(torch.ones(2, 12, 120, 120), torch.ones(2, 2, 120, 120),
                          [make_identity("a", "train"), make_identity("b", "test")])
    with pytest.raises(ValueError, match="mix"):
        adapt_phase1_output(mixed, make_features(2), np.ones((2, 2), dtype=np.float32),
                            [{"dimension": 2, "feature_names": ["x", "y"]}] * 2,
                            croma_provenance={"checkpoint": "x", "checkpoint_sha256": "y"})
    reports = [{"sample_id": "wrong", "dimension": 2, "feature_names": ["x", "y"]}]
    with pytest.raises(ValueError, match="sample identity"):
        adapt_phase1_output(batch, mismatched, np.ones((1, 2), dtype=np.float32), reports,
                            croma_provenance={"checkpoint": "x", "checkpoint_sha256": "y"})


def test_adapter_rejects_invalid_shapes_and_reports_optional_features():
    batch, _ = make_input()
    features = make_features(1)
    features["joint_encodings"] = torch.zeros(1, 224, 768)
    with pytest.raises(ValueError, match="Invalid CROMA"):
        adapt_phase1_output(batch, features, np.ones((1, 2), dtype=np.float32),
                            [{"dimension": 2, "feature_names": ["x", "y"]}],
                            croma_provenance={"checkpoint": "x", "checkpoint_sha256": "y"})
    _, adapted = make_input()
    assert adapted.missing_optional_features == ()
    reports = [{"sample_id": "sample-0", "dimension": 2, "feature_names": ["x", "y"], "schema": "legacy_joint_62"}]
    adapted = adapt_phase1_output(batch, make_features(1), np.ones((1, 2), dtype=np.float32), reports,
                                  croma_provenance={"checkpoint": "x", "checkpoint_sha256": "y"},
                                  missing_optional_features=["soil_moisture"])
    assert adapted.missing_optional_features == ("soil_moisture",)


def test_real_sample_reaches_adapter_when_fixture_is_available():
    from src.config import get_settings
    from src.dataset_loader import discover_samples, load_sample

    root = get_settings().dataset_root
    if not root.exists():
        pytest.skip("external BigEarthNet fixture is unavailable")
    sample = discover_samples(root)[0]
    loaded = load_sample(sample)
    identity = SampleIdentity(sample.patch_id, sample.sar_paths["VV"].stem, sample.optical_paths["B02"].stem,
                              "test", {"source": "BigEarthNet fixture"}, loaded.metadata["crs"],
                              tuple(loaded.metadata["bounds"]), tuple(loaded.metadata["resolution"]), {"source": str(root)})
    batch = prepare_batch(torch.from_numpy(loaded.raw_optical[None]), torch.from_numpy(loaded.raw_sar[None]), [identity])
    _, adapted = make_input()
    adapted = adapt_phase1_output(batch, make_features(1), np.ones((1, 2), dtype=np.float32),
                                  [{"sample_id": identity.patch_id, "dimension": 2, "feature_names": ["x", "y"]}],
                                  croma_provenance={"checkpoint": "x", "checkpoint_sha256": "y"})
    assert adapted.identities[0].patch_id == sample.patch_id