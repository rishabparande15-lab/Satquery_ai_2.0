import numpy as np
import pytest
import torch
from types import SimpleNamespace

from src.phase3_5_end_to_end import write_artifacts
from src.scientific_artifacts import save_artifact_bundle, verify_artifact_bundle


def test_artifact_bundle_round_trips_arrays_and_provenance(tmp_path):
    arrays = {
        "raw_optical": np.zeros((12, 120, 120), dtype=np.float32),
        "raw_sar": np.ones((2, 120, 120), dtype=np.float32),
        "physical_features": np.arange(62, dtype=np.float32),
        "croma_joint_encodings": np.zeros((225, 768), dtype=np.float32),
        "croma_joint_GAP": np.zeros(768, dtype=np.float32),
        "pooled_croma_features": np.zeros(2304, dtype=np.float32),
        "hybrid_features": np.zeros(192, dtype=np.float32),
    }
    receipt = save_artifact_bundle(
        tmp_path / "sample-1",
        sample_id="sample-1",
        arrays=arrays,
        provenance={"channel_order": {"optical": ["B01"], "sar": ["VV"]}},
    )

    verified = verify_artifact_bundle(tmp_path / "sample-1")

    assert receipt["status"] == "complete"
    assert verified["status"] == "verified"
    assert verified["sample_id"] == "sample-1"
    assert verified["arrays"]["physical_features"]["shape"] == [62]


def test_artifact_bundle_rejects_tampering(tmp_path):
    root = tmp_path / "sample-1"
    save_artifact_bundle(root, sample_id="sample-1", arrays={"physical_features": np.zeros(2, dtype=np.float32)}, provenance={})
    np.save(root / "physical_features.npy", np.ones(2, dtype=np.float32))

    with pytest.raises(ValueError, match="hash"):
        verify_artifact_bundle(root)


def test_phase3_5_writes_complete_single_sample_bundle(tmp_path):
    croma = {
        "optical_encodings": torch.zeros(1, 225, 768),
        "SAR_encodings": torch.zeros(1, 225, 768),
        "joint_encodings": torch.zeros(1, 225, 768),
        "optical_GAP": torch.zeros(1, 768),
        "SAR_GAP": torch.zeros(1, 768),
        "joint_GAP": torch.zeros(1, 768),
    }
    result = SimpleNamespace(
        raw_optical=np.zeros((12, 120, 120), dtype=np.float32),
        raw_sar=np.zeros((2, 120, 120), dtype=np.float32),
        physical_features=np.zeros((1, 62), dtype=np.float32),
        hybrid_features=np.zeros((1, 192), dtype=np.float32),
        croma_features=croma,
        phase2_input=SimpleNamespace(pooled_croma_features=np.zeros((1, 2304), dtype=np.float32)),
        receipt={"sample_id": "sample-1", "provenance": {"phase1": {"status": "validated"}}},
    )

    path = write_artifacts(result, tmp_path / "sample-1")

    assert path.name == "receipt.json"
    verified = verify_artifact_bundle(path.parent)
    assert set(verified["arrays"]) == {
        "raw_optical", "raw_sar", "physical_features", "pooled_croma_features", "hybrid_features",
        "croma_optical_encodings", "croma_SAR_encodings", "croma_joint_encodings",
        "croma_optical_GAP", "croma_SAR_GAP", "croma_joint_GAP",
    }