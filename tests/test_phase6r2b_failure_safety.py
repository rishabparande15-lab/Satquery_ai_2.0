"""Temporary-fixture safety checks required by the Phase 6R.2B audit."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
import torch

from scripts.run_pipeline3_5000_baseline import _load_probe, load_cache


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CACHE = Path(r"D:\Satquery_ai datasets\comparison\pipeline3-5000\scene_features")


def test_corrupted_feature_artifact_fails_closed(tmp_path):
    source_manifest = json.loads((SOURCE_CACHE / "manifest.json").read_text(encoding="utf-8"))
    source_entry = source_manifest["shards"][0]
    shard = tmp_path / "shard.npz"
    shutil.copyfile(source_entry["path"], shard)
    content = bytearray(shard.read_bytes())
    content[len(content) // 2] ^= 0xFF
    shard.write_bytes(content)
    manifest = {
        **source_manifest,
        "shards": [{**source_entry, "path": str(shard)}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Feature shard hash changed"):
        load_cache(tmp_path, include_test=True)


def test_incompatible_checkpoint_fails_closed(tmp_path):
    checkpoint = tmp_path / "incompatible.pt"
    torch.save({
        "input_dimension": 829,
        "dataset_fingerprint": "wrong",
        "state_dict": {},
    }, checkpoint)
    with pytest.raises(ValueError, match="Checkpoint contract mismatch"):
        _load_probe(checkpoint, 830)
