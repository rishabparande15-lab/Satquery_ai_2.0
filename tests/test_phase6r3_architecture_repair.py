from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.architecture_contracts import CapabilityRegistry
from src.deterministic_scene_analysis import TASK_TYPE
from src.pipeline3_scene_probe import (
    IdentityResolutionError,
    ModelContractError,
    Pipeline3ManifestResolver,
    _load_probe,
    validate_representation_dimensions,
)


def _resolver(rows):
    resolver = Pipeline3ManifestResolver.__new__(Pipeline3ManifestResolver)
    resolver.rows = tuple(rows)
    resolver._indices = {field: resolver._build_index(field) for field in ("area_id", "s2_identity", "s1_identity")}
    return resolver


def _row(area, s2, s1):
    return {"area_id": area, "s2_identity": s2, "s1_identity": s1}


def test_full_bigearthnet_identity_prevents_known_suffix_collision():
    first = _row("S2A_FULL_TILE_A_07_47", "S2A_SHORT_A_07_47", "S1A_FULL_A_07_47")
    second = _row("S2B_FULL_TILE_B_07_47", "S2B_SHORT_B_07_47", "S1B_FULL_B_07_47")
    resolver = _resolver([first, second])
    assert resolver.resolve_row(first["area_id"]) is first
    assert resolver.resolve_row(second["area_id"]) is second
    with pytest.raises(IdentityResolutionError, match="No exact"):
        resolver.resolve_row("07_47")


def test_ambiguous_authoritative_identity_fails_instead_of_selecting_first():
    first = _row("area-a", "shared-s2", "s1-a")
    second = _row("area-b", "shared-s2", "s1-b")
    resolver = _resolver([first, second])
    with pytest.raises(IdentityResolutionError, match="Ambiguous"):
        resolver.resolve_row("shared-s2")


def test_missing_area_is_explicit():
    resolver = _resolver([_row("area-a", "s2-a", "s1-a")])
    with pytest.raises(IdentityResolutionError, match="No exact Pipeline 3 area"):
        resolver.resolve_row("area-missing")


def test_missing_s2_band_is_explicit(tmp_path):
    row = {**_row("area-a_01_01", "s2-a", "s1-a"), "s2_source_roots": [str(tmp_path)]}
    resolver = _resolver([row])
    resolver.s1_archive = tmp_path / "s1.zip"
    resolver.reference_archive = tmp_path / "ref.zip"
    with pytest.raises(IdentityResolutionError, match="S2 B01"):
        resolver.resolve_bundle("area-a_01_01")


def test_missing_s1_archive_is_explicit(tmp_path, monkeypatch):
    row = {**_row("area-a_01_01", "s2-a", "s1-a"), "s2_source_roots": [str(tmp_path)]}
    resolver = _resolver([row])
    resolver.s1_archive = tmp_path / "missing-s1.zip"
    resolver.reference_archive = tmp_path / "missing-reference.zip"
    monkeypatch.setattr(Path, "is_file", lambda self: self.suffix == ".tif")
    with pytest.raises(IdentityResolutionError, match="Missing authoritative S1"):
        resolver.resolve_bundle("area-a_01_01")


def test_checkpoint_mismatch_fails_before_loading(tmp_path):
    checkpoint = tmp_path / "probe.pt"
    checkpoint.write_bytes(b"not the validated checkpoint")
    _load_probe.cache_clear()
    with pytest.raises(ModelContractError, match="file hash mismatch"):
        _load_probe(str(checkpoint))


def test_invalid_representation_dimensions_fail_explicitly():
    deep = {
        "optical_encodings": torch.zeros(1, 225, 768),
        "SAR_encodings": torch.zeros(1, 225, 768),
        "joint_encodings": torch.zeros(1, 225, 768),
        "joint_GAP": torch.zeros(1, 767),
    }
    with pytest.raises(ModelContractError, match="joint_GAP"):
        validate_representation_dimensions(np.zeros(62, dtype=np.float32), deep)


def test_registry_missing_capability_and_unsupported_task_are_explicit():
    registry = CapabilityRegistry()
    with pytest.raises(LookupError, match="unknown task type"):
        registry.require_task_type(TASK_TYPE)
    with pytest.raises(LookupError, match="unknown task type"):
        registry.require_task_type("captioning")

