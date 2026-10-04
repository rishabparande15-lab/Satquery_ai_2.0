import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.api as api
from src.api import (OPTICAL_BANDS, SAR_BANDS, Sample, _approved_demo_bundle_path,
                     _approved_pair_demo_sample, build_fixed_pair_route_request)


def _write(path, content=b"approved-sar"):
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_approved_sar_fixture_resolves_from_bundle(tmp_path):
    fixture = tmp_path / "phase3ad_validation_s1.tif"
    digest = _write(fixture)
    assert _approved_demo_bundle_path(tmp_path, fixture.name, digest) == fixture.resolve()


def test_missing_approved_sar_fixture_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="Approved demo sample unavailable"):
        _approved_demo_bundle_path(tmp_path, "phase3ad_validation_s1.tif", "0" * 64)


def test_sar_fixture_path_cannot_escape_bundle(tmp_path):
    escaped = tmp_path.parent / "phase3ad_validation_s1.tif"
    digest = _write(escaped)
    with pytest.raises(FileNotFoundError, match="Approved demo sample unavailable"):
        _approved_demo_bundle_path(tmp_path, "../phase3ad_validation_s1.tif", digest)


def test_unrelated_two_band_file_is_not_substituted(tmp_path):
    approved = tmp_path / "phase3ad_validation_s1.tif"
    unrelated = tmp_path / "unrelated_two_band.tif"
    digest = _write(approved, b"approved")
    _write(unrelated, b"unrelated")
    assert _approved_demo_bundle_path(tmp_path, approved.name, digest) == approved.resolve()


def test_approved_pair_uses_the_s1_acquisition_directory(tmp_path):
    patch_id = "S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22"
    s1_id = "S1B_IW_GRDH_1SDV_20170717T064605_29UPV_35_22"
    s2_dir = tmp_path / "BigEarthNet-S2" / patch_id.rsplit("_", 2)[0] / patch_id
    s1_dir = tmp_path / "BigEarthNet-S1" / s1_id.rsplit("_", 3)[0] / s1_id
    reference_dir = tmp_path / "Reference_Maps" / patch_id.rsplit("_", 2)[0] / patch_id
    for directory in (s2_dir, s1_dir, reference_dir):
        directory.mkdir(parents=True)
    for band in OPTICAL_BANDS:
        (s2_dir / f"{patch_id}_{band}.tif").touch()
    for band in SAR_BANDS:
        (s1_dir / f"{s1_id}_{band}.tif").touch()
    (reference_dir / f"{patch_id}_reference_map.tif").touch()
    sample = _approved_pair_demo_sample(tmp_path, patch_id, "validation")
    assert sample.sar_paths["VV"] == s1_dir / f"{s1_id}_VV.tif"


def test_approved_pair_normalizes_both_logical_patch_ids(monkeypatch):
    patch_id = "S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22"
    sample = Sample(patch_id, {"B02": Path("s2.tif")}, {"VV": Path("s1.tif")}, Path("reference.tif"), "validation")
    prepared = SimpleNamespace(
        sar=np.zeros((2, 120, 120)), optical=np.zeros((12, 120, 120)),
        metadata={"crs": "EPSG:32629", "resolution": (10, 10), "bounds": (0, 0, 1, 1), "shape": (120, 120), "optical_band_order": list(OPTICAL_BANDS), "sar_band_order": list(SAR_BANDS)},
    )
    monkeypatch.setattr(api, "load_sample", lambda _: prepared)
    resolved = build_fixed_pair_route_request({}, sample)
    assert resolved["s1_patch_id"] == resolved["s2_patch_id"] == patch_id
