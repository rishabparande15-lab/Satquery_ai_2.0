import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.eo_vlm.s2_asset_gate import (EXPECTED_NATIVE, S2_BANDS, VerifiedS2Asset, deterministic_manifest,
                                      inspect_area, training_gate, validate_annotation)


def annotation(**changes):
    value = {"record_id": "record-a", "annotation_id": "ann-a", "image_id": "area-a", "split": "train", "effective_partition": "train",
             "task_type": "binary_qa", "dependency_class": "VISUAL_ONLY", "box": None, "point": None, "geometry_frame": None,
             "source_revision": "rev-1", "source_record_sha256": "a" * 64}
    value.update(changes); return value


def asset(**changes):
    value = dict(sample_id="record-a", image_id="area-a", split="train", local_path="x", source_revision="rev-1", sha256="b" * 64,
                 shape=(12, 120, 120), dtype="uint16", band_order=S2_BANDS, normalization_revision="robust_channel_scale_v1",
                 provenance={"source_record_sha256": "a" * 64, "annotation_id": "ann-a"})
    value.update(changes); return VerifiedS2Asset(**value)


def write_area(root: Path, *, bad_band=None, nan=False, nodata=False):
    paths = {}
    for band, (height, width) in EXPECTED_NATIVE.items():
        path = root / f"area-a_{band}.tif"; data = np.ones((height, width), dtype=np.float32)
        if band == bad_band: data = np.ones((height - 1, width), dtype=np.float32)
        if nan and band == "B01": data[0, 0] = np.nan
        profile = {"driver": "GTiff", "height": data.shape[0], "width": data.shape[1], "count": 1, "dtype": "float32", "crs": "EPSG:32632", "transform": from_origin(0, 0, 10, 10), "nodata": -9999 if nodata and band == "B01" else None}
        if nodata and band == "B01": data[0, 0] = -9999
        with rasterio.open(path, "w", **profile) as dst: dst.write(data, 1)
        paths[band] = path
    return paths


def test_valid_raw_s2_asset(tmp_path):
    assert inspect_area(annotation(), write_area(tmp_path)).shape == (12, 120, 120)


def test_invalid_shape_rejected(tmp_path):
    with pytest.raises(ValueError, match="dimensions"): inspect_area(annotation(), write_area(tmp_path, bad_band="B01"))


def test_invalid_band_order_rejected(tmp_path):
    paths = write_area(tmp_path); paths = dict(reversed(list(paths.items())))
    with pytest.raises(ValueError, match="band ordering"): inspect_area(annotation(), paths)


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_nonfinite_rejected(tmp_path, bad):
    paths = write_area(tmp_path)
    with rasterio.open(paths["B01"], "r+") as dst:
        data = dst.read(1); data[0, 0] = float(bad); dst.write(data, 1)
    with pytest.raises(ValueError, match="NaN or Inf"): inspect_area(annotation(), paths)


def test_nodata_rejected(tmp_path):
    with pytest.raises(ValueError, match="nodata"): inspect_area(annotation(), write_area(tmp_path, nodata=True))


@pytest.mark.parametrize("change", [{"source_revision": ""}, {"source_record_sha256": ""}, {"split": "test"}, {"effective_partition": "unknown"}, {"box": [0, 0, 1, 1]}, {"task_type": "temporal_qa"}, {"task_type": "grounding"}, {"dependency_class": "SAR"}])
def test_invalid_annotation_boundaries(change):
    with pytest.raises(ValueError): validate_annotation(annotation(**change))


def test_missing_checksum_rejected():
    with pytest.raises(ValueError): asset(sha256="").validate()


def test_cross_split_collision_rejected():
    other = asset(sample_id="record-b", split="train")
    with pytest.raises(ValueError, match="duplicate image"): deterministic_manifest([asset(), other], {"record-a": annotation(), "record-b": annotation(record_id="record-b")})


def test_duplicate_hash_rejected():
    other = asset(sample_id="record-b", image_id="area-b")
    with pytest.raises(ValueError, match="duplicate S2"): deterministic_manifest([asset(), other], {"record-a": annotation(), "record-b": annotation(record_id="record-b", image_id="area-b")})


def test_manifest_is_deterministic():
    records = {"record-a": annotation()}
    assert deterministic_manifest([asset()], records) == deterministic_manifest([asset()], records)


def test_projector_smoke_schema_is_exposed():
    gate = training_gate(assets=[asset()], projector_smoke_test="PASS")
    assert set(("projector_smoke_test", "training_input_ready", "execution_permitted")) <= set(gate)


def test_training_gate_fails_closed():
    gate = training_gate(assets=[], projector_smoke_test="NOT_RUN")
    assert not gate["training_input_ready"] and not gate["execution_permitted"]


def test_complete_synthetic_fixture_is_ready_but_not_executable():
    gate = training_gate(assets=[asset()], projector_smoke_test="PASS")
    assert gate["training_input_ready"] and not gate["execution_permitted"]


def test_incomplete_synthetic_fixture_is_blocked():
    gate = training_gate(assets=[asset(sha256="")], projector_smoke_test="PASS")
    assert not gate["training_input_ready"]
