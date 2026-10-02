import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from src.generic_raster import GenericRasterInspectionError, inspect_generic_raster
from src.sensor_adapters import inspect_and_gate, validate_external_pair


def write_tiff(path, *, bands=3, values=None, crs="EPSG:32633", transform=None):
    values = np.ones((bands, 10, 10), dtype=np.float32) if values is None else values
    with rasterio.open(path, "w", driver="GTiff", height=10, width=10, count=bands, dtype="float32",
                       crs=crs, transform=transform or from_origin(0, 100, 10, 10)) as dst:
        dst.write(values)


def declaration(sensor, modality, role, bands):
    labels = {"sentinel-2": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
              "sentinel-1": ["VV", "VH"]}.get(sensor, [f"SOURCE_BAND_{index + 1}" for index in range(bands)])
    return {"sensor": sensor, "modality": modality, "role": role, "band_order": labels, "band_order_confirmed": sensor.startswith("sentinel")}


def test_png_jpeg_and_rgb_tiff_share_generic_rgb_contract(tmp_path):
    png = tmp_path / "rgb.png"; jpg = tmp_path / "rgb.jpg"; tif = tmp_path / "rgb.tif"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(png); Image.new("RGB", (8, 8), color=(1, 2, 3)).save(jpg)
    write_tiff(tif, bands=3)
    for path in (png, jpg, tif):
        result = inspect_and_gate(str(path), declaration("generic-rgb", "rgb", "SINGLE", 3), "SINGLE_IMAGE_SCENE_DESCRIPTION", inspector=inspect_generic_raster)
        assert result["compatibility_gate"]["eligible_for_agent"] is True
        assert result["inspection"]["band_count"] == 3


@pytest.mark.parametrize("bands", [3, 4, 8, 12, 17])
def test_arbitrary_multiband_is_never_a_sensor_guess(tmp_path, bands):
    path = tmp_path / f"multiband-{bands}.tif"; write_tiff(path, bands=bands)
    no_declaration = inspect_and_gate(str(path), None, "SINGLE_IMAGE_VQA", inspector=inspect_generic_raster)
    assert no_declaration["generic_classification"] in {"GENERIC_RGB_CANDIDATE", "GENERIC_MULTISPECTRAL"}
    assert no_declaration["compatibility_gate"]["code"] == "MISSING_SENSOR_DECLARATION"
    assert no_declaration["compatibility_gate"]["eligible_for_agent"] is False


def test_sentinel_declarations_are_validated_not_inferred(tmp_path):
    s2 = tmp_path / "s2.tif"; s1 = tmp_path / "s1.tif"
    write_tiff(s2, bands=12); write_tiff(s1, bands=2)
    s2_result = inspect_and_gate(str(s2), declaration("sentinel-2", "multispectral", "SINGLE", 12), "SINGLE_IMAGE_VQA", inspector=inspect_generic_raster)
    assert s2_result["compatibility_gate"]["status"] == "INCOMPATIBLE"  # dimensions are intentionally not the VQA contract
    s1_result = inspect_and_gate(str(s1), declaration("sentinel-1", "sar", "S1", 2), "OPTICAL_SAR_ANALYSIS", inspector=inspect_generic_raster)
    assert s1_result["adapter"]["adapter"] == "sentinel-1-declared"
    assert s1_result["sensor_declaration"]["band_order"] == ["VV", "VH"]


@pytest.mark.parametrize("payload", [b"", b"not an image", b"II*\x00truncated"])
def test_corrupt_tiff_is_structured_and_never_read_as_model_input(tmp_path, payload):
    path = tmp_path / "corrupt.tif"; path.write_bytes(payload)
    with pytest.raises(GenericRasterInspectionError) as exc:
        inspect_generic_raster(path)
    assert exc.value.code in {"INVALID_RASTER", "RASTER_UNREADABLE"}


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_block_compatibility(tmp_path, bad):
    values = np.ones((3, 10, 10), dtype=np.float32); values[0, 0, 0] = bad
    path = tmp_path / "nonfinite.tif"; write_tiff(path, values=values)
    result = inspect_and_gate(str(path), declaration("generic-rgb", "rgb", "SINGLE", 3), "SINGLE_IMAGE_SCENE_DESCRIPTION", inspector=inspect_generic_raster)
    assert result["compatibility_gate"]["code"] == "NONFINITE_RASTER_VALUES"


def test_external_pair_requires_spatial_facts_and_never_claims_coregistration(tmp_path):
    first_path = tmp_path / "first.tif"; second_path = tmp_path / "second.tif"
    write_tiff(first_path, bands=2, transform=from_origin(0, 100, 10, 10)); write_tiff(second_path, bands=12, transform=from_origin(1000, 100, 10, 10))
    first = inspect_and_gate(str(first_path), declaration("sentinel-1", "sar", "S1", 2), "OPTICAL_SAR_ANALYSIS", inspector=inspect_generic_raster)
    second = inspect_and_gate(str(second_path), declaration("sentinel-2", "multispectral", "S2", 12), "OPTICAL_SAR_ANALYSIS", inspector=inspect_generic_raster)
    assert validate_external_pair(first, second, pair_kind="OPTICAL_SAR")["code"] == "PAIR_NO_SPATIAL_OVERLAP"
    temporal = validate_external_pair({**first, "sensor_declaration": {**first["sensor_declaration"], "role": "T1"}}, {**second, "sensor_declaration": {**second["sensor_declaration"], "role": "T2"}}, pair_kind="TEMPORAL")
    assert temporal["code"] == "PAIR_NO_SPATIAL_OVERLAP"


def test_identical_or_missing_temporal_role_is_rejected_without_filename_guessing(tmp_path):
    path = tmp_path / "one.tif"; write_tiff(path, bands=3)
    result = inspect_and_gate(str(path), declaration("generic-rgb", "rgb", "T1", 3), "TEMPORAL_CHANGE_DESCRIPTION", inspector=inspect_generic_raster)
    assert validate_external_pair(result, result, pair_kind="TEMPORAL")["code"] == "TEMPORAL_ORDER_REQUIRED"
