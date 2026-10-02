import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.generic_raster import GenericRasterInspectionError, inspect_generic_raster
from src.sensor_adapters import SensorDeclarationError, inspect_and_gate


def write_raster(path, *, count=3, descriptions=None, finite=True):
    data = np.ones((count, 20, 20), dtype=np.float32)
    if not finite:
        data[0, 0, 0] = np.nan
    with rasterio.open(path, "w", driver="GTiff", height=20, width=20, count=count, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(600000, 5400000, 2, 2)) as dst:
        dst.write(data)
        if descriptions:
            for index, description in enumerate(descriptions, 1):
                dst.set_band_description(index, description)


def test_generic_raster_inspector_reports_metadata_without_sensor_inference(tmp_path):
    path = tmp_path / "cartosat.tif"
    write_raster(path, descriptions=["red", "green", "blue"])

    result = inspect_generic_raster(path)

    assert result["driver"] == "GTiff"
    assert result["shape"] == [20, 20]
    assert result["band_count"] == 3
    assert result["crs"] == "EPSG:32633"
    assert result["band_descriptions"] == ["red", "green", "blue"]
    assert result["finite_value_checks"]["all_unmasked_values_finite"] is True
    assert "sensor" not in result


@pytest.mark.parametrize("sensor, modality", [("cartosat-2s", "optical"), ("risat", "sar")])
def test_declared_indian_sensor_foundations_block_model_execution(tmp_path, sensor, modality):
    path = tmp_path / f"{sensor}.tif"
    write_raster(path, count=2 if sensor == "risat" else 3, descriptions=["band1", "band2"] if sensor == "risat" else ["red", "green", "blue"])

    result = inspect_and_gate(str(path), {"sensor": sensor, "modality": modality, "role": "SINGLE",
                                          "band_order": ["band1", "band2"] if sensor == "risat" else ["red", "green", "blue"]},
                              "SINGLE_IMAGE_SCENE_DESCRIPTION", inspector=inspect_generic_raster)

    assert result["status"] == "INSPECTED"
    assert result["adapter"]["status"] == "FOUNDATION_ONLY"
    assert result["compatibility_gate"]["status"] == "BLOCKED"
    assert result["compatibility_gate"]["code"] == "SENSOR_DOMAIN_NOT_VALIDATED"
    assert result["agent_handoff"]["eligible"] is False
    assert result["adapter"]["band_reordering_applied"] is False
    assert result["adapter"]["resampling_applied"] is False


def test_sentinel_vqa_gate_requires_explicit_canonical_order_and_preserves_order(tmp_path):
    path = tmp_path / "s2.tif"
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    with rasterio.open(path, "w", driver="GTiff", height=120, width=120, count=12, dtype="float32",
                       crs="EPSG:32633", transform=from_origin(0, 1200, 10, 10)) as dst:
        dst.write(np.ones((12, 120, 120), dtype=np.float32))
        for index, band in enumerate(bands, 1):
            dst.set_band_description(index, band)

    result = inspect_and_gate(str(path), {"sensor": "sentinel-2", "modality": "multispectral", "role": "SINGLE",
                                          "band_order": bands, "band_order_confirmed": True},
                              "SINGLE_IMAGE_VQA", inspector=inspect_generic_raster)

    assert result["compatibility_gate"]["status"] == "COMPATIBLE"
    assert result["adapter"]["transforms_applied"] == []
    assert result["adapter"]["band_reordering_applied"] is False


def test_declaration_is_required_and_nonfinite_rasters_fail_the_gate(tmp_path):
    path = tmp_path / "broken.tif"
    write_raster(path, finite=False, descriptions=["red", "green", "blue"])
    with pytest.raises(SensorDeclarationError, match="Declare sensor"):
        inspect_and_gate(str(path), {}, inspector=inspect_generic_raster)

    result = inspect_and_gate(str(path), {"sensor": "generic-rgb", "modality": "rgb", "role": "SINGLE",
                                          "band_order": ["red", "green", "blue"]},
                              "SINGLE_IMAGE_SCENE_DESCRIPTION", inspector=inspect_generic_raster)
    assert result["compatibility_gate"]["code"] == "NONFINITE_RASTER_VALUES"


def test_inspector_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "not-raster.bmp"
    path.write_bytes(b"not a raster")
    with pytest.raises(GenericRasterInspectionError, match="Supported external inputs"):
        inspect_generic_raster(path)
