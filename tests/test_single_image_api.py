import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src.api import build_single_image_request
from src.dataset_loader import OPTICAL_BANDS


PATCH = "S2_ONLY_01_02"


def make_dataset(root, split="train"):
    s2_root = root / "BigEarthNet-S2" / PATCH
    s2_root.mkdir(parents=True)
    transform = from_origin(0, 1200, 10, 10)
    for band in OPTICAL_BANDS:
        path = s2_root / f"{PATCH}_{band}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=120, width=120, count=1,
                           dtype="float32", crs="EPSG:32633", transform=transform) as dst:
            dst.write(np.ones((120, 120), dtype=np.float32), 1)
    pd.DataFrame([{"patch_id": PATCH, "split": split}]).to_parquet(root / "metadata.parquet")


def test_build_single_image_request_uses_only_s2_and_preserves_metadata(tmp_path):
    make_dataset(tmp_path)
    result = build_single_image_request({
        "dataset_root": str(tmp_path), "patch_id": PATCH, "task_type": "binary_qa",
        "question": "Is water visible?", "modality": "S2",
    })

    assert result["route"] == "SINGLE_IMAGE_VQA"
    assert result["split"] == "train"
    assert result["optical"].shape == (12, 120, 120)
    assert result["spatial_metadata"]["crs"] == "EPSG:32633"
    assert result["spatial_metadata"]["resolution"] == [10, 10]


def test_build_single_image_request_selects_blocked_grounding(tmp_path):
    make_dataset(tmp_path)
    result = build_single_image_request({
        "dataset_root": str(tmp_path), "patch_id": PATCH, "task": "SINGLE_IMAGE_GROUNDING",
        "task_type": "grounding", "question": "Highlight water",
    })
    assert result["route"] == "SINGLE_IMAGE_GROUNDING"
    assert result["optical"].shape == (12, 120, 120)


@pytest.mark.parametrize("split", ["test", "unknown"])
def test_build_single_image_request_never_resolves_test_patch(tmp_path, split):
    make_dataset(tmp_path, split=split)
    with pytest.raises(FileNotFoundError, match="Non-test S2 patch"):
        build_single_image_request({
            "dataset_root": str(tmp_path), "patch_id": PATCH, "task_type": "binary_qa",
            "question": "Is water visible?",
        })


def test_build_single_image_request_does_not_fallback_for_paired_intent(tmp_path):
    make_dataset(tmp_path)
    result = build_single_image_request({
        "dataset_root": str(tmp_path), "patch_id": PATCH, "task_type": "binary_qa",
        "question": "Use the SAR and optical image together.",
    })
    assert result["route"] == "OPTICAL_SAR_ANALYSIS"
    assert "optical" not in result


def test_build_single_image_request_rejects_more_than_one_image():
    with pytest.raises(ValueError, match="exactly one image"):
        build_single_image_request({"image_count": 2, "question": "compare"})


def test_build_single_image_request_accepts_confirmed_external_12_band_tiff(tmp_path):
    path=tmp_path/"external.tif"
    transform=from_origin(0,1200,10,10)
    with rasterio.open(path,"w",driver="GTiff",height=120,width=120,count=12,dtype="float32",
                       crs="EPSG:32633",transform=transform) as dst:
        for index in range(12): dst.write(np.full((120,120),index+1,dtype=np.float32),index+1)

    request=build_single_image_request({"task_type":"binary_qa","question":"Is water visible?",
                                        "band_order_confirmed":True,
                                        "sensor_declaration":{"sensor":"sentinel-2","modality":"multispectral","role":"SINGLE",
                                                              "band_order":list(OPTICAL_BANDS),"band_order_confirmed":True}},uploaded_path=path)

    assert request["split"]=="external_inference"
    assert request["optical"].shape==(12,120,120)
    assert request["image_id"].startswith("upload:")
    assert request["source_asset_sha256"]


def test_build_single_image_request_requires_band_confirmation_when_descriptions_absent(tmp_path):
    path=tmp_path/"external.tif"
    with rasterio.open(path,"w",driver="GTiff",height=120,width=120,count=12,dtype="float32",
                       crs="EPSG:32633",transform=from_origin(0,1200,10,10)) as dst:
        for index in range(12): dst.write(np.ones((120,120),dtype=np.float32),index+1)

    with pytest.raises(ValueError,match="confirm canonical order"):
        build_single_image_request({"task_type":"binary_qa","question":"Is water visible?",
                                    "sensor_declaration":{"sensor":"sentinel-2","modality":"multispectral","role":"SINGLE",
                                                          "band_order":list(OPTICAL_BANDS),"band_order_confirmed":True}},uploaded_path=path)


def test_build_single_image_request_rejects_undeclared_external_sensor(tmp_path):
    path=tmp_path/"external.tif"
    with rasterio.open(path,"w",driver="GTiff",height=120,width=120,count=12,dtype="float32",
                       crs="EPSG:32633",transform=from_origin(0,1200,10,10)) as dst:
        dst.write(np.ones((12,120,120),dtype=np.float32))
    with pytest.raises(ValueError,match="explicit Sentinel-2 sensor declaration"):
        build_single_image_request({"task_type":"binary_qa","question":"Is water visible?", "band_order_confirmed":True},uploaded_path=path)
