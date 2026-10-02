from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src.dataset_loader import (OPTICAL_BANDS, SAR_BANDS, Sample, discover_samples,
                                discover_s2_samples, load_optical_sample, load_sample)


def write(path, data, *, crs="EPSG:32633", transform=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = transform or from_origin(0, 1200, 10, 10)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                       dtype=data.dtype, crs=crs, transform=transform) as dst: dst.write(data, 1)


def area(tmp_path):
    o={}; s={}; base=np.ones((120,120),dtype=np.float32)
    for band in OPTICAL_BANDS: o[band]=tmp_path/f"x_{band}.tif"; write(o[band],base)
    for band in SAR_BANDS: s[band]=tmp_path/f"x_{band}.tif"; write(s[band],base)
    ref=tmp_path/"x_reference_map.tif"; write(ref,np.full((120,120),111,dtype=np.uint16))
    return Sample("area",o,s,ref)


def test_complete_area_loads_canonical_contract(tmp_path):
    item=load_sample(area(tmp_path))
    assert item.raw_optical.shape==(12,120,120) and item.raw_sar.shape==(2,120,120)
    assert item.reference.shape==(120,120) and item.metadata["optical_band_order"]==list(OPTICAL_BANDS)


@pytest.mark.parametrize("band", OPTICAL_BANDS)
def test_each_missing_optical_band_is_explicit(tmp_path,band):
    sample=area(tmp_path); del sample.optical_paths[band]
    with pytest.raises(FileNotFoundError,match=band): load_sample(sample)


@pytest.mark.parametrize("band", SAR_BANDS)
def test_each_missing_sar_band_is_explicit(tmp_path,band):
    sample=area(tmp_path); del sample.sar_paths[band]
    with pytest.raises(FileNotFoundError,match=band): load_sample(sample)


def test_missing_reference_is_explicit(tmp_path):
    sample=area(tmp_path); sample.reference_map=Path()
    with pytest.raises(FileNotFoundError,match="reference=True"): load_sample(sample)


@pytest.mark.parametrize("kind",["wrong_dimensions","invalid_crs","shifted_grid","nan","inf","corrupt"])
def test_malformed_inputs_are_rejected(tmp_path,kind):
    sample=area(tmp_path); path=sample.sar_paths["VV"]
    if kind=="corrupt": path.write_bytes(b"not a tiff")
    else:
        data=np.ones((119,120),dtype=np.float32) if kind=="wrong_dimensions" else np.ones((120,120),dtype=np.float32)
        if kind=="nan": data[0,0]=np.nan
        if kind=="inf": data[0,0]=np.inf
        write(path,data,crs=None if kind=="invalid_crs" else "EPSG:32633",transform=from_origin(10,1200,10,10) if kind=="shifted_grid" else None)
    with pytest.raises(ValueError): load_sample(sample)


def test_strict_discovery_uses_full_metadata_identity(tmp_path):
    for name in ("BigEarthNet-S2","BigEarthNet-S1","Reference_Maps"): (tmp_path/name).mkdir()
    pd.DataFrame([{"patch_id":"S2_FULL_01_02","s1_name":"S1_FULL_01_02"}]).to_parquet(tmp_path/"metadata.parquet")
    samples=discover_samples(tmp_path,strict=True)
    assert samples[0].patch_id=="S2_FULL_01_02"


def test_s2_discovery_and_load_need_no_s1_or_reference_folders(tmp_path):
    s2_root=tmp_path/"BigEarthNet-S2"/"S2_ONLY_01_02"
    paths={}
    for band in OPTICAL_BANDS:
        path=s2_root/f"S2_ONLY_01_02_{band}.tif"
        write(path,np.ones((120,120),dtype=np.float32))
        paths[band]=path
    pd.DataFrame([{"patch_id":"S2_ONLY_01_02","split":"train"}]).to_parquet(tmp_path/"metadata.parquet")

    samples=discover_s2_samples(tmp_path)
    prepared=load_optical_sample(samples[0])

    assert samples[0].patch_id=="S2_ONLY_01_02"
    assert samples[0].sar_paths=={} and samples[0].reference_map==Path()
    assert prepared.raw_optical.shape==(12,120,120)
    assert prepared.metadata["crs"]=="EPSG:32633"
    assert prepared.metadata["optical_band_order"]==list(OPTICAL_BANDS)


def test_s2_discovery_excludes_test_split_by_default(tmp_path):
    (tmp_path/"BigEarthNet-S2").mkdir()
    pd.DataFrame([{"patch_id":"held-out","split":"test"}]).to_parquet(tmp_path/"metadata.parquet")

    assert discover_s2_samples(tmp_path)==[]
