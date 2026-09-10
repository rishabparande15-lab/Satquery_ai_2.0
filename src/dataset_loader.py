from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .preprocessing import NormalizedPair, normalize_pair


OPTICAL_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
SAR_BANDS = ("VV", "VH")
PATCH_RE = re.compile(r"_(\d+_\d+)$")


@dataclass
class Sample:
    patch_id: str
    optical_paths: dict[str, Path]
    sar_paths: dict[str, Path]
    reference_map: Path


@dataclass
class PreparedSample:
    patch_id: str
    optical: np.ndarray
    sar: np.ndarray
    raw_optical: np.ndarray
    raw_sar: np.ndarray
    metadata: dict


def discover_samples(root: Path) -> list[Sample]:
    s2_files = list((root / "BigEarthNet-S2").rglob("*.tif"))
    s1_files = list((root / "BigEarthNet-S1").rglob("*.tif"))
    reference_files = list((root / "Reference_Maps").rglob("*_reference_map.tif"))
    samples: dict[str, Sample] = {}
    for path in s2_files:
        match = re.search(r"_(\d+_\d+)_B(?:01|02|03|04|05|06|07|08|8A|09|11|12)\.tif$", path.name)
        if match:
            patch_id = match.group(1)
            sample = samples.setdefault(patch_id, Sample(patch_id, {}, {}, Path()))
            band = path.stem.rsplit("_", 1)[-1]
            if band in sample.optical_paths:
                raise ValueError(f"{patch_id}: duplicate optical band {band}")
            sample.optical_paths[band] = path
    for path in s1_files:
        match = PATCH_RE.search(path.parent.name)
        if match:
            patch_id = match.group(1)
            sample = samples.setdefault(patch_id, Sample(patch_id, {}, {}, Path()))
            band = path.stem.rsplit("_", 1)[-1]
            if band in sample.sar_paths:
                raise ValueError(f"{patch_id}: duplicate SAR band {band}")
            sample.sar_paths[band] = path
    for path in reference_files:
        match = PATCH_RE.search(path.parent.name)
        if match and match.group(1) in samples:
            samples[match.group(1)].reference_map = path
    result = sorted(samples.values(), key=lambda sample: sample.patch_id)
    if not result:
        raise ValueError("No complete samples discovered")
    return result


def _read_to_grid(path: Path, grid: rasterio.io.DatasetReader, resampling: Resampling) -> np.ndarray:
    destination = np.empty((grid.height, grid.width), dtype=np.float32)
    with rasterio.open(path) as source:
        source_data = source.read(1).astype(np.float32)
        if source.nodata is not None:
            source_data[source_data == source.nodata] = np.nan
        reproject(
            source_data,
            destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return destination


def load_sample(sample: Sample) -> PreparedSample:
    missing_optical = [band for band in OPTICAL_BANDS if band not in sample.optical_paths]
    missing_sar = [band for band in SAR_BANDS if band not in sample.sar_paths]
    if missing_optical or missing_sar or not sample.reference_map:
        raise FileNotFoundError(f"{sample.patch_id}: missing optical={missing_optical}, sar={missing_sar}, reference={not sample.reference_map}")
    with rasterio.open(sample.optical_paths["B02"]) as grid:
        if grid.crs is None or grid.res[0] <= 0 or grid.res[1] <= 0:
            raise ValueError(f"{sample.patch_id}: invalid reference raster CRS/resolution")
        optical = np.stack([_read_to_grid(sample.optical_paths[band], grid, Resampling.bilinear) for band in OPTICAL_BANDS])
        sar = np.stack([_read_to_grid(sample.sar_paths[band], grid, Resampling.bilinear) for band in SAR_BANDS])
        grid_metadata = {
            "shape": [grid.height, grid.width],
            "crs": str(grid.crs),
            "resolution": list(grid.res),
            "bounds": list(grid.bounds),
            "transform": list(grid.transform),
            "dtype_reference": grid.dtypes[0],
        }
    if not np.isfinite(optical).any() or not np.isfinite(sar).any():
        raise ValueError(f"{sample.patch_id}: rasters contain no finite pixels")
    if not np.isfinite(optical).all() or not np.isfinite(sar).all():
        raise ValueError(f"{sample.patch_id}: tensors contain non-finite values after reprojection")
    normalized: NormalizedPair = normalize_pair(optical, sar)
    return PreparedSample(sample.patch_id, normalized.optical, normalized.sar, optical, sar, {
        **grid_metadata,
        "optical_band_order": list(OPTICAL_BANDS),
        "sar_band_order": list(SAR_BANDS),
        "optical_normalization": normalized.optical_method,
        "sar_normalization": normalized.sar_method,
        "optical_resampling": "rasterio.warp.reproject bilinear to B02 10 m grid",
        "sar_resampling": "rasterio.warp.reproject bilinear to B02 10 m grid",
        "reference_map_resampling": "not applied; categorical map retained at native 10 m",
    })