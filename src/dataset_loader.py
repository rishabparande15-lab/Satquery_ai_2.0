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
    reference: np.ndarray
    metadata: dict


def discover_samples(root: Path, *, strict: bool = False) -> list[Sample]:
    root = Path(root)
    required = tuple(root / name for name in ("BigEarthNet-S2", "BigEarthNet-S1", "Reference_Maps"))
    if any(not path.is_dir() for path in required):
        raise FileNotFoundError(f"Dataset root must contain BigEarthNet-S2, BigEarthNet-S1, and Reference_Maps: {root}")
    s2_files = list((root / "BigEarthNet-S2").rglob("*.tif"))
    s1_files = list((root / "BigEarthNet-S1").rglob("*.tif"))
    reference_files = list((root / "Reference_Maps").rglob("*_reference_map.tif"))
    if strict:
        return _discover_strict(root, s2_files, s1_files, reference_files)
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
        if match:
            patch_id = match.group(1)
            sample = samples.setdefault(patch_id, Sample(patch_id, {}, {}, Path()))
            if sample.reference_map != Path():
                raise ValueError(f"{patch_id}: duplicate reference map")
            sample.reference_map = path
    result = sorted(samples.values(), key=lambda sample: sample.patch_id)
    if not result:
        raise ValueError("No complete samples discovered")
    return result


def _discover_strict(root: Path, s2_files: list[Path], s1_files: list[Path], reference_files: list[Path]) -> list[Sample]:
    """Discover areas by full identities from the official metadata table."""
    metadata_path = root / "metadata.parquet"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Strict discovery requires metadata.parquet: {metadata_path}")
    try:
        import pandas as pd
        frame = pd.read_parquet(metadata_path, columns=["patch_id", "s1_name"])
    except Exception as error:
        raise ValueError(f"Strict metadata is unreadable: {metadata_path}") from error
    if frame["patch_id"].duplicated().any() or frame["s1_name"].duplicated().any():
        raise ValueError("Strict metadata contains duplicate S1/S2 area identities")
    s2_by_parent: dict[str, list[Path]] = {}
    s1_by_parent: dict[str, list[Path]] = {}
    ref_by_parent: dict[str, list[Path]] = {}
    for path in s2_files: s2_by_parent.setdefault(path.parent.name, []).append(path)
    for path in s1_files: s1_by_parent.setdefault(path.parent.name, []).append(path)
    for path in reference_files: ref_by_parent.setdefault(path.parent.name, []).append(path)
    samples = []
    for row in frame.itertuples(index=False):
        patch_id, s1_name = str(row.patch_id), str(row.s1_name)
        optical: dict[str, Path] = {}
        sar: dict[str, Path] = {}
        for path in s2_by_parent.get(patch_id, []):
            band = path.stem.rsplit("_", 1)[-1]
            if band in OPTICAL_BANDS:
                if band in optical: raise ValueError(f"{patch_id}: duplicate optical band {band}")
                optical[band] = path
        for path in s1_by_parent.get(s1_name, []):
            band = path.stem.rsplit("_", 1)[-1]
            if band in SAR_BANDS:
                if band in sar: raise ValueError(f"{patch_id}: duplicate SAR band {band}")
                sar[band] = path
        references = ref_by_parent.get(patch_id, [])
        if len(references) > 1: raise ValueError(f"{patch_id}: duplicate reference map")
        samples.append(Sample(patch_id, optical, sar, references[0] if references else Path()))
    if not samples:
        raise ValueError("No strict metadata areas discovered")
    return samples


def _validate_source(path: Path, *, patch_id: str, role: str, grid=None, exact_grid: bool = False) -> None:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"{patch_id}: unsupported {role} format: {path.suffix}")
    try:
        with rasterio.open(path) as source:
            if source.count != 1 or source.width <= 0 or source.height <= 0:
                raise ValueError(f"{patch_id}: invalid {role} dimensions/band count")
            if source.crs is None:
                raise ValueError(f"{patch_id}: invalid {role} CRS")
            transform = source.transform
            if transform.a <= 0 or transform.e >= 0 or transform.b != 0 or transform.d != 0:
                raise ValueError(f"{patch_id}: invalid {role} orientation")
            bounds = tuple(source.bounds)
            if not all(np.isfinite(bounds)) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError(f"{patch_id}: invalid {role} bounds")
            values = source.read(1, masked=True).astype(np.float32).filled(np.nan)
            if not np.isfinite(values).all():
                raise ValueError(f"{patch_id}: {role} contains NaN/Inf/nodata")
            if grid is not None:
                if source.crs != grid.crs:
                    raise ValueError(f"{patch_id}: mismatched {role} CRS")
                if not np.allclose(bounds, tuple(grid.bounds), atol=1e-6, rtol=0):
                    raise ValueError(f"{patch_id}: mismatched {role} bounds")
                if exact_grid and (source.width != grid.width or source.height != grid.height or source.transform != grid.transform):
                    raise ValueError(f"{patch_id}: shifted or mismatched {role} grid")
    except rasterio.errors.RasterioIOError as error:
        raise ValueError(f"{patch_id}: corrupted or unreadable {role} TIFF") from error


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
    missing_reference = sample.reference_map == Path()
    if missing_optical or missing_sar or missing_reference:
        raise FileNotFoundError(f"{sample.patch_id}: missing optical={missing_optical}, sar={missing_sar}, reference={missing_reference}")
    _validate_source(sample.optical_paths["B02"], patch_id=sample.patch_id, role="optical B02")
    with rasterio.open(sample.optical_paths["B02"]) as grid:
        if grid.crs is None or grid.res[0] <= 0 or grid.res[1] <= 0:
            raise ValueError(f"{sample.patch_id}: invalid reference raster CRS/resolution")
        if (grid.height, grid.width) != (120, 120):
            raise ValueError(f"{sample.patch_id}: B02 common grid must be 120x120")
        for band in OPTICAL_BANDS:
            _validate_source(sample.optical_paths[band], patch_id=sample.patch_id, role=f"optical {band}", grid=grid)
        for band in SAR_BANDS:
            _validate_source(sample.sar_paths[band], patch_id=sample.patch_id, role=f"SAR {band}", grid=grid, exact_grid=True)
        _validate_source(sample.reference_map, patch_id=sample.patch_id, role="reference", grid=grid, exact_grid=True)
        optical = np.stack([_read_to_grid(sample.optical_paths[band], grid, Resampling.bilinear) for band in OPTICAL_BANDS])
        sar = np.stack([_read_to_grid(sample.sar_paths[band], grid, Resampling.bilinear) for band in SAR_BANDS])
        reference_values = _read_to_grid(sample.reference_map, grid, Resampling.nearest)
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
    if not np.isfinite(reference_values).all() or not np.all(reference_values == np.floor(reference_values)):
        raise ValueError(f"{sample.patch_id}: reference must contain finite integer class codes")
    reference = reference_values.astype(np.int64)
    normalized: NormalizedPair = normalize_pair(optical, sar)
    return PreparedSample(sample.patch_id, normalized.optical, normalized.sar, optical, sar, reference, {
        **grid_metadata,
        "optical_band_order": list(OPTICAL_BANDS),
        "sar_band_order": list(SAR_BANDS),
        "optical_normalization": normalized.optical_method,
        "sar_normalization": normalized.sar_method,
        "optical_resampling": "rasterio.warp.reproject bilinear to B02 10 m grid",
        "sar_resampling": "rasterio.warp.reproject bilinear to B02 10 m grid",
        "reference_map_resampling": "not applied; categorical map retained at native 10 m",
        "reference_shape": list(reference.shape),
    })
