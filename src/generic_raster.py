"""Bounded, provenance-preserving inspection for external imagery.

This module reports source facts only. It never identifies a sensor from
pixels, changes a grid, repairs values, reorders bands, or builds a model
tensor. PNG/JPEG support exists solely to describe an explicit RGB upload.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
import rasterio


RASTER_SUFFIXES = frozenset({".tif", ".tiff"})
RGB_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
ALLOWED_SUFFIXES = RASTER_SUFFIXES | RGB_IMAGE_SUFFIXES
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_BAND_PIXELS = 4_000_000
MAX_BANDS = 32


class GenericRasterInspectionError(ValueError):
    """A stable, user-safe external-raster inspection failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _check_file(path: Path) -> None:
    if not path.is_file():
        raise GenericRasterInspectionError("RASTER_MISSING", "The uploaded file is unavailable.")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise GenericRasterInspectionError("UNSUPPORTED_FILE_FORMAT", "Supported external inputs are TIFF, PNG, and JPEG files.")
    if path.stat().st_size < 1:
        raise GenericRasterInspectionError("INVALID_RASTER", "The uploaded file is empty.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise GenericRasterInspectionError("RASTER_FILE_TOO_LARGE", f"The uploaded file exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MiB safety limit.")


def _finite_summary(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    raw = np.asarray(values)
    effective_mask = np.zeros(raw.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    unmasked = ~effective_mask
    nonfinite = np.logical_and(unmasked, ~np.isfinite(raw))
    valid = raw[unmasked]
    finite = valid[np.isfinite(valid)]
    return {
        "bounded_full_read": True,
        "total_band_pixels": int(raw.size), "valid_unmasked_values": int(unmasked.sum()),
        "masked_or_nodata_values": int(effective_mask.sum()),
        "nonfinite_unmasked_values": int(nonfinite.sum()),
        "all_unmasked_values_finite": int(nonfinite.sum()) == 0,
        "all_unmasked_values_zero": bool(finite.size and np.all(finite == 0)),
        "all_unmasked_values_constant": bool(finite.size and np.all(finite == finite[0])),
        "finite_min": None if not finite.size else float(np.min(finite)),
        "finite_max": None if not finite.size else float(np.max(finite)),
    }


def _band_statistics(values: np.ndarray, mask: np.ndarray | None = None, *, channel_axis: int = 0) -> list[dict[str, Any]]:
    """Return bounded, source-measurement statistics; never semantic evidence."""
    original = np.asarray(values)
    raw = np.moveaxis(original, channel_axis, 0)
    original_mask = np.zeros(original.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    masked = np.moveaxis(original_mask, channel_axis, 0)
    result = []
    for index, (band, band_mask) in enumerate(zip(raw, masked), start=1):
        total = int(band.size)
        unmasked = ~band_mask
        finite = band[unmasked & np.isfinite(band)]
        values_dict = {"band_index": index, "sample_count": total,
                       "nodata_fraction": float(band_mask.sum() / total),
                       "finite_fraction": float(finite.size / total)}
        if finite.size:
            percentiles = np.percentile(finite, [2, 50, 98])
            values_dict.update({"min": float(np.min(finite)), "max": float(np.max(finite)),
                                "mean": float(np.mean(finite)), "std": float(np.std(finite)),
                                "percentiles": {"p02": float(percentiles[0]), "p50": float(percentiles[1]), "p98": float(percentiles[2])}})
        else:
            values_dict.update({"min": None, "max": None, "mean": None, "std": None,
                                "percentiles": {"p02": None, "p50": None, "p98": None}})
        result.append(values_dict)
    return result


def _inspect_rgb_image(candidate: Path) -> dict[str, Any]:
    try:
        with Image.open(candidate) as image:
            image.verify()
        with Image.open(candidate) as image:
            width, height = image.size
            if width < 1 or height < 1:
                raise GenericRasterInspectionError("INVALID_RASTER_DIMENSIONS", "Image width and height must be positive.")
            if width * height > MAX_BAND_PIXELS:
                raise GenericRasterInspectionError("RASTER_INSPECTION_LIMIT", f"Image exceeds the {MAX_BAND_PIXELS} pixel safety limit. No downsampling was performed.")
            channels = {"RGB": 3, "RGBA": 4}.get(image.mode)
            if channels is None:
                raise GenericRasterInspectionError("UNSUPPORTED_BAND_CONFIGURATION", "Generic RGB scene description accepts explicit RGB or RGBA PNG/JPEG inputs only.")
            values = np.asarray(image)
            return {
                "readable": True, "input_kind": "RGB_IMAGE", "file_name": candidate.name,
                "file_suffix": candidate.suffix.lower(), "file_size_bytes": candidate.stat().st_size,
                "sha256": _digest(candidate), "driver": image.format or candidate.suffix.lstrip(".").upper(),
                "width": width, "height": height, "shape": [height, width], "band_count": channels,
                "dtypes": [str(values.dtype)] * channels, "crs": None, "transform": None,
                "resolution": None, "bounds": None, "nodata": [None] * channels,
                "band_descriptions": ["R", "G", "B", "A"][:channels],
                "color_interpretation": ["red", "green", "blue", "alpha"][:channels],
                "finite_value_checks": _finite_summary(values),
                "band_statistics": _band_statistics(values, channel_axis=-1),
            }
    except GenericRasterInspectionError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise GenericRasterInspectionError("INVALID_RASTER", "The image is corrupt, unsupported, or unreadable.") from exc


def _inspect_geotiff(candidate: Path) -> dict[str, Any]:
    try:
        with rasterio.open(candidate) as dataset:
            if dataset.width < 1 or dataset.height < 1 or dataset.count < 1:
                raise GenericRasterInspectionError("INVALID_RASTER_DIMENSIONS", "Raster width, height, and band count must be positive.")
            if dataset.count > MAX_BANDS:
                raise GenericRasterInspectionError("UNSUPPORTED_BAND_CONFIGURATION", f"Raster has {dataset.count} bands; the bounded inspector limit is {MAX_BANDS} bands.")
            band_pixels = dataset.width * dataset.height * dataset.count
            if band_pixels > MAX_BAND_PIXELS:
                raise GenericRasterInspectionError(
                    "RASTER_INSPECTION_LIMIT",
                    f"Raster has {band_pixels} band-pixels; the bounded inspector limit is {MAX_BAND_PIXELS}. No resampling was performed.",
                )
            values = dataset.read(out_dtype="float32", masked=True)
            summary = _finite_summary(values.data, np.ma.getmaskarray(values))
            resolution = list(dataset.res)
            if any(not np.isfinite(item) or item <= 0 for item in resolution):
                raise GenericRasterInspectionError("INVALID_RASTER", "Raster resolution must contain finite positive values.")
            bounds = list(dataset.bounds)
            if not all(np.isfinite(item) for item in bounds) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                raise GenericRasterInspectionError("INVALID_RASTER", "Raster bounds must be finite and non-empty.")
            return {
                "readable": True, "input_kind": "RASTER",
                "file_name": candidate.name,
                "file_suffix": candidate.suffix.lower(),
                "file_size_bytes": candidate.stat().st_size,
                "sha256": _digest(candidate),
                "driver": dataset.driver,
                "width": dataset.width,
                "height": dataset.height,
                "shape": [dataset.height, dataset.width],
                "band_count": dataset.count,
                "dtypes": list(dataset.dtypes),
                "crs": str(dataset.crs) if dataset.crs else None,
                "transform": list(dataset.transform),
                "resolution": resolution,
                "bounds": bounds,
                "nodata": [None if value is None or not np.isfinite(value) else float(value) for value in dataset.nodatavals],
                "band_descriptions": [item if item is not None else "" for item in dataset.descriptions],
                "color_interpretation": [item.name for item in dataset.colorinterp],
                "finite_value_checks": summary,
                "band_statistics": _band_statistics(values.data, np.ma.getmaskarray(values)),
            }
    except GenericRasterInspectionError:
        raise
    except (rasterio.errors.RasterioError, OSError, ValueError) as exc:
        raise GenericRasterInspectionError("INVALID_RASTER", "The raster is corrupt, unsupported, or unreadable.") from exc


def inspect_generic_raster(path: str | Path) -> dict[str, Any]:
    """Inspect a bounded TIFF/PNG/JPEG upload without inferring sensor semantics."""
    candidate = Path(path)
    _check_file(candidate)
    if candidate.suffix.lower() in RGB_IMAGE_SUFFIXES:
        return _inspect_rgb_image(candidate)
    return _inspect_geotiff(candidate)
