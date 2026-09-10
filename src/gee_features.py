from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FeatureExtractionError(ValueError):
    pass


class PhysicalFeatureProvider:
    def extract(self, optical: np.ndarray, sar: np.ndarray, optical_band_order: list[str], sar_band_order: list[str]) -> tuple[np.ndarray, dict]:
        raise NotImplementedError


@dataclass
class LocalRasterFeatureProvider(PhysicalFeatureProvider):
    def extract(self, optical: np.ndarray, sar: np.ndarray, optical_band_order: list[str], sar_band_order: list[str]) -> tuple[np.ndarray, dict]:
        return extract_gee_features(optical, sar, optical_band_order, sar_band_order)


class EarthEngineFeatureProvider(PhysicalFeatureProvider):
    def extract(self, optical: np.ndarray, sar: np.ndarray, optical_band_order: list[str], sar_band_order: list[str]) -> tuple[np.ndarray, dict]:
        raise RuntimeError("EarthEngineFeatureProvider requires a retrieved Earth Engine raster; use LocalRasterFeatureProvider for the local ZIP")


def _valid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values[np.isfinite(values)]


def _stats(values: np.ndarray, prefix: str) -> tuple[dict[str, float], int]:
    valid = _valid(values)
    if valid.size == 0:
        raise FeatureExtractionError(f"No finite pixels available for {prefix}")
    return {
        f"{prefix}_mean": float(valid.mean()),
        f"{prefix}_std": float(valid.std()),
        f"{prefix}_min": float(valid.min()),
        f"{prefix}_max": float(valid.max()),
    }, int(valid.size)


def _index(numerator: np.ndarray, denominator: np.ndarray, name: str) -> tuple[float, int]:
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    if not valid.any():
        return 0.0, 0
    values = numerator[valid] / denominator[valid]
    return float(np.mean(np.clip(values, -1.0, 1.0))), int(valid.sum())


def extract_gee_features(optical: np.ndarray, sar: np.ndarray, optical_band_order: list[str] | tuple[str, ...], sar_band_order: list[str] | tuple[str, ...]) -> tuple[np.ndarray, dict]:
    optical = np.asarray(optical, dtype=np.float32)
    sar = np.asarray(sar, dtype=np.float32)
    if optical.ndim != 3 or sar.ndim != 3 or optical.shape[1:] != sar.shape[1:]:
        raise FeatureExtractionError(f"Optical and SAR grids must be [B,H,W] with matching H,W; got {optical.shape} and {sar.shape}")
    optical_band_order = list(optical_band_order)
    sar_band_order = list(sar_band_order)
    if len(set(optical_band_order)) != len(optical_band_order) or len(set(sar_band_order)) != len(sar_band_order):
        raise FeatureExtractionError("Band orders must not contain duplicates")
    if set(optical_band_order) & set(sar_band_order):
        raise FeatureExtractionError("Optical and SAR band orders must be disjoint")
    if len(optical_band_order) != optical.shape[0] or len(sar_band_order) != sar.shape[0]:
        raise FeatureExtractionError("Band order does not match tensor channel count")
    optical_by_band = {band: optical[i] for i, band in enumerate(optical_band_order)}
    sar_by_band = {band: sar[i] for i, band in enumerate(sar_band_order)}
    values: list[float] = []
    names: list[str] = []
    valid_counts: dict[str, int] = {}
    detail: dict[str, float] = {}
    for band in optical_band_order:
        stats, count = _stats(optical_by_band[band], f"optical_{band}")
        for key, value in stats.items():
            names.append(key); values.append(value); detail[key] = value
        valid_counts[f"optical_{band}"] = count
    for band in sar_band_order:
        stats, count = _stats(sar_by_band[band], f"sar_{band}")
        for key, value in stats.items():
            names.append(key); values.append(value); detail[key] = value
        valid_counts[f"sar_{band}"] = count
    required_optical = {"B03", "B04", "B08", "B11", "B12"}
    required_sar = {"VV", "VH"}
    if not required_optical.issubset(optical_by_band) or not required_sar.issubset(sar_by_band):
        missing = sorted((required_optical - set(optical_by_band)) | (required_sar - set(sar_by_band)))
        raise FeatureExtractionError(f"Cannot compute requested indices; missing bands: {missing}")
    indices = {
        "NDVI": (optical_by_band["B08"] - optical_by_band["B04"], optical_by_band["B08"] + optical_by_band["B04"]),
        "NDWI": (optical_by_band["B03"] - optical_by_band["B08"], optical_by_band["B03"] + optical_by_band["B08"]),
        "MNDWI": (optical_by_band["B03"] - optical_by_band["B11"], optical_by_band["B03"] + optical_by_band["B11"]),
        "NDBI": (optical_by_band["B11"] - optical_by_band["B08"], optical_by_band["B11"] + optical_by_band["B08"]),
        "VV_minus_VH": (sar_by_band["VV"] - sar_by_band["VH"], np.ones_like(sar_by_band["VV"])),
        "VH_over_VV": (sar_by_band["VH"], sar_by_band["VV"]),
    }
    formulas = {
        "NDVI": "(B08-B04)/(B08+B04)", "NDWI": "(B03-B08)/(B03+B08)",
        "MNDWI": "(B03-B11)/(B03+B11)", "NDBI": "(B11-B08)/(B11+B08)",
        "VV_minus_VH": "VV-VH", "VH_over_VV": "VH/VV",
    }
    for name, (numerator, denominator) in indices.items():
        value, count = _index(numerator, denominator, name)
        feature_name = f"index_{name}_mean"
        names.append(feature_name); values.append(value); detail[feature_name] = value
        valid_counts[feature_name] = count
    vector = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise FeatureExtractionError("Derived GEE feature vector contains non-finite values")
    return vector, {"feature_names": names, "feature_values": detail, "valid_pixel_counts": valid_counts, "formulas": formulas, "dimension": int(vector.size)}
