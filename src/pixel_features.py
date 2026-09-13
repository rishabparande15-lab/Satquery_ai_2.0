"""Research-only pixel/local/token features for the Pass 5B feasibility probe."""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def safe_normalized_difference(a: np.ndarray, b: np.ndarray, *, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    """Return (a-b)/(a+b), with invalid/near-zero denominators masked as NaN."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(a + b) > eps)
    out = np.full(np.broadcast_shapes(a.shape, b.shape), np.nan, dtype=np.float32)
    np.divide(a - b, a + b, out=out, where=valid)
    return out, valid


def candidate_maps(optical: np.ndarray, sar: np.ndarray, optical_order, sar_order) -> dict[str, np.ndarray]:
    """Compute scientifically interpretable maps from raw, co-registered inputs."""
    o = {name: np.asarray(optical[i], dtype=np.float32) for i, name in enumerate(optical_order)}
    s = {name: np.asarray(sar[i], dtype=np.float32) for i, name in enumerate(sar_order)}
    required = {"B02", "B03", "B04", "B08", "B11", "B12"}
    if not required <= o.keys() or not {"VV", "VH"} <= s.keys():
        raise ValueError("Required optical/SAR bands are absent")
    ndvi, _ = safe_normalized_difference(o["B08"], o["B04"])
    ndwi, _ = safe_normalized_difference(o["B03"], o["B08"])
    mndwi, _ = safe_normalized_difference(o["B03"], o["B11"])
    ndbi, _ = safe_normalized_difference(o["B11"], o["B08"])
    # Bare Soil Index: (SWIR1+red - NIR-blue)/(SWIR1+red + NIR+blue).
    x, y = o["B11"] + o["B04"], o["B08"] + o["B02"]
    bsi, _ = safe_normalized_difference(x, y)
    vv_vh = s["VV"] - s["VH"]  # dB difference; equivalent to log-domain ratio.
    return {"B04": o["B04"], "B08": o["B08"], "NDVI": ndvi, "NDWI": ndwi,
            "MNDWI": mndwi, "NDBI": ndbi, "BSI": bsi, "VV": s["VV"], "VH": s["VH"],
            "VV_minus_VH": vv_vh.astype(np.float32)}


def local_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware centered local mean/std using reflect padding."""
    if window not in (3, 5, 7) or window % 2 == 0:
        raise ValueError("window must be one of 3, 5, 7")
    x = np.asarray(values, dtype=np.float32)
    pad = window // 2
    padded = np.pad(x, pad, mode="reflect")
    views = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    with np.errstate(invalid="ignore"):
        return np.nanmean(views, axis=(-2, -1)).astype(np.float32), np.nanstd(views, axis=(-2, -1)).astype(np.float32)


def edge_strength(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    gy, gx = np.gradient(x)
    return np.hypot(gx, gy).astype(np.float32)


def aggregate_tokens(values: np.ndarray, *, grid_size: int = 15) -> tuple[np.ndarray, list[str]]:
    """Aggregate a north-up 120x120 map into row-major 15x15 CROMA blocks."""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or x.shape[0] % grid_size:
        raise ValueError("map must be square and evenly divisible by grid_size")
    block = x.shape[0] // grid_size
    blocks = x.reshape(grid_size, block, grid_size, block).transpose(0, 2, 1, 3).reshape(grid_size * grid_size, -1)
    valid = np.isfinite(blocks)
    safe = np.where(valid, blocks, np.nan)
    with np.errstate(invalid="ignore"):
        result = np.column_stack((np.nanmean(safe, 1), np.nanstd(safe, 1), np.nanmin(safe, 1),
                                  np.nanmax(safe, 1), np.nanmedian(safe, 1),
                                  np.nanpercentile(safe, 25, axis=1), np.nanpercentile(safe, 75, axis=1), valid.mean(1)))
    return result.astype(np.float32), ["mean", "std", "min", "max", "median", "p25", "p75", "valid_ratio"]


def map_statistics(values: np.ndarray) -> dict:
    x = np.asarray(values)
    valid = x[np.isfinite(x)]
    if not valid.size:
        raise ValueError("map contains no valid pixels")
    return {"shape": list(x.shape), "dtype": str(x.dtype), "min": float(valid.min()), "max": float(valid.max()),
            "mean": float(valid.mean()), "std": float(valid.std()),
            "percentiles": {str(q): float(np.percentile(valid, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
            "invalid_pixel_count": int(x.size - valid.size)}
