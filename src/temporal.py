from __future__ import annotations

import numpy as np
from datetime import date


class TemporalPairError(ValueError):
    pass


def validate_temporal_pair(before: dict, after: dict) -> None:
    if not before.get("sample_id") or not after.get("sample_id"):
        raise TemporalPairError("Both temporal samples require sample_id values")
    if before["sample_id"] == after["sample_id"]:
        raise TemporalPairError("Before and after samples must be different")
    if not before.get("acquisition_date") or not after.get("acquisition_date") or before["acquisition_date"] >= after["acquisition_date"]:
        raise TemporalPairError("Temporal metadata must contain ordered before/after acquisition dates")
    try:
        date.fromisoformat(before["acquisition_date"])
        date.fromisoformat(after["acquisition_date"])
    except (ValueError, TypeError):
        raise TemporalPairError("Invalid temporal acquisition dates") from None
    from .input_validation import check_grid
    required = {"crs", "resolution", "bounds", "transform", "shape"}
    if not required.issubset(before) or not required.issubset(after) or not before["crs"] or not after["crs"]:
        raise TemporalPairError("Temporal pairing requires explicit geospatial metadata")
    if not all(check_grid(before, after).values()):
        raise TemporalPairError("Temporal rasters are not spatially corresponding")


def compare_representations(before: np.ndarray, after: np.ndarray) -> dict:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape:
        raise TemporalPairError(f"Representation shapes differ: {before.shape} vs {after.shape}")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise TemporalPairError("Temporal representations must be finite")
    if not before.size:
        raise TemporalPairError("Temporal representations must not be empty")
    difference = after - before
    distance = float(np.linalg.norm(difference))
    average = float(np.mean(np.abs(difference)))
    if not np.isfinite(distance) or not np.isfinite(average):
        raise TemporalPairError("Temporal difference exceeded numerical range")
    return {"l2_distance": distance, "mean_absolute_difference": average, "dimension": int(before.size)}
