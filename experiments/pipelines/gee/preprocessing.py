import numpy as np


def difference_and_ratio(before: np.ndarray, after: np.ndarray) -> dict:
    if before.shape != after.shape:
        raise ValueError(f"Before/after shapes differ: {before.shape} vs {after.shape}")
    difference = after.astype(np.float32) - before.astype(np.float32)
    ratio = after.astype(np.float32) / (np.abs(before.astype(np.float32)) + 1e-6)
    return {"difference": difference, "ratio": ratio}


def normalized_display(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0, 1).astype(np.float32)
