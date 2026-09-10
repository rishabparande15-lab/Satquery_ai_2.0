import numpy as np


def analyze(before: np.ndarray, after: np.ndarray, threshold: float = 0.25) -> dict:
    difference = after.astype(np.float32) - before.astype(np.float32)
    normalized_difference = difference / (np.abs(after.astype(np.float32)) + np.abs(before.astype(np.float32)) + 1e-6)
    mask = np.any(np.abs(normalized_difference) >= threshold, axis=0)
    return {"absolute_difference": difference, "normalized_difference": normalized_difference, "change_mask": mask, "threshold": threshold, "changed_pixel_percentage": float(mask.mean() * 100), "formula": "(after-before)/(abs(after)+abs(before)+1e-6)"}