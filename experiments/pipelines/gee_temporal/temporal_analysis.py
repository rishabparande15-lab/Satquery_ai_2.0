import numpy as np


def analyze(before_optical: np.ndarray, after_optical: np.ndarray, before_sar: np.ndarray, after_sar: np.ndarray, threshold: float = 0.25) -> dict:
    optical_difference = after_optical.astype(np.float32) - before_optical.astype(np.float32)
    sar_difference = after_sar.astype(np.float32) - before_sar.astype(np.float32)
    ndvi_before = (before_optical[3] - before_optical[2]) / (before_optical[3] + before_optical[2] + 1e-6)
    ndvi_after = (after_optical[3] - after_optical[2]) / (after_optical[3] + after_optical[2] + 1e-6)
    ndvi_difference = ndvi_after - ndvi_before
    change_mask = np.abs(ndvi_difference) >= threshold
    return {"optical_difference": optical_difference, "sar_difference": sar_difference, "ndvi_difference": ndvi_difference, "change_mask": change_mask, "threshold": threshold, "changed_pixel_percentage": float(change_mask.mean() * 100), "statistics": {"optical_difference_mean": float(np.nanmean(optical_difference)), "sar_difference_mean": float(np.nanmean(sar_difference)), "ndvi_difference_mean": float(np.nanmean(ndvi_difference))}}