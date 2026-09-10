import numpy as np

from src.croma_adapter import CROMAAdapter


def validate_inputs(optical: np.ndarray, sar: np.ndarray) -> dict:
    if optical.shape != (12, 120, 120) or sar.shape != (2, 120, 120):
        raise ValueError(f"Expected optical [12,120,120] and SAR [2,120,120], got {optical.shape} and {sar.shape}")
    return {"optical_shape": list(optical.shape), "sar_shape": list(sar.shape), "finite": bool(np.isfinite(optical).all() and np.isfinite(sar).all()), "croma_normalization": "Official adapter applies raw-channel mean +/- 2 std clipped to [0,1]."}


def infer(adapter: CROMAAdapter, optical: np.ndarray, sar: np.ndarray):
    validate_inputs(optical, sar)
    return adapter.infer(optical, sar)
