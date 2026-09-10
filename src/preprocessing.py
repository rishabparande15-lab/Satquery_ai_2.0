from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizedPair:
    optical: np.ndarray
    sar: np.ndarray
    optical_method: str
    sar_method: str


def _robust_channel_scale(array: np.ndarray) -> np.ndarray:
    output = np.empty_like(array, dtype=np.float32)
    for channel in range(array.shape[0]):
        values = array[channel]
        finite = np.isfinite(values)
        if not finite.any():
            output[channel] = 0.0
            continue
        valid = values[finite]
        center = float(valid.mean())
        spread = float(valid.std())
        lower = center - 2.0 * spread
        upper = center + 2.0 * spread
        if upper <= lower:
            output[channel] = 0.0
            output[channel][finite] = 0.0
            continue
        scaled = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
        output[channel] = np.where(finite, scaled, 0.0).astype(np.float32)
    return output


def normalize_pair(optical: np.ndarray, sar: np.ndarray) -> NormalizedPair:
    return NormalizedPair(
        optical=_robust_channel_scale(optical.astype(np.float32, copy=False)),
        sar=_robust_channel_scale(sar.astype(np.float32, copy=False)),
        optical_method="per-channel mean +/- 2 std, clipped to [0, 1]; nodata=0",
        sar_method="per-channel mean +/- 2 std, clipped to [0, 1]; nodata=0",
    )