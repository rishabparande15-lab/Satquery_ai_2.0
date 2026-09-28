"""Single-modal Sentinel-1 VV/VH learned adapter for Qwen."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SAR_BANDS = ("VV", "VH")
SAR_ADAPTER_ID = "phase3h_sar_qwen_token_projector_v1"
QWEN_HIDDEN_SIZE = 2048
QWEN_PATCH_SIZE = 14
QWEN_MERGE_SIZE = 2
QWEN_GRID = 8
QWEN_TOKEN_COUNT = 16


def normalize_sar(sar: np.ndarray, *, nodata: float | None = None) -> np.ndarray:
    """Deterministically scale canonical VV,VH values, preserving channel order."""
    values = np.asarray(sar, dtype=np.float32)
    if values.shape != (2, 120, 120):
        raise ValueError("SAR input must have exact shape [2,120,120] (VV,VH)")
    valid = np.ones(values.shape, dtype=bool) if nodata is None else values != nodata
    if not np.isfinite(values[valid]).all():
        raise ValueError("SAR input contains NaN/Inf outside declared nodata")
    out = np.zeros_like(values)
    for channel in range(2):
        mask = valid[channel]
        if not mask.any():
            continue
        sample = values[channel][mask]
        lo, hi = float(sample.mean() - 2 * sample.std()), float(sample.mean() + 2 * sample.std())
        if hi > lo:
            out[channel] = np.where(mask, np.clip((values[channel] - lo) / (hi - lo), 0, 1), 0)
    return out


@dataclass(frozen=True)
class SARProvenance:
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    channel_order: tuple[str, str] = SAR_BANDS
    preprocessing: str = "per-channel mean +/- 2 std, clipped [0,1], nodata=0"
    scientific_representation_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"adapter_id": SAR_ADAPTER_ID, "input_shape": list(self.input_shape),
                "output_shape": list(self.output_shape), "channel_order": list(self.channel_order),
                "preprocessing": self.preprocessing,
                "scientific_representation_used": self.scientific_representation_used}


class SARProjector(nn.Module):
    """Small raw-SAR projector emitting the verified Qwen post-vision width."""
    def __init__(self, patch_hidden_size: int = 128) -> None:
        super().__init__()
        self.patch_hidden_size = patch_hidden_size
        self.patch_embed = nn.Conv2d(2, patch_hidden_size, 14, 14)
        merged = patch_hidden_size * 4
        self.norm = nn.LayerNorm(merged)
        self.output = nn.Sequential(nn.Linear(merged, QWEN_HIDDEN_SIZE), nn.GELU(),
                                    nn.Linear(QWEN_HIDDEN_SIZE, QWEN_HIDDEN_SIZE))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        if not isinstance(sar, torch.Tensor) or sar.ndim != 4 or tuple(sar.shape[1:]) != (2, 120, 120):
            raise ValueError("SAR projector input must be [batch,2,120,120] (VV,VH)")
        if not torch.isfinite(sar).all():
            raise ValueError("SAR projector input contains non-finite values")
        x = self.patch_embed(F.interpolate(sar, size=(112, 112), mode="bilinear", align_corners=False))
        b, c, h, w = x.shape
        x = x.reshape(b, c, 4, 2, 4, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, 16, c * 4)
        return self.output(self.norm(x))

    def provenance(self, input_shape: tuple[int, ...] = (1, 2, 120, 120)) -> dict[str, Any]:
        return SARProvenance(input_shape, (input_shape[0], QWEN_TOKEN_COUNT, QWEN_HIDDEN_SIZE)).to_dict()

    def fingerprint(self) -> str:
        digest = sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
