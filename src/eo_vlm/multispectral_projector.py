"""Learned 12-band S2 to Qwen visual-token projection.

This is a language-facing adapter only.  It does not import or alter the
scientific predictor, CROMA representations, or Pipeline-3 preprocessing.
The projector emits the exact hidden width and merged token geometry exposed
by the pinned Qwen2.5-VL-3B checkpoint; the Qwen base model remains frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..eo_vlm_adapter import S2_BANDS


S2_QWEN_ADAPTER_ID = "phase3g_s2_qwen_token_projector_v1"
QWEN_HIDDEN_SIZE = 2048
QWEN_VISION_HIDDEN_SIZE = 1280
QWEN_PATCH_SIZE = 14
QWEN_SPATIAL_MERGE_SIZE = 2
QWEN_PROJECTOR_GRID = 8
QWEN_IMAGE_TOKEN_COUNT = (QWEN_PROJECTOR_GRID // QWEN_SPATIAL_MERGE_SIZE) ** 2
QWEN_IMAGE_GRID_THW = (1, QWEN_PROJECTOR_GRID, QWEN_PROJECTOR_GRID)


@dataclass(frozen=True)
class ProjectorProvenance:
    adapter_id: str
    model_id: str
    checkpoint_revision: str
    canonical_band_order: tuple[str, ...]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    qwen_hidden_size: int
    qwen_patch_size: int
    qwen_spatial_merge_size: int
    representation: str
    scientific_representation_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "model_id": self.model_id,
            "checkpoint_revision": self.checkpoint_revision,
            "canonical_band_order": list(self.canonical_band_order),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "qwen_hidden_size": self.qwen_hidden_size,
            "qwen_patch_size": self.qwen_patch_size,
            "qwen_spatial_merge_size": self.qwen_spatial_merge_size,
            "representation": self.representation,
            "scientific_representation_used": self.scientific_representation_used,
        }


class S2MultispectralProjector(nn.Module):
    """Project canonical S2 directly into Qwen-compatible visual tokens.

    The 120x120 canonical grid is deterministically resized to the largest
    Qwen patch-aligned 112x112 grid.  A learned 12-channel patch embedding,
    2x2 spatial merge, and output projection produce 16 tokens x 2048, which
    matches Qwen2.5-VL-3B's `out_hidden_size=2048` and merge geometry.  This is
    not RGB conversion: every canonical band is an input channel and the
    target is language loss against the real annotation.
    """

    input_bands = len(S2_BANDS)
    output_hidden_size = QWEN_HIDDEN_SIZE
    token_count = QWEN_IMAGE_TOKEN_COUNT

    def __init__(self, *, patch_hidden_size: int = 128) -> None:
        super().__init__()
        if patch_hidden_size <= 0:
            raise ValueError("patch_hidden_size must be positive")
        self.patch_hidden_size = int(patch_hidden_size)
        self.patch_embed = nn.Conv2d(
            self.input_bands,
            self.patch_hidden_size,
            kernel_size=QWEN_PATCH_SIZE,
            stride=QWEN_PATCH_SIZE,
            bias=True,
        )
        self.norm = nn.LayerNorm(self.patch_hidden_size * QWEN_SPATIAL_MERGE_SIZE**2)
        self.output = nn.Sequential(
            nn.Linear(self.patch_hidden_size * QWEN_SPATIAL_MERGE_SIZE**2, QWEN_HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(QWEN_HIDDEN_SIZE, QWEN_HIDDEN_SIZE),
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, s2: torch.Tensor) -> torch.Tensor:
        if not isinstance(s2, torch.Tensor) or s2.ndim != 4:
            raise ValueError("S2 projector input must be [batch,12,120,120]")
        if tuple(s2.shape[1:]) != (len(S2_BANDS), 120, 120):
            raise ValueError("S2 projector input must have shape [batch,12,120,120]")
        if not torch.isfinite(s2).all():
            raise ValueError("S2 projector input contains non-finite values")
        # Qwen's patch and 2x2 merge geometry is explicit in the pinned config.
        aligned = F.interpolate(s2, size=(112, 112), mode="bilinear", align_corners=False)
        patches = self.patch_embed(aligned)  # [B, patch_hidden, 8, 8]
        batch, channels, height, width = patches.shape
        if (height, width) != (QWEN_PROJECTOR_GRID, QWEN_PROJECTOR_GRID):
            raise RuntimeError("unexpected Qwen-aligned patch grid")
        merged = patches.reshape(
            batch,
            channels,
            height // QWEN_SPATIAL_MERGE_SIZE,
            QWEN_SPATIAL_MERGE_SIZE,
            width // QWEN_SPATIAL_MERGE_SIZE,
            QWEN_SPATIAL_MERGE_SIZE,
        ).permute(0, 2, 4, 1, 3, 5).reshape(batch, self.token_count, -1)
        return self.output(self.norm(merged))

    def provenance(self, *, model_id: str, checkpoint_revision: str, input_shape: tuple[int, ...]) -> dict[str, Any]:
        output_shape = (input_shape[0], self.token_count, QWEN_HIDDEN_SIZE)
        return ProjectorProvenance(
            adapter_id=S2_QWEN_ADAPTER_ID,
            model_id=model_id,
            checkpoint_revision=checkpoint_revision,
            canonical_band_order=S2_BANDS,
            input_shape=input_shape,
            output_shape=output_shape,
            qwen_hidden_size=QWEN_HIDDEN_SIZE,
            qwen_patch_size=QWEN_PATCH_SIZE,
            qwen_spatial_merge_size=QWEN_SPATIAL_MERGE_SIZE,
            representation="learned_12_band_patch_tokens",
            scientific_representation_used=False,
        ).to_dict()

    def fingerprint(self) -> str:
        digest = sha256()
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
