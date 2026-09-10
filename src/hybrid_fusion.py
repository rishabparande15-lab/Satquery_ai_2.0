from __future__ import annotations

import numpy as np
import torch
from torch import nn


class HybridFusion(nn.Module):
    """Modular projection and fusion layer; weights are an untrained prototype."""

    def __init__(self, gee_dim: int, croma_dim: int, gee_projection: int = 64, croma_projection: int = 128, output_dim: int | None = None):
        super().__init__()
        output_dim = output_dim or gee_projection + croma_projection
        self.gee_projection = nn.Sequential(nn.LayerNorm(gee_dim), nn.Linear(gee_dim, gee_projection), nn.GELU())
        self.croma_projection = nn.Sequential(nn.LayerNorm(croma_dim), nn.Linear(croma_dim, croma_projection), nn.GELU())
        self.output = nn.Sequential(nn.Linear(gee_projection + croma_projection, output_dim), nn.LayerNorm(output_dim), nn.GELU())
        self.output_dim = output_dim

    def forward(self, gee: torch.Tensor, croma: torch.Tensor) -> torch.Tensor:
        return self.output(torch.cat([self.gee_projection(gee), self.croma_projection(croma)], dim=-1))


class FeatureModeFusion(nn.Module):
    """Common output contract for gee_only, croma_only, and hybrid experiments."""

    def __init__(self, mode: str, gee_dim: int, croma_dim: int, output_dim: int = 192):
        super().__init__()
        if mode not in {"gee_only", "croma_only", "hybrid"}:
            raise ValueError(f"Unsupported feature mode: {mode}")
        self.mode = mode
        input_dim = gee_dim if mode == "gee_only" else croma_dim if mode == "croma_only" else gee_dim + croma_dim
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim), nn.GELU())
        self.output_dim = output_dim

    def forward(self, gee: torch.Tensor, croma: torch.Tensor) -> torch.Tensor:
        if self.mode == "gee_only":
            inputs = gee
        elif self.mode == "croma_only":
            inputs = croma
        else:
            inputs = torch.cat([gee, croma], dim=-1)
        return self.projection(inputs)


class LandCoverClassificationHead(nn.Module):
    def __init__(self, input_dim: int, class_count: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, class_count)

    def forward(self, hybrid: torch.Tensor) -> torch.Tensor:
        return self.classifier(hybrid)

    def probabilities(self, hybrid: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(hybrid))

    @staticmethod
    def loss(logits: torch.Tensor, multi_hot_labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, multi_hot_labels.float())


class ChangeAnalysisHead(nn.Module):
    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if before.shape != after.shape:
            raise ValueError(f"Before/after hybrid representations must match: {before.shape} vs {after.shape}")
        return after - before


def pooled_croma_features(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    required = ("optical_GAP", "SAR_GAP", "joint_GAP")
    missing = [key for key in required if key not in outputs]
    if missing:
        raise ValueError(f"CROMA output missing pooled representations: {missing}")
    arrays = [outputs[key].detach().cpu().numpy().reshape(-1).astype(np.float32) for key in required]
    vector = np.concatenate(arrays)
    if vector.size != 2304 or not np.isfinite(vector).all():
        raise ValueError(f"Unexpected pooled CROMA feature vector: {vector.shape}")
    return vector
