"""A small, explicit data-cube contract used by analysis services."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MultimodalCube:
    optical: np.ndarray | None = None
    sar: np.ndarray | None = None
    before: np.ndarray | None = None
    after: np.ndarray | None = None
    dem: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    aoi: dict[str, Any] | None = None
    quality_masks: dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self) -> None:
        arrays = [item for item in (self.optical, self.sar, self.before, self.after, self.dem) if item is not None]
        if not arrays:
            raise ValueError("Cube requires at least one raster")
        if (self.before is None) != (self.after is None):
            raise ValueError("Temporal cube requires both before and after")
        if self.before is not None:
            if self.before.shape != self.after.shape:
                raise ValueError("Before/after arrays must be spatially aligned")
            raise ValueError("Temporal cube requires a validated temporal metadata contract; currently unavailable")
        if self.optical is not None and self.optical.ndim != 3:
            raise ValueError("Optical cube data must be [bands, height, width]")
        if self.sar is not None and self.sar.ndim != 3:
            raise ValueError("SAR cube data must be [bands, height, width]")
        if self.optical is not None and self.sar is not None and self.optical.shape[1:] != self.sar.shape[1:]:
            raise ValueError("Optical and SAR grids must be spatially aligned")
        if any(not np.isfinite(item).all() for item in arrays):
            raise ValueError("Cube contains non-finite values")
        for name, expected in (("optical", 12), ("sar", 2)):
            array = getattr(self, name)
            if array is not None and (array.shape[0] != expected or min(array.shape[1:]) < 1):
                raise ValueError(f"{name} cube has invalid band count or empty dimensions")
        if self.optical is not None and self.sar is not None:
            from .input_validation import check_grid
            if not all(key in self.metadata for key in ("optical", "sar")):
                raise ValueError("Joint cube requires per-modality geospatial metadata")
            if not all(check_grid(self.metadata["optical"], self.metadata["sar"]).values()):
                raise ValueError("Joint cube metadata is not spatially aligned")
        for name, mask in self.quality_masks.items():
            if mask.shape != arrays[0].shape[-2:]:
                raise ValueError(f"Quality mask {name} has incompatible dimensions")

    def summary(self) -> dict[str, Any]:
        self.validate()
        return {"modalities": [name for name, value in (("optical", self.optical), ("sar", self.sar), ("before", self.before), ("after", self.after), ("dem", self.dem)) if value is not None],
                "spatial_dimensions": list(self.optical.shape[1:] if self.optical is not None else self.sar.shape[1:]) if (self.optical is not None or self.sar is not None) else None,
                "metadata": self.metadata, "aoi": self.aoi, "quality_masks": list(self.quality_masks)}
