"""Deterministic image-region to CROMA token-grid mapping.

This is a data foundation, not a learned grounding model. Coordinates use the
half-open pixel convention ``[x_min, x_max) x [y_min, y_max)``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def box_to_token_overlaps(box: Box, *, image_size: int = 120, grid_size: int = 15) -> dict[int, float]:
    """Return row-major token indices and fraction of each token covered."""
    values = (box.x_min, box.y_min, box.x_max, box.y_max)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Box coordinates must be finite")
    if box.x_max <= box.x_min or box.y_max <= box.y_min:
        raise ValueError("Box must have positive width and height")
    if image_size <= 0 or grid_size <= 0 or image_size % grid_size:
        raise ValueError("Image size must be evenly divisible by grid size")
    x0, y0 = max(0.0, box.x_min), max(0.0, box.y_min)
    x1, y1 = min(float(image_size), box.x_max), min(float(image_size), box.y_max)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Box does not overlap the image")
    token_size = image_size / grid_size
    col_start = max(0, int(math.floor(x0 / token_size)))
    row_start = max(0, int(math.floor(y0 / token_size)))
    col_end = min(grid_size - 1, int(math.ceil(x1 / token_size) - 1))
    row_end = min(grid_size - 1, int(math.ceil(y1 / token_size) - 1))
    overlaps: dict[int, float] = {}
    token_area = token_size * token_size
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            left, top = col * token_size, row * token_size
            right, bottom = left + token_size, top + token_size
            width = max(0.0, min(x1, right) - max(x0, left))
            height = max(0.0, min(y1, bottom) - max(y0, top))
            if width and height:
                overlaps[row * grid_size + col] = width * height / token_area
    return overlaps


def box_to_token_indices(box: Box, *, image_size: int = 120, grid_size: int = 15) -> list[int]:
    return list(box_to_token_overlaps(box, image_size=image_size, grid_size=grid_size))
