"""Language-facing Earth-observation VLM adaptation components."""

from .multispectral_projector import (
    QWEN_HIDDEN_SIZE,
    QWEN_IMAGE_GRID_THW,
    S2MultispectralProjector,
    S2_QWEN_ADAPTER_ID,
)
from .temporal_contracts import ContractOnlyTemporalFusionAdapter, TemporalInput, TemporalRepresentation

__all__ = [
    "QWEN_HIDDEN_SIZE",
    "QWEN_IMAGE_GRID_THW",
    "S2MultispectralProjector",
    "S2_QWEN_ADAPTER_ID",
    "ContractOnlyTemporalFusionAdapter",
    "TemporalInput",
    "TemporalRepresentation",
]
