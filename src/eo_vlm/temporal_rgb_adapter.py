"""RGB pair packaging at the temporal language-execution boundary.

This module deliberately preserves two RGB observations.  It performs no
registration, fusion, difference calculation, grounding, or learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

import numpy as np

from ..eo_vlm_adapter import EOInput, PreparedEOInput, Qwen25VLRGBAdapter, RemoteSensingVisionAdapter
from .temporal_contracts import (
    UNKNOWN, SpatialCorrespondenceStatus, TemporalInput, TemporalModality,
    TemporalOrderStatus,
)


TEMPORAL_RGB_ADAPTER_ID = "phase3j5_temporal_rgb_adapter_v1"
FIXED_TEMPORAL_RGB_PROMPT = (
    "Compare the two remote-sensing images and describe the observable differences "
    "between the before image and the after image. Only report differences that are "
    "visually supported by the supplied pair. Do not invent objects, counts, "
    "measurements, dates, or causes."
)


@dataclass(frozen=True)
class PreparedTemporalRGBInput:
    """Two independently prepared RGB images in documented dataset order."""

    temporal_input: TemporalInput
    model_input: tuple[np.ndarray, np.ndarray]
    adapter_id: str
    prompt: str
    preprocessing: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def as_qwen_input(self) -> PreparedEOInput:
        """Expose the existing Qwen RGB request shape without reprocessing pixels."""
        return PreparedEOInput(
            modality="TEMPORAL_RGB_PAIR",
            model_input=self.model_input,
            adapter_id=self.adapter_id,
            preprocessing=self.preprocessing,
            provenance=self.provenance,
        )


class TemporalRGBAdapter(RemoteSensingVisionAdapter):
    """Model-swappable RGB-pair adapter, backed by a single-image RGB adapter."""

    adapter_id = TEMPORAL_RGB_ADAPTER_ID

    def __init__(self, vision_adapter: RemoteSensingVisionAdapter | None = None) -> None:
        self.vision_adapter = vision_adapter or Qwen25VLRGBAdapter()

    def validate_temporal_rgb(self, temporal_input: TemporalInput, t1: np.ndarray, t2: np.ndarray) -> None:
        if not isinstance(temporal_input, TemporalInput):
            raise ValueError("temporal_input must be TemporalInput")
        if temporal_input.t1.modality is not TemporalModality.OPTICAL or temporal_input.t2.modality is not TemporalModality.OPTICAL:
            raise ValueError("temporal RGB adapter requires two optical RGB frames")
        if temporal_input.t1.reference_id == temporal_input.t2.reference_id:
            raise ValueError("temporal RGB pair requires distinct T1/T2 identities")
        self.vision_adapter.validate_inputs(EOInput("OPTICAL_RGB", optical=t1))
        self.vision_adapter.validate_inputs(EOInput("OPTICAL_RGB", optical=t2))

    def prepare_temporal_rgb(self, temporal_input: TemporalInput, t1: np.ndarray, t2: np.ndarray,
                             *, prompt: str = FIXED_TEMPORAL_RGB_PROMPT) -> PreparedTemporalRGBInput:
        self.validate_temporal_rgb(temporal_input, t1, t2)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        first = self.vision_adapter.prepare_inputs(EOInput("OPTICAL_RGB", optical=t1))
        second = self.vision_adapter.prepare_inputs(EOInput("OPTICAL_RGB", optical=t2))
        images = (np.asarray(first.model_input, dtype=np.uint8), np.asarray(second.model_input, dtype=np.uint8))
        pair_digest = sha256(images[0].tobytes() + images[1].tobytes()).hexdigest()
        pair_attributes = temporal_input.provenance.attributes if temporal_input.provenance else {}
        provenance = {
            "adapter_id": self.adapter_id,
            "vision_adapter_id": first.adapter_id,
            "temporal_input_fingerprint": temporal_input.fingerprint(),
            "pair_input_sha256": pair_digest,
            "t1_reference_id": temporal_input.t1.reference_id,
            "t2_reference_id": temporal_input.t2.reference_id,
            "temporal_order": temporal_input.temporal_order.value,
            "timestamp_status": UNKNOWN if temporal_input.temporal_metadata.t1_timestamp is None else "SUPPLIED",
            "spatial_correspondence_status": temporal_input.spatial_status.value,
            "registration_status": pair_attributes.get("registration_status", UNKNOWN),
            "scientific_representation_used": False,
            "learned_temporal_fusion": False,
        }
        return PreparedTemporalRGBInput(
            temporal_input=temporal_input, model_input=images, adapter_id=self.adapter_id,
            prompt=prompt.strip(), preprocessing={"operation": "per-image identity_uint8_rgb", "image_count": 2},
            provenance=provenance,
        )

    # Generic vision-adapter aliases retain a model-swappable surface.
    def validate_inputs(self, request: EOInput) -> None:
        self.vision_adapter.validate_inputs(request)

    def prepare_inputs(self, request: EOInput) -> PreparedEOInput:
        return self.vision_adapter.prepare_inputs(request)

    def get_capabilities(self) -> Mapping[str, Any]:
        return {"rgb_temporal_pair": "IMPLEMENTED", "image_count": 2,
                "learned_temporal_fusion": False, "registration": "NOT_PERFORMED",
                "grounding": "NOT_IMPLEMENTED", "supported_modality": "optical_rgb"}

    def get_provenance(self, request: EOInput | None = None, prepared: np.ndarray | None = None) -> Mapping[str, Any]:
        return {"adapter_id": self.adapter_id, "implementation": "pair_packaging_only",
                "vision_adapter": type(self.vision_adapter).__name__, "scientific_representation_used": False}

    def estimate_resources(self, temporal_input: TemporalInput | None = None, **kwargs: Any) -> Mapping[str, Any]:
        if temporal_input is not None and not isinstance(temporal_input, TemporalInput):
            raise ValueError("temporal_input must be TemporalInput")
        estimate = self.vision_adapter.estimate_resources(image_count=2, **kwargs)
        return {**estimate, "estimated_image_count": 2, "temporal_fusion": "NOT_COMPUTED",
                "registration": "NOT_PERFORMED"}

    def generate(self, runtime: Mapping[str, Any], prepared: PreparedTemporalRGBInput,
                 *, max_new_tokens: int = 32) -> Mapping[str, Any]:
        """Delegate two already-prepared RGB images to the Qwen-compatible runtime."""
        if not isinstance(self.vision_adapter, Qwen25VLRGBAdapter):
            raise RuntimeError("configured vision adapter does not provide Qwen generation")
        return self.vision_adapter.generate(runtime, prepared.as_qwen_input(), prepared.prompt, max_new_tokens=max_new_tokens)
