"""Isolated Phase 3D input adapter contract for a local EO-VLM smoke test.

This module deliberately does not import Transformers, load weights, or touch
the scientific Pipeline 3. It validates and packages model-facing inputs so a
future concrete runtime can be swapped behind the same boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .preprocessing import _robust_channel_scale


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_ID = "phase3d_qwen25vl_rgb_boundary_v1"
S2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
S1_BANDS = ("VV", "VH")


@dataclass(frozen=True)
class EOInput:
    modality: str
    optical: np.ndarray | None = None
    sar: np.ndarray | None = None
    t1: np.ndarray | None = None
    t2: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PreparedEOInput:
    modality: str
    model_input: np.ndarray | tuple[np.ndarray, ...]
    adapter_id: str
    preprocessing: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class InputContract:
    name: str
    status: str
    shape: tuple[int, ...] | None
    semantics: Mapping[str, Any]
    qwen_path: str


class RemoteSensingVisionAdapter:
    """Model-swappable contract implemented by concrete RS vision adapters."""

    def validate_inputs(self, request: EOInput) -> None:
        raise NotImplementedError

    def prepare_inputs(self, request: EOInput) -> PreparedEOInput:
        raise NotImplementedError

    def get_capabilities(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def get_provenance(self, request: EOInput, prepared: np.ndarray | None = None) -> Mapping[str, Any]:
        raise NotImplementedError

    def estimate_resources(self, **kwargs: Any) -> Mapping[str, Any]:
        raise NotImplementedError


class Qwen25VLRGBAdapter(RemoteSensingVisionAdapter):
    """A model-swappable RGB boundary; it intentionally does not load Qwen."""

    model_id = MODEL_ID
    adapter_id = ADAPTER_ID

    def get_input_contracts(self) -> dict[str, InputContract]:
        return {
            "OPTICAL_RGB": InputContract("OPTICAL_RGB", "SUPPORTED", (None, None, 3),
                                         {"channels": ("R", "G", "B"), "lossless_to_qwen": False}, "DIRECT_QWEN_INPUT"),
            "S2_MULTISPECTRAL": InputContract("S2_MULTISPECTRAL", "ADAPTER_REQUIRED", (None, 12, 120, 120),
                                               {"bands": S2_BANDS, "approaches": ("selected_band_pseudo_rgb", "multi_channel_projection", "grouped_bands", "learned_projector", "native_encoder")}, "PROJECTOR_REQUIRED"),
            "S1_SAR": InputContract("S1_SAR", "ADAPTER_REQUIRED", (None, 2, 120, 120),
                                     {"bands": S1_BANDS, "native_sar": False, "approaches": ("two_channel_projection", "pseudo_rgb", "dedicated_encoder")}, "PROJECTOR_REQUIRED"),
            "OPTICAL_SAR": InputContract("OPTICAL_SAR", "ADAPTER_REQUIRED", None,
                                          {"fusion": ("shared_visual_tokens", "cross_modal_fusion", "multi_image_packing", "dedicated_encoders"), "implemented_fusion": False}, "PROJECTOR_REQUIRED"),
            "TEMPORAL_PAIR": InputContract("TEMPORAL_PAIR", "NOT_IMPLEMENTED", None,
                                            {"fields": ("T1", "T2", "timestamps", "spatial_correspondence", "metadata"), "learned_temporal_reasoning": False}, "PROJECTOR_REQUIRED"),
            "CROMA_TOKENS": InputContract("CROMA_TOKENS", "ADAPTER_REQUIRED", (None, 225, 768),
                                           {"direct_qwen_input": False}, "PROJECTOR_REQUIRED"),
            "CROMA_GAP": InputContract("CROMA_GAP", "ADAPTER_REQUIRED", (None, 768),
                                        {"direct_qwen_input": False}, "PROJECTOR_REQUIRED"),
            "PHYSICAL_62D": InputContract("PHYSICAL_62D", "UNSUPPORTED", (None, 62),
                                           {"role": "scientific physical representation"}, "UNSUPPORTED"),
            "HYBRID_830D": InputContract("HYBRID_830D", "UNSUPPORTED", (None, 830),
                                          {"role": "SCIENTIFIC_PREDICTOR_ONLY"}, "SCIENTIFIC_PREDICTOR_ONLY"),
        }

    def get_grounding_contract(self) -> dict[str, Any]:
        return {"status": "NOT_VERIFIED", "outputs": ("point", "bbox", "polygon", "mask", "region_reference"),
                "source_geometry": "preserved_only", "learned_grounding": False, "phase2f_mapping": "FAIL_CLOSED"}

    def load_model(self, snapshot: str | Path, *, dtype: str = "float16", device: str = "cuda") -> dict[str, Any]:
        """Load Qwen lazily; imports are kept out of ordinary unit-test paths."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device != "cuda" or dtype not in {"float16", "bfloat16"}:
            raise ValueError("Phase 3D.3 loader supports only CUDA float16/bfloat16")
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            str(snapshot), local_files_only=True, torch_dtype=torch_dtype,
            device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
        )
        return {"model": model, "processor": processor, "load_seconds": time.perf_counter() - started,
                "dtype": dtype, "device": device, "quantization": "none",
                "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3"}

    def prepare_s2(self, s2: np.ndarray, *, band_order: tuple[str, ...] = S2_BANDS,
                   strategy: str = "true_color", source_image_id: str | None = None,
                   source_reference: str | None = None, nodata_value: float | None = None) -> PreparedEOInput:
        """Deterministically adapt one canonical S2 sample for Qwen's RGB processor."""
        self.validate_s2(s2, band_order=band_order, nodata_value=nodata_value)
        cube = np.asarray(s2, dtype=np.float32)
        if cube.ndim == 4:
            cube = cube[0]
        if nodata_value is not None:
            cube = np.where(cube == nodata_value, np.nan, cube)
        normalized = _robust_channel_scale(cube)
        indices = {band: index for index, band in enumerate(S2_BANDS)}
        if strategy == "true_color":
            views = (self._rgb_view(normalized, (indices["B04"], indices["B03"], indices["B02"])),)
            bands_used = ("B04", "B03", "B02")
        elif strategy == "false_color":
            views = (self._rgb_view(normalized, (indices["B08"], indices["B04"], indices["B03"])),)
            bands_used = ("B08", "B04", "B03")
        elif strategy == "band_group_views":
            groups = (("B01", "B02", "B03", "B04"), ("B05", "B06", "B07", "B08"),
                      ("B8A", "B09", "B11", "B12"))
            views = tuple(self._group_view(normalized, tuple(indices[band] for band in group)) for group in groups)
            bands_used = tuple(band for group in groups for band in group)
        else:
            raise ValueError(f"unknown S2 strategy: {strategy}")
        arrays = tuple(view.astype(np.uint8, copy=False) for view in views)
        model_input: np.ndarray | tuple[np.ndarray, ...] = arrays[0] if len(arrays) == 1 else arrays
        fingerprint = sha256(b"".join(view.tobytes() for view in arrays)).hexdigest()
        provenance = {
            "model_id": self.model_id, "checkpoint_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "adapter_id": "language_facing_s2_adapter_v1", "source_image_id": source_image_id,
            "source_s2_reference": source_reference, "canonical_band_order": list(S2_BANDS),
            "bands_used": list(bands_used), "strategy": strategy,
            "normalization": "existing per-channel mean +/- 2 std, clipped to [0,1]; nodata=0",
            "output_shapes": [list(view.shape) for view in arrays], "output_dtype": "uint8",
            "information_loss": True, "preprocessing_fingerprint": fingerprint,
            "scientific_representation_used": False,
        }
        return PreparedEOInput("S2_MULTISPECTRAL", model_input, "language_facing_s2_adapter_v1",
                               {"strategy": strategy, "bands_used": list(bands_used),
                                "normalization": provenance["normalization"], "information_loss": True}, provenance)

    @staticmethod
    def validate_s2(s2: np.ndarray, *, band_order: tuple[str, ...], nodata_value: float | None = None) -> None:
        if tuple(band_order) != S2_BANDS:
            raise ValueError("S2 band order must exactly match canonical B01..B12 order")
        if not isinstance(s2, np.ndarray) or s2.ndim not in {3, 4}:
            raise ValueError("S2 input must be [12,120,120] or [N,12,120,120]")
        if s2.ndim == 3 and s2.shape != (12, 120, 120):
            raise ValueError("S2 input must have shape [12,120,120]")
        if s2.ndim == 4 and (s2.shape[1:] != (12, 120, 120) or s2.shape[0] != 1):
            raise ValueError("S2 input must have shape [1,12,120,120] for one Qwen request")
        values = np.asarray(s2, dtype=np.float32)
        valid = values != nodata_value if nodata_value is not None else np.ones(values.shape, dtype=bool)
        if not np.isfinite(values[valid]).all():
            raise ValueError("S2 input contains non-finite values outside declared nodata")

    @staticmethod
    def _rgb_view(cube: np.ndarray, channels: tuple[int, int, int]) -> np.ndarray:
        return np.clip(np.moveaxis(cube[list(channels)], 0, -1) * 255.0, 0, 255)

    @staticmethod
    def _group_view(cube: np.ndarray, channels: tuple[int, int, int, int]) -> np.ndarray:
        fourth = (cube[channels[2]] + cube[channels[3]]) / 2.0
        return np.clip(np.stack((cube[channels[0]], cube[channels[1]], fourth), axis=-1) * 255.0, 0, 255)

    def generate(self, runtime: Mapping[str, Any], prepared: PreparedEOInput, prompt: str,
                 *, max_new_tokens: int = 64) -> dict[str, Any]:
        from PIL import Image
        import torch

        arrays = prepared.model_input if isinstance(prepared.model_input, tuple) else (prepared.model_input,)
        images = [Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB") for value in arrays]
        processor = runtime["processor"]
        model = runtime["model"]
        messages = [{"role": "user", "content": ([{"type": "image", "image": image} for image in images] + [{"type": "text", "text": prompt}])}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {"text": output, "generation_seconds": time.perf_counter() - started,
                "generated_tokens": int(generated.shape[1] - inputs["input_ids"].shape[1]),
                "provenance": {**prepared.provenance, "dtype": runtime["dtype"],
                               "device": runtime["device"], "quantization": runtime["quantization"],
                               "checkpoint_revision": runtime["model_revision"]}}

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "rgb": "PASS",
            "multispectral_12_band": "NOT SUPPORTED",
            "sar_vv_vh": "NOT SUPPORTED",
            "optical_sar_joint": "NOT SUPPORTED",
            "temporal_pair": "NOT VERIFIED",
            "croma_gap": "NOT SUPPORTED",
            "croma_tokens": "NOT SUPPORTED",
            "physical_62d": "NOT SUPPORTED",
            "hybrid_830d": "NOT SUPPORTED",
            "grounding": "NOT VERIFIED",
            "input_contracts": {key: value.status for key, value in self.get_input_contracts().items()},
        }

    def validate_inputs(self, request: EOInput) -> None:
        if request.modality == "OPTICAL_RGB":
            self._require_image(request.optical, 3, "optical RGB")
            return
        if request.modality in {"OPTICAL", "OPTICAL_SAR", "SAR", "TEMPORAL"}:
            raise ValueError(f"{request.modality}: unsupported by RGB-only smoke adapter")
        raise ValueError(f"unknown modality: {request.modality}")

    def prepare_inputs(self, request: EOInput) -> PreparedEOInput:
        self.validate_inputs(request)
        assert request.optical is not None
        # Preserve an explicit adapter boundary. A future processor may replace
        # this conversion with the model's documented processor.
        image = np.asarray(request.optical, dtype=np.uint8).copy()
        return PreparedEOInput(
            modality=request.modality,
            model_input=image,
            adapter_id=self.adapter_id,
            preprocessing={"operation": "identity_uint8_rgb", "native_model_processor": "not_loaded"},
            provenance=self.get_provenance(request, image),
        )

    def get_provenance(self, request: EOInput, prepared: np.ndarray | None = None) -> dict[str, Any]:
        value = prepared if prepared is not None else request.optical
        digest = sha256(np.asarray(value).tobytes()).hexdigest() if value is not None else None
        return {
            "model_id": self.model_id,
            "adapter_id": self.adapter_id,
            "modality": request.modality,
            "input_sha256": digest,
            "scientific_representation_used": False,
            "temporal_metadata": dict(request.metadata or {}),
        }

    def estimate_resources(self, *, dtype: str = "float16", quantization: str = "none",
                           image_count: int = 1, image_pixels: int = 120 * 120,
                           max_new_tokens: int = 64) -> dict[str, Any]:
        measured_rgb_vram = 7520955904
        return {
            "status": "ESTIMATE_ONLY",
            "device": "unknown_until_runtime",
            "dtype": dtype,
            "quantization": quantization,
            "estimated_vram_bytes": measured_rgb_vram,
            "estimated_context_cost": {"image_count": image_count, "image_pixels": image_pixels},
            "estimated_image_count": image_count,
            "estimated_generation_cost": {"max_new_tokens": max_new_tokens},
            "peak_vram_bytes": None,
            "peak_ram_bytes": None,
            "basis": "measured FP16 post-load allocation for this checkpoint; generation/context overhead not modeled",
        }

    @staticmethod
    def _require_image(value: np.ndarray | None, channels: int, label: str) -> None:
        if not isinstance(value, np.ndarray) or value.ndim != 3 or value.shape[2] != channels:
            raise ValueError(f"{label} must be HxWx{channels} uint8-compatible array")
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"{label} contains no finite pixels")
