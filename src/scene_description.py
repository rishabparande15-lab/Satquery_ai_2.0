"""Verified remote-sensing RGB scene-description specialist for Phase 3X.

The specialist is intentionally separate from the Sentinel-2 VQA projector.
It accepts an explicit RGB rendition only; it never infers a satellite sensor
from pixels and never returns calibrated confidence or spatial geometry.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor


MODEL_ID = "AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct"
MODEL_ENV = "SATQUERY_SCENE_DESCRIPTION_MODEL"
DEFAULT_PROMPT = "Describe this remote-sensing image in one concise sentence."


class SceneDescriptionInputError(ValueError):
    """A fail-closed scene-description input rejection."""


def render_s2_rgb(optical: np.ndarray) -> Image.Image:
    """Render canonical S2 B04/B03/B02 as RGB for this RGB-only specialist."""
    if not isinstance(optical, np.ndarray) or optical.shape != (12, 120, 120):
        raise SceneDescriptionInputError("Scene description requires canonical [12,120,120] S2 data for a local S2 input.")
    if not np.isfinite(optical).all():
        raise SceneDescriptionInputError("Scene-description S2 input contains non-finite pixels.")
    rgb = np.stack((optical[3], optical[2], optical[1]), axis=-1)
    lo, hi = np.percentile(rgb, (2, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise SceneDescriptionInputError("Scene-description RGB rendering has invalid contrast limits.")
    return Image.fromarray((np.clip((rgb - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8), mode="RGB")


def load_rgb_image(path: str | Path) -> Image.Image:
    """Load a true RGB upload; grayscale and multispectral files fail closed."""
    candidate = Path(path)
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise SceneDescriptionInputError("Scene description accepts PNG, JPEG, or RGB TIFF inputs only.")
    try:
        with Image.open(candidate) as source:
            if source.width < 1 or source.height < 1:
                raise SceneDescriptionInputError("Scene-description image dimensions are invalid.")
            if source.mode not in {"RGB", "RGBA"}:
                raise SceneDescriptionInputError("Scene-description uploads must be explicitly RGB or RGBA; multispectral sensor identity is not inferred.")
            return source.convert("RGB").copy()
    except SceneDescriptionInputError:
        raise
    except Exception as exc:
        raise SceneDescriptionInputError("Scene-description image is unreadable or corrupt.") from exc


class SceneDescriptionController:
    """Lazy local controller for the official Apache-2.0 scene-description model."""

    def __init__(self, *, model_root: str | Path | None = None) -> None:
        configured = model_root or os.environ.get(MODEL_ENV)
        if not configured:
            raise RuntimeError(f"{MODEL_ENV} must point to the approved {MODEL_ID} checkpoint.")
        self.model_root = Path(configured)
        if not (self.model_root / "model.safetensors").is_file():
            raise FileNotFoundError(f"Scene-description checkpoint is unavailable: {self.model_root}")
        self._model = None
        self._processor = None
        # The approved checkpoint is immutable for one controller lifetime.
        # Hash it once after the model is admitted, then reuse that provenance
        # value for warm requests instead of rereading multi-gigabyte weights.
        self._model_sha256: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Scene-description specialist requires the validated CUDA runtime.")
        tokenizer = AutoTokenizer.from_pretrained(self.model_root, local_files_only=True)
        processor = Qwen2VLProcessor(
            image_processor=Qwen2VLImageProcessor(min_pixels=3136, max_pixels=12845056),
            tokenizer=tokenizer,
        )
        processor.chat_template = json.loads((self.model_root / "chat_template.json").read_text(encoding="utf-8"))["chat_template"]
        self._processor = processor
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_root, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True,
        ).eval()
        self._model_sha256 = hashlib.sha256((self.model_root / "model.safetensors").read_bytes()).hexdigest()

    def run(self, *, image: Image.Image, image_identity: str, source_kind: str,
            source_metadata: Mapping[str, Any] | None = None, query: str | None = None) -> dict[str, Any]:
        if not isinstance(image_identity, str) or not image_identity.strip():
            raise SceneDescriptionInputError("A stable scene-description image identity is required.")
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise SceneDescriptionInputError("Scene-description input must be an RGB image.")
        started = perf_counter(); self._load()
        prompt_text = str(query or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt_text},
        ]}]
        prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=96, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        description = self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        if not description:
            raise RuntimeError("Scene-description specialist returned no non-empty language output.")
        assert self._model_sha256 is not None
        return {
            "status": "OK", "task": "SINGLE_IMAGE_SCENE_DESCRIPTION", "selected_specialist": MODEL_ID,
            "model_tool": MODEL_ID, "answer": description, "generated_response": description,
            "image_identity": image_identity, "source_kind": source_kind,
            "image_size": [image.width, image.height], "source_metadata": dict(source_metadata or {}),
            "visual_evidence": {"type": "INPUT_IMAGE_ONLY", "masks": [], "boxes": []},
            "provenance": {"model_id": MODEL_ID, "model_sha256": self._model_sha256, "model_license": "Apache-2.0", "device": "cuda:0", "source_kind": source_kind},
            "warnings": ["Scene description is generated from an explicit RGB rendering or RGB upload; sensor identity is not inferred from pixels.", "Answer confidence is not calibrated."],
            "runtime_seconds": round(perf_counter() - started, 6),
        }

    def close(self) -> None:
        """Release resident model references at a specialist-family boundary."""
        self._model = None
        self._processor = None
