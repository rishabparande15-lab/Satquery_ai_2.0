"""Single-image multispectral VQA using the frozen Phase 3O S2 baseline."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .eo_vlm.generation_interface import generate_with_multimodal_prefix
from .eo_vlm.multispectral_projector import S2MultispectralProjector
from .eo_vlm.training import build_qwen_visual_token_batch, freeze_qwen
from .eo_vlm_adapter import MODEL_ID, Qwen25VLRGBAdapter, S2_BANDS
from .preprocessing import _robust_channel_scale


ROOT = Path(__file__).resolve().parents[1]
PROJECTOR_CHECKPOINT = ROOT / "artifacts" / "training" / "phase3o" / "phase3o5_run" / "s2_multispectral_projector.pt"
EXPECTED_PROJECTOR_SHA256 = "e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db"
EXPECTED_PROJECTOR_FINGERPRINT = "db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46"
QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
QWEN_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen2.5-VL-3B-Instruct"
    / "snapshots"
    / QWEN_REVISION
)
VQA_TASKS = frozenset({"binary_qa", "multiple_choice_qa"})
GROUNDING_TERMS = ("highlight", "locate", "where", "outline", "bounding box", "bbox", "mask", "polygon", "point")
TEMPORAL_TERMS = ("before", "after", "temporal", "change", "compare")


def classify_single_image_route(query: str, *, image_count: int = 1,
                                modalities: Sequence[str] = ("s2",), task: str | None = None) -> str:
    text = query.lower()
    modality_set = {str(value).lower() for value in modalities}
    if any(term in text for term in TEMPORAL_TERMS):
        return "TEMPORAL_ROUTE_NOT_IMPLEMENTED"
    if modality_set in ({"s1", "s2"}, {"sar", "optical"}) or ("sar" in text and "optical" in text):
        return "OPTICAL_SAR_ANALYSIS"
    if image_count != 1:
        return "UNSUPPORTED_INPUT_CONFIGURATION"
    if not modality_set or not modality_set <= {"s2", "optical", "multispectral"}:
        return "UNSUPPORTED_INPUT_CONFIGURATION"
    if task in {"SINGLE_IMAGE_GROUNDING", "grounding", "TEXT_GUIDED_GROUNDING"} or any(term in text for term in GROUNDING_TERMS):
        return "SINGLE_IMAGE_GROUNDING"
    return "SINGLE_IMAGE_VQA"


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_single_image_input(
    *,
    image_id: str,
    split: str,
    optical: np.ndarray,
    metadata: Mapping[str, Any] | None,
    question: str,
    task_type: str,
) -> None:
    if not image_id or not isinstance(image_id, str):
        raise ValueError("A stable single-image identity is required")
    if split not in {"train", "validation", "external_inference"}:
        raise ValueError("Single-image VQA accepts train, validation, or external inference inputs only")
    if task_type not in VQA_TASKS:
        raise ValueError("Single-image VQA supports binary_qa and multiple_choice_qa only")
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("A non-empty VQA question of at most 2000 characters is required")
    if not isinstance(optical, np.ndarray) or optical.shape != (12, 120, 120):
        raise ValueError("S2 input must have canonical shape [12,120,120]")
    if not np.isfinite(optical).all():
        raise ValueError("S2 input contains non-finite values")
    if not isinstance(metadata, Mapping) or metadata.get("shape") != [120, 120]:
        raise ValueError("S2 image metadata must declare the canonical 120x120 grid")
    if not metadata.get("crs"):
        raise ValueError("S2 image metadata requires a CRS")
    resolution = metadata.get("resolution")
    bounds = metadata.get("bounds")
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError("S2 image metadata requires two resolution values")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise ValueError("S2 image metadata requires four bounds")
    values = np.asarray([*resolution, *bounds], dtype=np.float64)
    if not np.isfinite(values).all() or any(value <= 0 for value in resolution):
        raise ValueError("S2 resolution and bounds must be finite with positive resolution")
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("S2 image bounds must have positive extent")


def parse_vqa_response(task_type: str, text: str, choices: Sequence[Mapping[str, Any]] | None = None) -> str | None:
    if task_type == "binary_qa":
        match = re.match(r"^\s*(yes|no|true|false)(?=$|[\s.,:;!?])", text, flags=re.IGNORECASE)
        if not match:
            return None
        answer = match.group(1).lower()
        return {"true": "yes", "false": "no"}.get(answer, answer)
    if task_type == "multiple_choice_qa":
        match = re.match(r"^\s*(?:option\s+)?\(?([a-z])\)?(?=$|[\s.,:;!?])", text, flags=re.IGNORECASE)
        if not match:
            return None
        key = match.group(1).lower()
        allowed = {str(choice.get("key", "")).lower() for choice in choices or ()}
        if not allowed:
            allowed = {"a", "b", "c", "d"}
        return key if key in allowed else None
    return None


class SingleImageVQAController:
    """Frozen S2 projector + pinned Qwen path; no SAR input or grounding output."""

    route = "SINGLE_IMAGE_VQA"
    specialist = "phase3o5_s2_projector_qwen25vl"

    def __init__(
        self,
        *,
        projector_checkpoint: Path = PROJECTOR_CHECKPOINT,
        qwen_snapshot: Path = QWEN_SNAPSHOT,
        device: str = "cuda",
    ) -> None:
        self.projector_checkpoint = Path(projector_checkpoint)
        self.qwen_snapshot = Path(qwen_snapshot)
        self.device = device
        self.projector = self.model = self.tokenizer = None
        self.adapter = Qwen25VLRGBAdapter()
        self.provenance: dict[str, Any] = {}

    def load(self) -> None:
        if self.model is not None:
            return
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SINGLE_IMAGE_VQA_REQUIRES_VALIDATED_CUDA_RUNTIME")
        if not self.projector_checkpoint.is_file():
            raise FileNotFoundError("S2 VQA projector checkpoint is unavailable")
        checkpoint_sha = hashlib.sha256(self.projector_checkpoint.read_bytes()).hexdigest()
        if checkpoint_sha != EXPECTED_PROJECTOR_SHA256:
            raise RuntimeError("UNAPPROVED_S2_VQA_PROJECTOR_CHECKPOINT")
        if not self.qwen_snapshot.is_dir() or self.qwen_snapshot.name != QWEN_REVISION:
            raise FileNotFoundError("Pinned Qwen2.5-VL snapshot is unavailable")
        payload = torch.load(self.projector_checkpoint, map_location="cpu", weights_only=True)
        projector = S2MultispectralProjector().eval()
        projector.load_state_dict(payload["state_dict"])
        fingerprint = state_fingerprint(projector.state_dict())
        if fingerprint != EXPECTED_PROJECTOR_FINGERPRINT:
            raise RuntimeError("UNAPPROVED_S2_VQA_PROJECTOR_FINGERPRINT")
        runtime = self.adapter.load_model(self.qwen_snapshot, dtype="float16", device=self.device)
        model = runtime["model"].eval()
        freeze = freeze_qwen(model)
        if freeze["qwen_trainable_parameter_count"] != 0:
            raise RuntimeError("SINGLE_IMAGE_VQA_QWEN_MUST_REMAIN_FROZEN")
        self.projector = projector.to(self.device).eval()
        self.model = model
        self.tokenizer = runtime["processor"].tokenizer
        self.provenance = {
            "model_id": MODEL_ID,
            "qwen_revision": QWEN_REVISION,
            "projector_checkpoint_sha256": checkpoint_sha,
            "projector_state_fingerprint": fingerprint,
            "projector_adapter_id": "phase3g_s2_qwen_token_projector_v1",
            "canonical_band_order": list(S2_BANDS),
            "normalization": "robust_channel_scale_v1",
            "generation_interface": "phase3o10_multimodal_prefix_v1",
            "scientific_representation_used": False,
            "qwen_trainable_parameter_count": 0,
        }

    @torch.inference_mode()
    def run(
        self,
        *,
        image_id: str,
        split: str,
        optical: np.ndarray,
        metadata: Mapping[str, Any],
        question: str,
        task_type: str,
        choices: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        validate_single_image_input(
            image_id=image_id, split=split, optical=optical, metadata=metadata,
            question=question, task_type=task_type,
        )
        self.load()
        scaled = _robust_channel_scale(optical).astype(np.float32, copy=False)
        tensor = torch.from_numpy(scaled).unsqueeze(0).to(self.device)
        visual_tokens = self.projector(tensor)
        batch = build_qwen_visual_token_batch(
            tokenizer=self.tokenizer,
            model=self.model,
            visual_tokens=visual_tokens,
            questions=[question],
            answers=None,
        )
        generated = generate_with_multimodal_prefix(
            self.model,
            input_ids=batch["input_ids"],
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            image_grid_thw=batch["image_grid_thw"],
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            use_cache=True,
        )
        new_tokens = generated[0, batch["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {
            "status": "OK",
            "task": self.route,
            "task_type": task_type,
            "selected_specialist": self.specialist,
            "image_id": image_id,
            "split": split,
            "route": self.route,
            "question": question,
            "answer": text.strip(),
            "generated_response": text,
            "parsed_answer": parse_vqa_response(task_type, text, choices),
            "visual_evidence": {
                "type": "S2_MULTISPECTRAL_PROJECTED_TOKENS",
                "input_shape": [12, 120, 120],
                "projected_token_shape": list(visual_tokens.shape),
                "crs": metadata["crs"],
                "resolution": list(metadata["resolution"]),
                "bounds": list(metadata["bounds"]),
                "bounding_boxes": [],
                "masks": [],
            },
            "execution_trace": [
                {"step": "route_selection", "status": "EXECUTED", "route": self.route},
                {"step": "single_s2_validation", "status": "EXECUTED", "shape": [12, 120, 120]},
                {"step": "s2_projector", "status": "EXECUTED", "token_shape": list(visual_tokens.shape)},
                {"step": "frozen_qwen_generation", "status": "EXECUTED", "qwen_revision": QWEN_REVISION},
            ],
            "provenance": {**self.provenance, "image_id": image_id, "split": split},
            "warnings": [
                "S2 language capability is an internal projector proof of concept, not an RSVQA benchmark result.",
                "This route uses S2 only; no S1 input or optical-SAR fusion is used.",
            ],
        }

    def close(self) -> None:
        """Release resident model references at a specialist-family boundary."""
        self.projector = self.model = self.tokenizer = None
