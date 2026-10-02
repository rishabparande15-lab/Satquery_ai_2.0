"""Fail-closed Sentinel-1 SAR-only VQA using the audited Phase 3P asset."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import get_settings
from .croma_adapter import CROMAAdapter
from .eo_vlm.generation_interface import generate_with_multimodal_prefix
from .eo_vlm.image_conditioned_objective import target_sequence_log_probability
from .eo_vlm.sar_projector import S1SARProjector
from .eo_vlm.training import build_qwen_visual_token_batch, freeze_qwen
from .eo_vlm_adapter import MODEL_ID, Qwen25VLRGBAdapter, S1_BANDS
from .single_image_vqa import QWEN_REVISION, QWEN_SNAPSHOT, VQA_TASKS, parse_vqa_response, state_fingerprint


ROOT = Path(__file__).resolve().parents[1]
PROJECTOR_CHECKPOINT = ROOT / "artifacts" / "training" / "phase3p" / "phase3p1_sar_adaptation" / "phase3p1_sar_projector_final.pt"
EXPECTED_PROJECTOR_SHA256 = "a7e57bff0cba141db5c34f549d0bc3f25638960fe39ab25246bbab49e979accd"


def validate_single_sar_input(*, image_id: str, split: str, sar: np.ndarray, metadata: Mapping[str, Any], question: str, task_type: str) -> None:
    if not image_id or split not in {"train", "validation", "external_inference"}:
        raise ValueError("A stable non-test Sentinel-1 image identity and permitted split are required")
    if task_type not in VQA_TASKS or not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("SAR VQA supports a non-empty binary_qa or multiple_choice_qa question only")
    if not isinstance(sar, np.ndarray) or sar.shape != (2, 120, 120):
        raise ValueError("S1 SAR input must have canonical shape [2,120,120]")
    if not np.isfinite(sar).all():
        raise ValueError("S1 SAR input contains non-finite values")
    if metadata.get("sar_band_order") != list(S1_BANDS) or not metadata.get("crs"):
        raise ValueError("S1 SAR metadata must declare CRS and canonical VV/VH band order")


class SingleImageSARVQAController:
    """Official CROMA S1 encoder -> audited S1 projector -> frozen pinned Qwen."""
    route = "SINGLE_IMAGE_SAR_VQA"
    specialist = "official_croma_s1_phase3p1_projector_qwen25vl"

    def __init__(self, *, projector_checkpoint: Path = PROJECTOR_CHECKPOINT, qwen_snapshot: Path = QWEN_SNAPSHOT, device: str = "cuda") -> None:
        self.projector_checkpoint, self.qwen_snapshot, self.device = Path(projector_checkpoint), Path(qwen_snapshot), device
        self.croma = self.projector = self.model = self.tokenizer = None
        self.adapter = Qwen25VLRGBAdapter(); self.provenance: dict[str, Any] = {}

    def load(self) -> None:
        if self.model is not None: return
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SINGLE_IMAGE_SAR_VQA_REQUIRES_VALIDATED_CUDA_RUNTIME")
        if not self.projector_checkpoint.is_file(): raise FileNotFoundError("S1 SAR VQA projector checkpoint is unavailable")
        digest = hashlib.sha256(self.projector_checkpoint.read_bytes()).hexdigest()
        if digest != EXPECTED_PROJECTOR_SHA256: raise RuntimeError("UNAPPROVED_S1_SAR_PROJECTOR_CHECKPOINT")
        if not self.qwen_snapshot.is_dir() or self.qwen_snapshot.name != QWEN_REVISION: raise FileNotFoundError("Pinned Qwen2.5-VL snapshot is unavailable")
        settings = get_settings()
        projector = S1SARProjector().eval()
        projector.load_state_dict(torch.load(self.projector_checkpoint, map_location="cpu", weights_only=True)["state_dict"])
        projector_fingerprint = state_fingerprint(projector.state_dict())
        runtime = self.adapter.load_model(self.qwen_snapshot, dtype="float16", device=self.device)
        model = runtime["model"].eval()
        if freeze_qwen(model)["qwen_trainable_parameter_count"] != 0: raise RuntimeError("SINGLE_IMAGE_SAR_VQA_QWEN_MUST_REMAIN_FROZEN")
        self.croma = CROMAAdapter(settings.croma_source, settings.croma_checkpoint, device=self.device)
        self.projector, self.model, self.tokenizer = projector.to(self.device).eval(), model, runtime["processor"].tokenizer
        self.provenance = {"model_id": MODEL_ID, "qwen_revision": QWEN_REVISION, "croma_model": "official CROMA Base S1 encoder", "croma_image_resolution": 120,
            "projector_checkpoint_sha256": digest, "projector_state_fingerprint": projector_fingerprint, "projector_adapter_id": "phase3p1_s1_sar_projector",
            "canonical_band_order": list(S1_BANDS), "normalization": "official CROMA per-channel mean +/- 2 std, clipped to [0,1]", "generation_interface": "phase3o10_multimodal_prefix_v1", "qwen_trainable_parameter_count": 0}

    @torch.inference_mode()
    def run(self, *, image_id: str, split: str, sar: np.ndarray, metadata: Mapping[str, Any], question: str, task_type: str, choices: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        validate_single_sar_input(image_id=image_id, split=split, sar=sar, metadata=metadata, question=question, task_type=task_type)
        self.load()
        assert self.croma is not None and self.projector is not None and self.model is not None and self.tokenizer is not None
        s1_tokens = self.croma.infer_modality(sar=sar.astype(np.float32, copy=False))["SAR_encodings"]
        visual_tokens = self.projector(s1_tokens)
        batch = build_qwen_visual_token_batch(tokenizer=self.tokenizer, model=self.model, visual_tokens=visual_tokens, questions=[question], answers=None)
        generated = generate_with_multimodal_prefix(self.model, input_ids=batch["input_ids"], inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"], image_grid_thw=batch["image_grid_thw"], max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0, use_cache=True)
        text = self.tokenizer.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        answer_source = "QWEN_GENERATION"
        candidate_scores: dict[str, float] | None = None
        # The audited Phase 3P path can legitimately emit EOS immediately for
        # binary prompts.  Do not invent a string: use the same frozen Qwen
        # teacher-forced candidate likelihood diagnostic that Phase 3P used.
        if not text and task_type == "binary_qa":
            candidate_scores = {}
            for candidate in ("yes", "no"):
                scored = build_qwen_visual_token_batch(tokenizer=self.tokenizer, model=self.model, visual_tokens=visual_tokens, questions=[question], answers=[candidate])
                candidate_scores[candidate] = float(target_sequence_log_probability(self.model(**scored).logits, scored["labels"]))
            text = max(candidate_scores, key=candidate_scores.get)
            answer_source = "QWEN_CANDIDATE_LIKELIHOOD"
        return {"status": "OK", "task": self.route, "task_type": task_type, "selected_specialist": self.specialist, "image_id": image_id, "split": split, "route": self.route, "question": question, "answer": text, "generated_response": text, "parsed_answer": parse_vqa_response(task_type, text, choices),
                "visual_evidence": {"type": "S1_SAR_CROMA_PROJECTED_TOKENS", "input_shape": [2, 120, 120], "croma_token_shape": list(s1_tokens.shape), "projected_token_shape": list(visual_tokens.shape), "crs": metadata["crs"], "resolution": list(metadata.get("resolution") or []), "bounds": list(metadata.get("bounds") or []), "bounding_boxes": [], "masks": []},
                "provenance": {**self.provenance, "image_id": image_id, "split": split, "answer_source": answer_source, "candidate_scores": candidate_scores}, "warnings": ["SAR_ONLY_SEMANTIC_VALIDATION_LIMITED", "Standalone SAR did not demonstrate clear semantic improvement; this route has no optical fallback.", *( ["Qwen generation emitted no text; binary answer is the frozen Qwen candidate-likelihood decision."] if answer_source != "QWEN_GENERATION" else [])],
                "execution_trace": [{"step": "s1_validation", "status": "EXECUTED", "shape": [2, 120, 120]}, {"step": "official_croma_s1_encoder", "status": "EXECUTED", "token_shape": list(s1_tokens.shape)}, {"step": "phase3p1_s1_projector", "status": "EXECUTED", "token_shape": list(visual_tokens.shape)}, {"step": "frozen_qwen_generation", "status": "EXECUTED", "qwen_revision": QWEN_REVISION}]}

    def close(self) -> None:
        self.croma = self.projector = self.model = self.tokenizer = None
