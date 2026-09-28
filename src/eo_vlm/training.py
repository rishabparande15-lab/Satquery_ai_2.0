"""Small, split-safe training utilities for the Phase 3G projector pilot."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .multispectral_projector import (
    QWEN_HIDDEN_SIZE,
    QWEN_IMAGE_GRID_THW,
    QWEN_IMAGE_TOKEN_COUNT,
    S2MultispectralProjector,
)


@dataclass(frozen=True)
class PilotConfig:
    seed: int = 17
    train_limit: int = 100
    validation_limit: int = 50
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_answer_tokens: int = 32
    precision: str = "float16"
    quantization: str = "none"


def freeze_qwen(model: torch.nn.Module) -> dict[str, int]:
    """Freeze every Qwen parameter and return the separation counts."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"total_parameter_count": total, "qwen_trainable_parameter_count": 0}


def optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    """Return optimizer membership by object identity, never tensor equality."""
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def trainable_parameter_summary(projector: S2MultispectralProjector, model: torch.nn.Module) -> dict[str, int]:
    qwen = freeze_qwen(model)
    projector_count = projector.trainable_parameter_count
    return {
        **qwen,
        "projector_trainable_parameter_count": projector_count,
        "total_trainable_parameter_count": projector_count,
    }


def build_qwen_token_batch(
    *,
    tokenizer: Any,
    model: torch.nn.Module,
    projector: S2MultispectralProjector,
    s2: torch.Tensor,
    questions: list[str],
    answers: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Build a differentiable Qwen input with learned visual-token slots.

    The placeholder IDs and grid match Qwen's own image-token interface.  The
    projector embeddings replace those slots through `inputs_embeds`; labels
    mask the visual prefix and question, leaving only real annotation answer
    tokens supervised.
    """
    if len(questions) != s2.shape[0] or (answers is not None and len(answers) != len(questions)):
        raise ValueError("batch fields must have equal length")
    if not hasattr(model, "config"):
        raise ValueError("model config is required")
    device = next(model.parameters()).device
    visual = projector(s2.to(device))
    tokenizer_inputs: list[list[int]] = []
    label_inputs: list[list[int]] = []
    for index, question in enumerate(questions):
        if not isinstance(question, str) or not question:
            raise ValueError("question text is required")
        prompt = f"Question: {question}\nAnswer:"
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        answer_ids: list[int] = []
        if answers is not None:
            answer_ids = tokenizer(
                " " + answers[index], add_special_tokens=False,
                max_length=32, truncation=True,
            )["input_ids"]
            if tokenizer.eos_token_id is not None:
                answer_ids.append(tokenizer.eos_token_id)
        prefix = [model.config.vision_start_token_id]
        prefix += [model.config.image_token_id] * QWEN_IMAGE_TOKEN_COUNT
        prefix += [model.config.vision_end_token_id]
        tokenizer_inputs.append(prefix + prompt_ids + answer_ids)
        label_inputs.append([-100] * (len(prefix) + len(prompt_ids)) + answer_ids)
    max_length = max(len(value) for value in tokenizer_inputs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(tokenizer_inputs), max_length), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    attention = torch.zeros_like(input_ids)
    for index, (ids, target) in enumerate(zip(tokenizer_inputs, label_inputs)):
        input_ids[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        labels[index, :len(target)] = torch.tensor(target, dtype=torch.long, device=device)
        attention[index, :len(ids)] = 1
    embeddings = model.get_input_embeddings()(input_ids)
    image_mask = input_ids.eq(model.config.image_token_id).unsqueeze(-1).expand_as(embeddings)
    visual_flat = visual.to(dtype=embeddings.dtype).reshape(-1)
    if int(image_mask.sum()) != visual.numel():
        raise RuntimeError("visual token count does not match Qwen image-token slots")
    embeddings = embeddings.masked_scatter(image_mask, visual_flat)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "inputs_embeds": embeddings,
        "labels": labels,
        "image_grid_thw": torch.tensor([QWEN_IMAGE_GRID_THW] * len(questions), dtype=torch.long, device=device),
    }


def save_projector_checkpoint(
    path: Path,
    projector: S2MultispectralProjector,
    *,
    provenance: Mapping[str, Any],
    config: PilotConfig,
    training_summary: Mapping[str, Any],
) -> str:
    """Save only adapter weights plus a JSON provenance sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": projector.state_dict(), "adapter_id": provenance["adapter_id"]}, path)
    checkpoint_sha256 = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    receipt = {
        "adapter_id": provenance["adapter_id"],
        "model_id": provenance["model_id"],
        "checkpoint_revision": provenance["checkpoint_revision"],
        "adapter_checkpoint": path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "config": config.__dict__,
        "training_summary": dict(training_summary),
        "provenance": dict(provenance),
        "scientific_representation_used": False,
    }
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checkpoint_sha256
