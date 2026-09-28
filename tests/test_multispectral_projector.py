import json

import pytest
import torch
from torch import nn

from src.eo_vlm_adapter import S2_BANDS
from src.eo_vlm.multispectral_projector import (
    QWEN_HIDDEN_SIZE,
    QWEN_IMAGE_TOKEN_COUNT,
    S2MultispectralProjector,
)
from src.eo_vlm.training import (
    PilotConfig,
    build_qwen_token_batch,
    freeze_qwen,
    save_projector_checkpoint,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True, max_length=None, truncation=False):
        values = [10 + (ord(char) % 31) for char in text]
        if add_special_tokens:
            values = [1] + values
        if max_length is not None and truncation:
            values = values[:max_length]
        return {"input_ids": values}


class TinyConfig:
    vision_start_token_id = 151652
    image_token_id = 151655
    vision_end_token_id = 151653


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TinyConfig()
        self.embedding = nn.Embedding(151700, QWEN_HIDDEN_SIZE)
        self.extra = nn.Linear(4, 4)

    def get_input_embeddings(self):
        return self.embedding


def test_projector_preserves_canonical_band_order_and_output_shape():
    assert S2_BANDS == ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
    projector = S2MultispectralProjector()
    output = projector(torch.ones(2, 12, 120, 120))
    assert output.shape == (2, QWEN_IMAGE_TOKEN_COUNT, QWEN_HIDDEN_SIZE)
    assert torch.isfinite(output).all()


def test_projector_rejects_invalid_inputs():
    projector = S2MultispectralProjector()
    with pytest.raises(ValueError, match="shape"):
        projector(torch.ones(1, 3, 120, 120))
    with pytest.raises(ValueError, match="non-finite"):
        value = torch.ones(1, 12, 120, 120)
        value[0, 0, 0, 0] = float("nan")
        projector(value)


def test_qwen_is_frozen_and_projector_is_the_only_trainable_component():
    model = TinyQwen()
    projector = S2MultispectralProjector()
    summary = freeze_qwen(model)
    assert summary["qwen_trainable_parameter_count"] == 0
    assert projector.trainable_parameter_count > 0
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in projector.parameters())


def test_qwen_token_batch_injects_learned_tokens_and_masks_prompt():
    model = TinyQwen()
    projector = S2MultispectralProjector()
    batch = build_qwen_token_batch(
        tokenizer=TinyTokenizer(), model=model, projector=projector,
        s2=torch.ones(1, 12, 120, 120), questions=["What is visible?"], answers=["vegetation"],
    )
    assert batch["inputs_embeds"].shape[-1] == QWEN_HIDDEN_SIZE
    assert int(batch["input_ids"].eq(model.config.image_token_id).sum()) == QWEN_IMAGE_TOKEN_COUNT
    assert (batch["labels"] == -100).any()
    batch["inputs_embeds"].sum().backward()
    assert any(parameter.grad is not None for parameter in projector.parameters())


def test_projector_provenance_and_checkpoint_round_trip(tmp_path):
    projector = S2MultispectralProjector()
    provenance = projector.provenance(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        checkpoint_revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
        input_shape=(1, 12, 120, 120),
    )
    assert provenance["canonical_band_order"] == list(S2_BANDS)
    assert provenance["output_shape"] == [1, QWEN_IMAGE_TOKEN_COUNT, QWEN_HIDDEN_SIZE]
    path = tmp_path / "projector.pt"
    digest = save_projector_checkpoint(
        path, projector, provenance=provenance, config=PilotConfig(), training_summary={"status": "structural_test"}
    )
    assert path.is_file()
    receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert receipt["checkpoint_sha256"] == digest
    restored = S2MultispectralProjector()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    assert restored.fingerprint() == projector.fingerprint()
