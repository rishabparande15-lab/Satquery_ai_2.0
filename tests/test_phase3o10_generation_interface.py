import torch

from src.eo_vlm.generation_interface import (
    GENERATION_INTERFACE_VERSION,
    generate_with_multimodal_prefix,
)


class _GenerationFixture:
    """Small deterministic stand-in for the HF prefill/drop-ID behavior."""
    def __init__(self):
        self.calls = []

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        prepared = dict(kwargs)
        # This is the observed pre-fix HF contract for inputs_embeds prefill.
        prepared["input_ids"] = None if kwargs.get("inputs_embeds") is not None else input_ids
        prepared["cache_position"] = torch.arange(input_ids.shape[1])
        return prepared

    def generate(self, *, input_ids, **kwargs):
        prepared = self.prepare_inputs_for_generation(input_ids, **kwargs)
        self.calls.append(prepared)
        return prepared


def test_phase3o10_restores_placeholder_ids_only_at_generation_boundary():
    fixture = _GenerationFixture()
    ids = torch.tensor([[11, 99, 99, 12]], dtype=torch.long)
    embeds = torch.randn(1, 4, 8)
    mask = torch.ones_like(ids)

    before = fixture.generate(input_ids=ids, inputs_embeds=embeds, attention_mask=mask)
    assert before["input_ids"] is None  # pre-fix regression condition

    after = generate_with_multimodal_prefix(
        fixture, input_ids=ids, inputs_embeds=embeds, attention_mask=mask,
    )
    assert GENERATION_INTERFACE_VERSION == "phase3o10_multimodal_prefix_v1"
    assert torch.equal(after["input_ids"], ids)
    assert torch.equal(after["inputs_embeds"], embeds)
    assert torch.equal(after["attention_mask"], mask)
    # The patch is scoped to this call; historical/raw generation remains reproducible.
    assert fixture.prepare_inputs_for_generation(ids, inputs_embeds=embeds)["input_ids"] is None
