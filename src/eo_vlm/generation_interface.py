"""Versioned Phase 3O.10 generation-prefix compatibility boundary.

Qwen's stock generation preparation intentionally drops ``input_ids`` when
``inputs_embeds`` is supplied for prefill.  Phase 3O's projected-S2 contract
requires both: placeholder image IDs identify the visual slots while the
corresponding embedding rows hold the projector output.  This module restores
only those IDs for the first model invocation and delegates every later decode
step to the unmodified Transformers implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator


GENERATION_INTERFACE_VERSION = "phase3o10_multimodal_prefix_v1"


def _is_prefill(model_inputs: dict[str, Any]) -> bool:
    """Return whether a prepared generation call is the initial prefix call."""
    cache_position = model_inputs.get("cache_position")
    if cache_position is None:
        return model_inputs.get("past_key_values") is None
    try:
        return int(cache_position.reshape(-1)[0]) == 0
    except (AttributeError, IndexError, TypeError, ValueError):
        return model_inputs.get("past_key_values") is None


@contextmanager
def preserve_multimodal_prefix_input_ids(model: Any, input_ids: Any) -> Iterator[None]:
    """Temporarily preserve placeholder IDs on the generation prefill call.

    The patch is instance-local and is always removed on exit.  It does not
    alter model/configuration/parameter state or the cached continuation path.
    """
    original_prepare = model.prepare_inputs_for_generation

    @wraps(original_prepare)
    def repaired_prepare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        model_inputs = original_prepare(*args, **kwargs)
        if (model_inputs.get("inputs_embeds") is not None
                and _is_prefill(model_inputs)
                and model_inputs.get("input_ids") is None):
            model_inputs["input_ids"] = input_ids
        return model_inputs

    model.prepare_inputs_for_generation = repaired_prepare
    try:
        yield
    finally:
        model.prepare_inputs_for_generation = original_prepare


def generate_with_multimodal_prefix(model: Any, *, input_ids: Any, **kwargs: Any) -> Any:
    """Call ``generate`` while retaining Phase 3O's multimodal ID contract."""
    with preserve_multimodal_prefix_input_ids(model, input_ids):
        return model.generate(input_ids=input_ids, **kwargs)
