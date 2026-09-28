# Phase 3O.10 Generation Interface Repair

Status: `PHASE3O10_COMPLETE_FIX_VALIDATED`

Phase 3O.9 established that Qwen's stock `generate()` prefill supplied the
projected `inputs_embeds` and attention mask but omitted the supplied
placeholder `input_ids`. The direct-forward contract supplied both. This
versioned, inference-only repair is implemented in
`src/eo_vlm/generation_interface.py` as
`phase3o10_multimodal_prefix_v1`.

The wrapper temporarily wraps the model instance's
`prepare_inputs_for_generation` during a single call to `generate()`. On the
initial `inputs_embeds` prefill only, it restores the caller's original
placeholder IDs if the stock preparation removed them. It restores the
original preparation method in `finally`; cached continuation calls and model,
configuration, projector, and checkpoint state are untouched.

The deterministic 15-record non-test panel used five binary, five MCQ, and
five caption identities from the established Phase 3O.8/3O.7 diagnostics.
Stock generation omitted IDs in every initial call; repaired generation kept
them in every call. Projected embeddings, attention masks, sequence lengths,
visual slots (1--16), dtype, and device matched direct forward exactly.
The repaired first-step logit comparison against direct forward was L2 `0.0`
and maximum absolute difference `0.0` for every record (tolerance `1e-5`).

Caption probes were interface probes only: all five were empty before and all
five became nonempty after the repair. Phase 3O.5 trained on zero caption
examples, so this is not evidence of learned captioning.

The machine-readable receipt is
`artifacts/training/phase3o/phase3o10_generation_interface_repair.json`.
No TEST data were read; no optimizer or backward pass was created. Qwen and
projector state hashes, the checkpoint SHA-256, and historical artifact hashes
were unchanged.
