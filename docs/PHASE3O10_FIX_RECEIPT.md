# Phase 3O.10 Fix Receipt

Status: `PHASE3O10_COMPLETE_FIX_VALIDATED`  
Classification: `B_GENERATE_INPUT_PREPARATION_MISMATCH_FIXED`

## Defect and root cause

Before the repair, the Hugging Face generation preparation path dropped
`input_ids` when `inputs_embeds` was present on the initial Qwen prefill. The
Phase 3O S2 contract needs both the image-placeholder IDs and the projected
embedding rows. Direct forward preserved both, while stock `generate()` did
not. For each of the five caption probes, stock first-step EOS rank was 1;
direct-forward EOS rank was 2.

## Minimal repair

Changed files:

- `src/eo_vlm/generation_interface.py`: scoped, instance-local generation
  wrapper that restores original IDs only at multimodal prefill.
- `scripts/phase3o10_generation_interface_repair.py`: non-test regression and
  15-sample smoke audit.
- `tests/test_phase3o10_generation_interface.py`: deterministic regression
  fixture.

The wrapper does not edit Transformers source, checkpoints, Qwen weights,
projector weights, decoding configuration, or the cached continuation path.
The pre-fix behavior remains reproducible by calling `model.generate()`
directly without the versioned wrapper.

## Evidence

Across the fixed 15-sample panel, pre-fix initial calls had no `input_ids`;
post-fix calls had IDs exactly equal to direct forward. `inputs_embeds` and
attention masks were exactly equal in all repaired calls; visual slots were
unchanged at positions 1--16. Repaired-versus-direct first-step logits were
exactly equal for every record: maximum L2 `0.0`, maximum absolute difference
`0.0`, against tolerance `1e-5`.

For captions, EOS rank changed from 1 before the fix to 2 after the fix for
all five probes. Smoke outputs changed/unchanged and correct-before/after were
binary `2/3`, `4/4`; MCQ `1/4`, `1/2`; captions empty-before/empty-after were
`5/0` (all five outputs changed). These are repair-smoke observations, not
benchmark results.

## Immutability and limitations

The Phase 3O.5 checkpoint SHA-256 remains
`e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db`; the
projector fingerprint remains
`db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46`.
Qwen, projector, and historical Phase 3O artifacts were hash-unchanged; TEST
access count was zero; no optimizer/backward pass existed. Captions have no
Phase 3O.5 supervision and no captioning, benchmark, generalization, SAR,
temporal, grounding, or optical-SAR claim follows from this repair.
