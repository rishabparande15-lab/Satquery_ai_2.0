# Phase 3O.6 Held-Out Semantic Generation Audit

Status: PHASE3O6_COMPLETE

This is an internal controlled validation audit, not benchmark evaluation. The
Phase 3O.5 checkpoint was used read-only with deterministic autoregressive
generation on the existing 20-record validation selection only.

## Results

- Validation records: 20; train/validation image overlap: 0; test access: 0.
- Task distribution: 16 binary VQA, 4 multiple-choice VQA, 0 captions.
- Binary VQA: 9/16 exact normalized correct (0.5625); 7 incorrect; 0 unparsable.
- Multiple-choice VQA: 1/4 exact correct (0.25); 2 incorrect; 1 unparsable.
- Captions: none in the fixed validation selection, so no caption metric or
  descriptive caption audit applies.
- Degenerate-output check: one empty generation (the unparsable MCQ); outputs
  were not all the same; no prompt echo, punctuation-only output, or excessive
  repetition was detected.

## Integrity and provenance

- Checkpoint SHA-256 verified:
  e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db.
- Projector model-state fingerprint verified:
  db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46.
- Pinned Qwen revision verified:
  66285546d2b821cf421d4f5eb2576359d3770cd3.
- Qwen was fully frozen and unchanged; the loaded projector was unchanged;
  the checkpoint SHA was unchanged after inference.
- Every validation S2 asset was checked against its historical compact-JSON
  per-band source hash before generation. No optimizer was created and zero
  gradient updates occurred.

## Generation configuration

Greedy deterministic decoding used max_new_tokens=16, do_sample=false,
temperature=null, top_p=null, top_k=null, repetition_penalty=1.0, and
use_cache=true. The prompt was Question: {question}\nAnswer: with tokenizer
special tokens enabled and no chat template.

The original exact untrained projector initialization is not reconstructible
from the retained provenance, so the trained-versus-untrained comparison is
UNTRAINED_BASELINE_UNAVAILABLE.

## Conservative interpretation

The Phase 3O.5 S2-adapted projector produced measurable held-out generation
behavior on this controlled internal validation audit. These results do not
establish benchmark performance, generalization, or any claim about other
datasets, SAR, temporal reasoning, grounding, or native multispectral
superiority.

The complete per-sample evidence is in
artifacts/training/phase3o/phase3o6_semantic_audit.json.
