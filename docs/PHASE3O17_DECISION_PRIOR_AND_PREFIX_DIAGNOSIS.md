# Phase 3O.17: Decision-Prior and Prefix Diagnosis

Status: `PHASE3O17_COMPLETE`. This was a read-only Phase 3O.12 diagnostic:
no optimizer, backward pass, training, parameter update, LoRA, or TEST access.
Historical Phase 3O.16 and recovery Phase 3O.16R are not merged with this evidence.

## MCQ: candidate-prior diagnosis

The validation-only panel had 24 records, excluding prior Phase 3O.16 and
3O.16R identities. Candidate likelihoods were measured as complete causal
sequences under correct, shuffled, zero, mean, and neutralized visuals.

Correct imagery recovered the target at rank one for 4/24 samples. Correct and
shuffled full rankings agreed for 21/24 samples; correct and zero rankings
agreed for 3/24. Option `b` was the top candidate in 17/24 correct-image
examples. Subgroups were: image changes scores but not ranking 15, target
recovery 4, distractor always dominates 3, and ranking changes not to target 2.

The supported root cause is `MCQ_DISTRACTOR_PRIOR_DOMINANCE`, with visual
signals usually changing scores without changing the candidate decision.

## Caption: prefix diagnosis

The 24-record validation panel had one unique token 1, one unique three-token
prefix, one unique five-token prefix, and one unique complete generation.
The largest three-token prefix mode occurred 24/24 times. Mean teacher-forced
correct-versus-shuffled logit sensitivity was 29.04; free-generation
sensitivity was 22.37. This supports `CAPTION_AUTOREGRESSIVE_PREFIX_LOCK_IN`:
image-dependent logit variation persists under teacher forcing but does not
prevent a shared autoregressive prefix and identical free generations.

## Binary: boundary diagnosis

On 12 new validation records, mean distance to the YES/NO sequence-score
boundary was 0.47272, while the mean natural-image-induced margin shift was
0.02605. The shift is much smaller than the boundary distance, supporting the
interpretation that natural-image score motion generally does not cross the
decision boundary.

## Integrity and recommendation

Qwen and the Phase 3O.12 projector were unchanged; TEST access was zero.
The evidence supports task-specific future study, not Qwen adaptation:
Phase 3O.18 should, if separately authorized, preregister an MCQ
candidate-ranking/distractor-prior control together with caption
prefix-conditioned image supervision. Do not start Phase 3O.18 here.
