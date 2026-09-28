# Phase 3O.11 Post-fix Semantic and Image-dependence Re-audit

Status: `PHASE3O11_COMPLETE`  
Generation interface: `phase3o10_multimodal_prefix_v1`

This is an exact re-run of the immutable Phase 3O.7 validation identities and
task-local shuffle mapping, using only the Phase 3O.10 generation wrapper.
The panel SHA-256 is
`f59dc04057fbb1de2f917ff6a631643666eb7a0172cd3393c1a0eb060fd2a074` and
the shuffle-map SHA-256 is
`de10a2118511bd07bd26d893ccf929b864b70b5121cb279437e98fcb49920955`.
Both contain 30 binary, 30 MCQ, and 30 caption validation records.

## Classification results

| Task | Correct image | Shuffled image | Normal minus shuffled |
| --- | --- | --- | --- |
| Binary | 16/30 (0.5333), unparsable 0 | 16/30 (0.5333), unparsable 0 | 0.0 |
| MCQ | 8/30 (0.2667), unparsable 2 | 7/30 (0.2333), unparsable 2 | 0.0333 |

Binary paired outcomes were 16 correct/correct and 14 incorrect-or-unparsable
/incorrect-or-unparsable. MCQ paired outcomes were 7 correct/correct, 1
correct/incorrect-or-unparsable, and 22 incorrect-or-unparsable/
incorrect-or-unparsable.

For context only, historical Phase 3O.7 normal/shuffled results were binary
17/30 (0.5667) and 16/30 (0.5333), and MCQ 8/30 (0.2667) and 8/30 (0.2667).
The repair changed 23 binary outputs and 19 MCQ outputs on the normal
condition, but output changes alone are not image-dependence evidence.

## Caption and degeneracy audit

All 30 captions were nonempty in both conditions, compared with 30/30 empty
in each Phase 3O.7 condition. All terminated through EOS, with no prompt echo,
punctuation-only output, or within-output repetition. However, each condition
collapsed to the same generic sentence for all 30 records: “Sentinel-2 is a
satellite that takes images of the Earth.” Normal and shuffled captions were
identical for all 30 pairs. This is structurally healthier generation but is
not learned captioning and shows no task-output image sensitivity.

Across all tasks, the only residual empty/malformed classification behavior was
two empty malformed MCQ answers per condition; abnormal stopping was zero.

## Decision

Classification: `WEAK_IMAGE_DEPENDENCE`.

The fixed interface removes pathological immediate-empty generation, but
correct-image versus shuffled-image results are identical for binary and only
one MCQ response differs in correctness. This does not establish meaningful
dependence on correct S2 imagery. The recommended next phase is a newly
preregistered, task-balanced S2-projector-only adaptation experiment, not a
continuation of Phase 3O.5. It must use frozen Qwen, the fixed Phase 3O.10
interface, balanced binary and MCQ supervision plus captions, deterministic
initial/final projector checkpoints and fingerprints, train/validation
separation, pretrained-versus-untrained comparison, normal-versus-shuffled
audit, and zero TEST access.

## Immutability

The Phase 3O.5 checkpoint SHA and projector fingerprint remained respectively
`e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db` and
`db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46`.
Qwen, projector, and every protected historical artifact were hash-unchanged.
No optimizer, backward pass, training, or TEST access occurred.

The detailed raw outputs and reference pairs are in
`artifacts/training/phase3o/phase3o11_postfix_semantic_audit.json`.
