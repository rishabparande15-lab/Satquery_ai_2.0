# Phase 3O.15 — Bounded contrastive objective ablation

## Design and integrity

Phase 3O.15 compared five preregistered projector-only objectives from the same immutable Phase 3O.12 final projector (fingerprint `168c6e64264369d0385697c5c008fd1424ee84fe1ffd12ca9079943e45cd036b`). Qwen revision `66285546d2b821cf421d4f5eb2576359d3770cd3` remained frozen. Every variant used the same 8/8/8 TRAIN panel, 4/4/4 validation micro-panel, deterministic same-task maps, seed, AdamW configuration, precision, and eight three-task microbatches.

The variants were: A, base correct-image CE only; B, the Phase 3O.14 hinge (weight 0.25, target margin 0.25); C, lower-weight hinge (0.10, target margin 0.25); D, smooth sequence-likelihood separation (`0.25 * softplus(score_shuffled - score_correct)`); and E, that smooth term only for Caption while Binary/MCQ retained CE.

All variants score full causal authoritative target sequences, including multi-token targets and EOS. Every loss was finite, projector gradients were nonzero, Qwen gradient tensors were zero, and the Phase 3O.12 checkpoint remained unchanged. TEST access was zero.

## Validation margin changes

| Variant | Binary | MCQ | Caption | Caption unique pre → post | Classification |
| --- | ---: | ---: | ---: | ---: | --- |
| A Base CE | +0.06482 | +0.00258 | -0.03213 | 1 → 1 | TRAIN_ONLY_EFFECT |
| B Hinge 0.25 | +0.03783 | -0.00243 | -0.07309 | 1 → 2 | TRAIN_ONLY_EFFECT |
| C Hinge 0.10 | +0.04367 | -0.01001 | +0.02885 | 1 → 2 | TRAIN_ONLY_EFFECT |
| D Smooth likelihood | +0.05648 | -0.00190 | -0.01550 | 1 → 1 | TRAIN_ONLY_EFFECT |
| E Caption-only smooth | +0.02227 | -0.00129 | +0.03376 | 1 → 1 | TRAIN_ONLY_EFFECT |

No variant met the preregistered promising gate: at least two validation task margins must improve by more than 0.001, no task may materially worsen (less than -0.001), and Caption diversity may not degrade. All variants had at least one materially worse validation margin. No post hoc winner was selected.

The validation output micro-check is diagnostic only, not benchmark evaluation. Caption correct-versus-shuffled output differences remained limited; variants B and C raised observed Caption unique outputs from one to two, but both failed the all-task validation-margin gate.

## Conclusion

Phase status is `PHASE3O15_COMPLETE`; overall conclusion is `NO_VARIANT_JUSTIFIED_FOR_FULL_TRAINING`. The ablation validates the loss implementations and frozen-projector workflow, but provides no reliable objective with simultaneous Binary, MCQ, and Caption validation-margin benefit.

Do not start full training or Phase 3O.16 automatically. If separately authorized, Phase 3O.16 should be a new bounded design study—not a scale-up—testing task-specific objective formulations with a preregistered caption quality/collapse gate and independently sized diagnostic panels. More data, LoRA, and partial-Qwen adaptation remain unjustified by this result.
