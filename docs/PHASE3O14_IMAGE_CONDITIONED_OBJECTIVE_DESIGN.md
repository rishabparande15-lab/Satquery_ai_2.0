# Phase 3O.14 — Controlled image-conditioned objective design

## Scope and preregistration

Phase 3O.14 designed and smoke-tested a small projector-only objective in a new `phase3o14_objective_design` namespace. It began from the verified Phase 3O.12 final projector, rather than a fresh initialization, because the purpose was to test whether the proposed objective could increase natural-image discrimination from the existing learned state. The historical Phase 3O.12 checkpoint remained unchanged (SHA-256 `5e3bc79d5cf1ee194db433503360cdf7530bb5fc908a3d936de7c43b449933c0`; fingerprint `168c6e64264369d0385697c5c008fd1424ee84fe1ffd12ca9079943e45cd036b`).

Qwen remained frozen at revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. No LoRA, TEST access, SAR, temporal, grounding, optical-SAR, or hybrid representation was used.

Before the smoke run, the following configuration was fixed: base LM weight 1.0, contrastive weight 0.25, hinge margin 0.25, equal task weights, AdamW learning rate 0.0001 with weight decay 0.0001, seed 3014, and four smoke steps. The train panel contained 4 Binary, 4 MCQ, and 4 Caption examples. The separate validation micro-panel contained 2 examples per task and was never passed to the optimizer.

## Objective

For each authoritative target sequence `y` and fixed prompt `p`, the model scores every existing supervision token—including EOS where present—using its causal log probability:

`s(image) = sum_t log P(y_t | image, p, y_<t)`

The smoke loss was:

`L = L_CE(correct) + 0.25 * mean(max(0, 0.25 - (s(correct) - s(shuffled))))`

This does not assume `yes`, `no`, or MCQ letters are one token. Binary, MCQ, and Caption use the exact existing supervision answer format; multi-token targets are scored as full causal sequences.

## Pairs and integrity

A deterministic train-only shuffle map was persisted for all 600 Phase 3O.12 training records. Each negative is a different-image sample from the same task and TRAIN partition. No validation or TEST record was used to create a training negative. The validation micro-panel uses its own deterministic validation-only pairing solely for pre/post read-only measurement.

All four steps were finite. Projector gradient norms were 0.5151, 1.2817, 0.7756, and 0.7964; Qwen gradient tensors were zero at every step. The Phase 3O.12 checkpoint and Qwen state matched their pre-run identities after the smoke run. TEST access remained zero.

## Margin results

| Task | Train pre | Train post | Train change | Validation micro change |
| --- | ---: | ---: | ---: | ---: |
| Binary | 0.00511 | 0.00390 | -0.00121 | +0.00087 |
| MCQ | 0.00420 | 0.00700 | +0.00280 | -0.00159 |
| Caption | -0.00930 | -0.00745 | +0.00185 | -0.04604 |

The run lowered base loss (1.1172 to 1.0193), but the aggregate correct-minus-shuffled score margin was near zero and did not improve monotonically across the four steps. Improvement on the TRAIN panel was task-dependent and did not transfer consistently to the validation micro-panel.

## Caption micro-check

The four pre-smoke Caption generations had two unique outputs. After smoke training, all four became the same generic output: “The image shows a rural area with a mix of agricultural and forested land,” yielding one unique output. This is evidence that the small objective and four-step configuration did not remedy caption collapse; it worsened diversity in this micro-check.

## Classification and decision

The classification is `OBJECTIVE_MOVES_TRAIN_MARGIN_ONLY`, not `OBJECTIVE_PROMISING`. The implementation is valid and stable, reaches the projector, preserves frozen Qwen, and respects the partitions. However, it does not produce consistent margin improvement across tasks or held-out micro measurements, and it does not improve Caption diversity.

Do not start full Phase 3O.15 training. The next intervention should be a separately designed, preregistered Phase 3O.15 ablation of objective formulation and task weighting on similarly bounded train-only smoke panels. It should compare sequence-level contrastive variants and a caption-specific image-conditioned likelihood treatment before any full training is considered. More data and partial-Qwen adaptation are not justified by this smoke result.
