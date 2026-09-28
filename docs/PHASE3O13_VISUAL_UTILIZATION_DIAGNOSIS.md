# Phase 3O.13 — Visual utilization and modality-sensitivity diagnosis

## Scope and integrity

Phase 3O.13 is a read-only diagnostic of the completed Phase 3O.12 model. It used the verified final projector (`5e3bc79d5cf1ee194db433503360cdf7530bb5fc908a3d936de7c43b449933c0`; fingerprint `168c6e64264369d0385697c5c008fd1424ee84fe1ffd12ca9079943e45cd036b`) and frozen Qwen revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.

No training, optimizer, backward pass, parameter update, TEST access, SAR, temporal, grounding, optical-SAR, or hybrid representation was used. The final identity checks after the diagnostic reconfirmed that Qwen and the checkpoint remained unchanged. TEST access count is zero.

## Panel and conditions

The deterministic panel is the first 10 record-ID-sorted records per task from the existing 3O.12 validation manifest: 10 Binary, 10 MCQ, and 10 Caption records. It contains no TRAIN or TEST records. For every record, decoding and text prompts were held fixed while comparing the correct S2 image, the existing deterministic shuffled image, an all-zero S2 tensor, and the mean of the correct image's 16 projected tokens repeated at all visual positions.

## Projector sensitivity

The projector is strongly image-sensitive. Mean correct-versus-shuffled projector-output L2 was 43.22 (Binary), 45.05 (MCQ), and 40.41 (Caption); correct-versus-zero L2 was 955.27, 951.28, and 950.00 respectively. Thus the learned projector is not constant across natural S2 images.

## Layer and logit sensitivity

At the first answer position, natural-image differences were very small at the early transformer layer (correct-versus-shuffled L2: Binary 0.0081, MCQ 0.0056, Caption 0.0084), then increased modestly by the final layer (1.80, 1.13, 1.57). Zero-image differences were much larger at the final layer (108.10, 76.08, 123.24).

First-answer full-logit-vector L2 for correct versus shuffled was 10.66 (Binary), 7.45 (MCQ), and 10.93 (Caption), compared with 738.12, 480.51, and 925.20 for correct versus zero. So visual differences do reach the LLM hidden states and logits, but natural-image changes are much smaller than the zero-image intervention.

## Output and task-specific utilization

| Task | Shuffled output changes | Zero-image output changes | Mean-token output changes |
| --- | ---: | ---: | ---: |
| Binary | 0/10 | 6/10 | 0/10 |
| MCQ | 0/10 | 8/10 | 0/10 |
| Caption | 6/10 | 10/10 | 0/10 |

Binary and MCQ greedy outputs were completely invariant to natural-image shuffling and to mean-token replacement. Caption text changed for 6/10 natural-image shuffles, but its distribution remained two modes. This is task-specific visual utilization: Caption is more output-sensitive than Binary/MCQ, although all three tasks are highly sensitive to zero imagery.

The zero-image control is not evidence of strong natural-image discrimination: it is an off-manifold intervention whose projector representation is far from every observed natural representation. For Binary and MCQ, identical correct/shuffled outputs and near-zero correct-minus-shuffled target margins are stronger evidence for weak discrimination among valid imagery.

## Candidate margins

Single-token answer mappings were unambiguous for the selected Binary and MCQ candidates. Mean target margins were 0.0453 for Binary under both correct and shuffled images (difference 0.0000), and -0.2188 versus -0.2219 for MCQ (correct-minus-shuffled 0.0031). Logit differences therefore did not become decision-relevant target-margin separation on this panel.

## Caption collapse

The 200 training captions have 192 unique exact targets; the largest exact-target frequency is only 3. This rejects simple exact-caption target imbalance as a sufficient explanation for the two generated modes.

Across the 10 Caption-panel samples, correct, shuffled, and mean-token conditions each produced the same two generic modes: “The image shows a rural area with a mix of green and brown fields” (6) and “The image shows a rural area” (4). Zero imagery produced 10 empty captions. The evidence is most consistent with generation mode collapse and weakly discriminative visual conditioning, rather than a lack of target diversity. It does not establish causal language-prior dominance.

## Attention

`ATTENTION_ANALYSIS_UNAVAILABLE`: full attention tensors were not requested because the loaded Qwen attention implementation cannot guarantee compact exposure and materializing all tensors would create an unreasonable memory risk. No model configuration was altered merely to extract them.

## Classification and decision

Evidence-backed classifications:

- `A_VISUAL_SIGNAL_STRONG_AT_PROJECTOR_WEAK_AT_LLM`
- `C_VISUAL_SIGNAL_REACHES_LOGITS_BUT_NOT_DECISION_MARGIN`
- `E_CAPTION_MODE_COLLAPSE`
- `F_TASK_SPECIFIC_VISUAL_UTILIZATION`

The recommended intervention is an image-contrastive or shuffle-aware objective together with a caption-specific objective. More data alone is not the indicated first intervention, and partial-Qwen adaptation/LoRA is not recommended from this evidence: the frozen interface transmits measurable visual signal to logits, but the present objective does not make natural-image differences decision-relevant.

Recommended next phase: **Phase 3O.14, controlled image-conditioned objective design**. It was not started.
