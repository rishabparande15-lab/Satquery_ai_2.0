# Phase 3P.2: SAR supervision and generation diagnosis

Status: `PHASE3P2_COMPLETE`. This was inference-only on the final Phase 3P.1
projector and the identical 3 Binary / 3 MCQ / 3 Caption validation identities.
No optimizer, backward pass, parameter update, S2 input, fusion, or TEST record
was used.

## Binary and MCQ priors

Binary choices did not change under either shuffled or zero SAR: the selections
were `no, yes, no` in all three conditions. Score margins moved, but did not
produce a correct-SAR decision advantage. The first positive target was instead
ranked below `no` under all conditions (-0.798 correct, -0.868 shuffled,
-0.383 zero); the other positive and the negative were selected under all
conditions.

MCQ correct-SAR target ranks were 1, 4, and 4, exactly matching shuffled SAR.
Zero SAR changed rankings on two records, including one target from rank 4 to
rank 1, but this is not a correct-image benefit. The largest correct-SAR
target-vs-best-distractor margins were +3.311, -7.202, and -10.751. This is
`MCQ_SAR_DISTRACTOR_PRIOR_DOMINANCE`, not useful SAR identity selection.

## Caption and representation trace

Correct and shuffled SAR produced the same generic prefix, “Sentinel-1 radar
scene of the city of”, for two captions; only one of three changed under
shuffling. All three zero-SAR outputs collapsed to empty. Correct-minus-shuffled
reference likelihood was positive once and negative twice (+0.725, -0.956,
-0.594), so it does not consistently prefer the correct image.

CROMA correct-vs-shuffled L2 differences were substantial (354.5–486.0), as
were projector/hidden/first-answer-logit changes. However, those natural-image
changes yielded zero Binary decision changes and zero MCQ ranking changes. By
contrast, correct-vs-zero hidden differences were roughly 77.6–90.5 and
first-answer-logit L2 750–1496, with 2/3 MCQ ranking changes and 3/3 caption
output changes. The model detects SAR presence much more strongly than SAR
identity.

## Supervision audit and conclusion

The 18 training records were conservatively audited against SAR-only semantic
support. Binary: 0 supported, 3 partially supported, 3 not established. MCQ:
0 supported, 3 partially supported, 3 not established. Caption: 0 supported,
0 partially supported, 6 not established. The labels originate from optical /
reference-map semantics such as topology, area, country, climate, and season;
they are not established as SAR-only targets.

Primary failure modes are `SAR_LABEL_MODALITY_MISMATCH`,
`SAR_PRESENCE_SIGNAL_WITH_WEAK_IDENTITY_USE`, `SAR_LANGUAGE_PRIOR_DOMINANCE`,
`SAR_DISTRACTOR_PRIOR_DOMINANCE`, and `SAR_SUPERVISION_TOO_SMALL`.

Recommendation: `SAR_PROJECTOR_TRAINING_NOT_JUSTIFIED_WITH_CURRENT_LABELS`.
Larger standalone SAR training is not justified. SAR routing and S1+S2 fusion
remain blocked. The exact next project step is to build and provenance-audit a
separate SAR-native supervision pool before considering any further SAR model
adaptation; do not train from the present optical-derived labels.
