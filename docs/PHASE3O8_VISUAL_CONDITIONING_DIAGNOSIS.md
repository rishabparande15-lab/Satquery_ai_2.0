# Phase 3O.8 Visual Conditioning Diagnosis

Status: PHASE3O8_COMPLETE

This diagnostic is read-only: it used the exact Phase 3O.5 projector
checkpoint and pinned frozen Qwen revision, did not access test data, did not
create an optimizer, and made no parameter update.

## Identity and panel

- Checkpoint SHA-256 verified:
  e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db.
- Projector fingerprint verified:
  db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46.
- Qwen revision: 66285546d2b821cf421d4f5eb2576359d3770cd3; trainable
  parameters: zero.
- Diagnostic panel: 5 binary, 5 MCQ, and 5 caption records from the immutable
  Phase 3O.7 validation panel, with their deterministic shuffled partners.
- Test access: zero.

## Actual token path

The observed input layout is:

    [vision_start][visual_0 ... visual_15][vision_end][Question/Answer prompt]

Visual tokens occupy inclusive indices 1–16, vision-start is index 0, and
vision-end is index 17. Every diagnostic attention mask was all ones, so all
16 visual tokens were unmasked and occurred before prompt and generated-answer
positions. Qwen’s causal decoder can attend from later language positions to
this visual prefix.

Training uses the same prefix and exactly the same input IDs, embeddings, and
attention-mask prefix as inference. Training alone appends answer tokens and
EOS and labels them; generation instead decodes autoregressively with cache.

## Conditioning sensitivity

Correct-versus-shuffled S2 images changed projector outputs for every
diagnostic sample: projector-output L2 difference ranged from 9.9171 to
18.1966 (mean 14.1957). The final Qwen input embeddings changed by the same
amount inside visual slots, while the text-slot difference was exactly zero.

The change propagated into final-layer answer-position hidden states: L2
difference ranged from 1.0102 to 4.3967 (mean 2.1669). First-answer logit
vectors also changed: L2 difference ranged from 6.3480 to 25.0577 (mean
14.9872). Thus the S2 projector output is inserted, visible, and causally
affects Qwen computations; visual-token insertion, masking, and position
construction are not implicated.

## Training labels and gradient path

Three training-format examples had S2 input [1,12,120,120], projector output
[1,16,2048], and final embedding sequences of lengths 109, 57, and 38. Their
only supervised positions were answer tokens plus EOS: 2 positions each,
starting at 107, 55, and 36 respectively. Prefix/prompt/visual and padding
positions were ignored with the loss ignore index.

The one permitted no-step backward diagnostic produced loss 1.0039007664,
projector gradient norm 3.0647085629, and nonzero gradients on all 8 projector
parameter tensors. Qwen gradient tensors remained zero because Qwen was
frozen. No optimizer or checkpoint was created.

## Caption diagnosis

Phase 3O.5 training contained zero caption records. On all five diagnostic
caption prompts, raw forward EOS probability was 0.1818–0.2261 and EOS was not
raw-logit top-1. However, the generation API’s processed first-step scores
ranked EOS top-1 with probability 0.3732–0.4284, and generation emitted EOS as
the first token every time. This establishes immediate generation stopping for
the caption interface. It does not, by itself, prove a source-code defect:
the discrepancy must be diagnosed at Qwen generation-preparation/logits
processor level before any implementation change.

## Classification and next step

Evidence-backed categories are:

- F_GENERATION_STOPPING_OR_EOS_ERROR
- G_VISUAL_SIGNAL_PRESENT_BUT_WEAKLY_USED

No projector collapse, token insertion fault, attention-mask/position fault,
training/inference prefix mismatch, or label-mask error was established. No
implementation defect was declared and no fix was applied.

The next authorized work should be a narrowly scoped Qwen generation
preparation/logits-processor and caption-supervision interface diagnosis,
followed only by a tiny smoke validation if a specific defect is demonstrated.
Do not scale training yet.

Focused existing tests with the repository interpreter passed 18/18. No new
tests were added because no concrete code defect was established.
