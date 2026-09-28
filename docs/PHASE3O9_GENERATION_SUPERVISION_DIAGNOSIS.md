# Phase 3O.9 Generation Processing and Supervision Diagnosis

Status: PHASE3O9_COMPLETE_DEFECT_FOUND

## Supervision composition

The authoritative Phase 3O.5 training manifest contains 100 unique images:
68 binary VQA, 32 MCQ, and 0 captions. Binary labels are exactly balanced
(34 yes, 34 no). MCQ answer keys are A=10, B=8, C=8, D=6; every MCQ has four
options. There are 98 distinct questions and two repeated question strings.

The original validation manifest contains 16 binary VQA, 4 MCQ, and 0 captions,
matching Phase 3O.6. Its binary labels are yes=10 and no=6; MCQ keys are
A=1, B=1, D=2. Thus the pilot had no caption supervision and a 68/32 binary/MCQ
task imbalance. These counts alone do not prove the observed behavior, but
they preclude any captioning-capability conclusion.

## First-step generation trace

For each of five Phase 3O.8 caption probes:

- Direct forward on the supplied generation prefix ranked EOS below top-1
  (for the first probe, EOS rank 2; top-1 was the token for Sentinel).
- Generation’s processed score ranked EOS first and emitted EOS immediately.
- The actual logits-processor list was empty for every probe. No logits
  processor, warper, forced EOS, suppression list, repetition penalty, or
  cache setting promoted EOS.

The material difference is input preparation. Direct forward received both
the placeholder input IDs and the inserted input embeddings. On the first
model call from generate(), inputs_embeds, attention_mask, cache_position,
dtype, device, and sequence length were retained, but input_ids was absent.
This is a GENERATE_INPUT_PREPARATION_MISMATCH and explains why direct-forward
scores cannot be treated as the scores used by generate().

## Cache and special tokens

The use_cache=true and use_cache=false probe produced identical first and
second generated tokens and EOS ranks for 2 binary, 2 MCQ, and 2 caption
examples. No cache-path inconsistency was found.

EOS is 151645 (<|im_end|>). PAD and generation BOS are both 151643. The
generation configuration lists eos_token_id as [151645, 151643], has no forced
EOS/BOS, no suppression lists, no forced decoder IDs, and no custom processor.
The EOS/PAD/BOS configuration was recorded but no accidental aliasing or
processor defect was established.

## Decision

Classifications:

- B_GENERATE_INPUT_PREPARATION_MISMATCH
- E_CAPTION_UNSUPERVISED
- F_TRAINING_TASK_IMBALANCE
- G_VISUAL_SIGNAL_WEAKLY_LEARNED

This phase found a concrete generation-input mismatch but applied no fix:
changing the historical generation interface without a dedicated regression
test and tiny smoke validation would invalidate comparison with Phase 3O.7.
No optimizer was created, no parameter changed, test access was zero, and the
Phase 3O.5 checkpoint remained unchanged.

Focused existing tests passed: 18/18 for the Phase 3O adaptation test module.

Recommended next step: create a separately versioned, minimal generation-input
interface repair with a regression test that demonstrates pre-fix
direct-forward/generate divergence and post-fix equivalence, then run only a
tiny fixed-panel smoke validation. Do not retrain or scale the dataset yet.
