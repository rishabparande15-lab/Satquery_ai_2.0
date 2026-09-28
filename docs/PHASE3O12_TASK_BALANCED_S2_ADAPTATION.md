# Phase 3O.12 — Task-balanced Sentinel-2 projector adaptation

## Objective

Phase 3O.12 tested whether fresh, task-balanced Binary/MCQ/Caption adaptation of a Sentinel-2 projector improves semantic behavior and dependence on the correct Sentinel-2 imagery. This is an internal controlled validation experiment, not a benchmark evaluation.

## Dataset and controls

The preregistered manifest contains 200 Binary, 200 MCQ, and 200 Caption records for TRAIN (600 total), and 50 records per task for VALIDATION (150 total). TEST count and access were zero. The train/validation image overlap was zero. The deterministic, seed-3012 validation shuffle map contains 150 records and its sample membership exactly matches the validation manifest.

The final read-only integrity audit verified the source hashes pinned in `preregistered_config.json`, the registered task counts, and all expected artifacts. No SAR, temporal, grounding, optical/SAR, or hybrid-830d task was included.

## Model and initial state

Qwen was frozen at revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. A fresh `S2MultispectralProjector(patch_hidden_size=128)` was the only trainable model component (5,549,184 parameters). The fixed Phase 3O.10 generation interface was `phase3o10_multimodal_prefix_v1`.

The initial projector identity was independently recomputed and matched its pretraining receipt:

- SHA-256: `50e8857ba7b9e5210de7757fd596e28112e0e82bfadb0957ff6e127fe8fabc2e`
- State fingerprint: `948f3e8785141e51edb2e79431958975ae8ea190e88701865fdfc2cd22eb976a`

Before training, both normal and shuffled Binary/MCQ accuracy were 0/50. Captions were 50 empty, 0 nonempty, with one unique output in each condition.

## Supervision diagnostic

The diagnostic established two supervised tokens for Binary and MCQ answers, 33 supervised Caption tokens, and supervised Caption EOS. All 8 projector tensors received gradients; Qwen gradient tensors were 0. The diagnostic created no optimizer and performed no optimizer step, and it left the initial projector hash and fingerprint unchanged.

## Training

Training used 600 records for one epoch, batch size 1, gradient accumulation 8, and 75 AdamW optimizer steps. Mean loss was 2.4698194989437856; initial and final observed training losses were 3.6719655990600586 and 0.7041215300559998. Per-task training losses were 1.5739649171754717 (Binary), 1.9686256928741932 (MCQ), and 3.8668678867816926 (Caption).

A historical preflight launch failed before record 1 and optimizer step 1 because tensor-equality was used for optimizer-membership checking. This was a non-scientific software defect: no model state or data record was touched. It was corrected to identity-based membership, with regression coverage in `tests/test_phase3o12_optimizer_identity.py`.

## Validation and reproducibility

The original validation mean was 1.179943935573101: 0.4255617192387581 for Binary, 0.6773896324634552 for MCQ, and 2.4368804550170897 for Caption.

A fresh-process reload gave the same mean, 1.179943935573101. The absolute difference was 0.0 and agreement was `EXACT_MATCH`.

The final projector was recomputed after the audit and remains:

- SHA-256: `5e3bc79d5cf1ee194db433503360cdf7530bb5fc908a3d936de7c43b449933c0`
- State fingerprint: `168c6e64264369d0385697c5c008fd1424ee84fe1ffd12ca9079943e45cd036b`

It differs from the initial projector and matches the training and reload receipts. The final audit also verified Qwen trainable parameters = 0, Qwen optimizer parameters = 0, Qwen gradient tensors = 0, and zero recorded TEST access across registration, training, diagnostics, reload, and both semantic audits.

## Semantic results

| Task | Untrained normal | Trained normal | Trained shuffled | Normal − shuffled |
| --- | ---: | ---: | ---: | ---: |
| Binary | 0/50 (0.00) | 27/50 (0.54) | 26/50 (0.52) | 0.02 |
| MCQ | 0/50 (0.00) | 13/50 (0.26) | 13/50 (0.26) | 0.00 |

For Caption, the untrained projector produced 50 empty and 0 nonempty outputs. The trained projector produced 50 nonempty outputs with only 2 unique outputs. Of the 50 paired normal/shuffled caption outputs, 24 were identical and 26 changed.

## Conservative interpretation

Task-balanced projector training produced a clear improvement in structural semantic behavior relative to the immutable untrained projector: binary and MCQ outputs became parseable and captions became nonempty. However, correct-versus-shuffled S2 separation remained weak at the task output level. Binary accuracy differed by only 0.02 and MCQ accuracy did not differ between normal and shuffled imagery. Caption generation remained strongly collapsed to only two unique outputs. Therefore Phase 3O.12 establishes semantic learning but does not establish strong dependence on the correct Sentinel-2 image.

The evidence classification is `SEMANTIC_BEHAVIOR_IMPROVES_BUT_IMAGE_DEPENDENCE_REMAINS_WEAK`.

## Limitations

- Internal controlled validation only; there was no TEST access.
- No benchmark, generalization, or SOTA claim is supported.
- Correct-versus-shuffled separation is weak and Caption output is collapsed.
- Qwen remained frozen; adaptation covered one epoch and 600 training samples.
- SAR, temporal inputs, and grounding were not included.

## Completion and next recommendation

The artifact audit passed and the focused regression suite reported 52 passed, 0 failed, and 1 skipped. Phase status is `PHASE3O12_COMPLETE`.

Do not automatically start a new experiment. The recommended next phase is a targeted Phase 3O.13 diagnosis of weak image utilization: measure visual-token attention/utilization and modality sensitivity by layer; compare correct, shuffled, and zero-image logit margins; inspect language-prior dominance and task-specific projector behavior; diagnose caption collapse; and determine whether stronger image-conditioned objectives are necessary.
