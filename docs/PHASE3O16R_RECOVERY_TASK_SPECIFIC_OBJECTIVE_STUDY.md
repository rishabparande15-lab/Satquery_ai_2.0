# Phase 3O.16R Recovery Task-Specific Objective Study

Historical Phase 3O.16 remains `PHASE3O16_BLOCKED`: its post-trial states, raw output evidence, and serialized gates were absent. This document reports only the independent recovery experiment, `PHASE3O16R_COMPLETE`.

## Protocol and integrity

The recovery used deterministic seed 30161, 24 authoritative training and 12 authoritative validation records per task, and excluded all 108 original Phase 3O.16 panel identities. Train/validation images do not overlap. The preregistration digest is `ca5602fcc8d8fa7132f7e77fd3c7262c89289ef5c1017bbffe3cf5745dd55d9a`.

Every variant reloaded the immutable Phase 3O.12 state, completed eight steps, and saved a state plus raw validation receipt before classification. Phase 3O.12 remained immutable; Qwen remained frozen and unchanged, with zero optimizer/gradient membership; TEST access was zero.

## Results

Binary: Base CE, Smooth Likelihood, and Low-weight Hinge increased the validation target margin from -0.00619 to -0.00285, -0.00257, and 0.00129. Yet all were 5/12 correct under both natural and shuffled imagery and had zero output changes. All are `NOT_PROMISING`.

MCQ diagnosis: `DISTRACTOR_DOMINATES_REGARDLESS_OF_IMAGE` / language-prior dominance. The baseline target-vs-best-distractor margin was -0.06122. Base CE, ranking, and joint variants produced -0.12896, -0.13547, and -0.13286. All are `NOT_PROMISING`; the ranking objective's 6/12 versus 5/12 natural/shuffled accuracy did not overcome its declining margin gates.

Caption collapse begins in the `EARLY_PREFIX`: pre-trial first-token and three-token-prefix diversity were both one. Base CE and sequence separation collapsed two unique outputs to one and increased the dominant mode from 8 to 12. The early-token objective preserved two outputs and increased prefix diversity to two, but increased the dominant mode from 8 to 11. All are `NOT_PROMISING`.

## Decision

No task has a complete auditable promising result. `NO_LARGER_TRAINING_JUSTIFIED`. Do not start Phase 3O.17. The exact next recommendation is to retain the Phase 3O.16R receipts as negative evidence and, only under separate authorization, preregister a new intervention after addressing the MCQ distractor prior and caption early-prefix collapse.
