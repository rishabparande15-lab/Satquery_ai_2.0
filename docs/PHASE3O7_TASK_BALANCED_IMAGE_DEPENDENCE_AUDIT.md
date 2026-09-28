# Phase 3O.7 Task-Balanced Image-Dependence Audit

Status: PHASE3O7_COMPLETE

This is an internal controlled validation audit, not benchmark evaluation. It
used the Phase 3O.5 projector read-only with the pinned frozen Qwen revision.
The deterministic secondary panel contains 90 authoritative validation records:
30 binary VQA, 30 visual-only MCQ, and 30 captions. The panel excludes all
Phase 3O.6 identities and all Phase 3O.5 training images.

## Availability and panel

- Authoritative eligible validation counts: 1,552 binary VQA, 817 visual-only
  MCQ, and 197 caption records.
- After the Phase 3O.6 identity exclusion and Phase 3O.5 train-image
  exclusion: 1,536 binary VQA, 813 visual-only MCQ, and 197 captions were
  available with complete local S2 assets.
- The preregistered panel selected 30 unique images per task. Train/validation
  image overlap is zero. The deterministic same-task cyclic shuffle has no
  fixed points and changes the S2 image for every panel record.

## Normal versus shuffled results

| Task | Normal | Shuffled image | Normal minus shuffled |
| --- | ---: | ---: | ---: |
| Binary VQA | 17/30 = 0.5666666666666667 | 16/30 = 0.5333333333333333 | 0.033333333333333326 |
| MCQ | 8/30 = 0.26666666666666666 | 8/30 = 0.26666666666666666 | 0.0 |

The binary majority-label baseline is 0.5 (15 yes and 15 no references).
All MCQ questions have four authoritative options; the simple random-choice
expectation is 0.25.

Paired binary outcomes: normal-correct/shuffled-correct = 16;
normal-correct/shuffled-incorrect-or-unparsable = 1;
normal-incorrect-or-unparsable/shuffled-correct = 0;
normal-incorrect-or-unparsable/shuffled-incorrect-or-unparsable = 13.

Paired MCQ outcomes: normal-correct/shuffled-correct = 8;
normal-correct/shuffled-incorrect-or-unparsable = 0;
normal-incorrect-or-unparsable/shuffled-correct = 0;
normal-incorrect-or-unparsable/shuffled-incorrect-or-unparsable = 22.

## Caption audit

The fixed prompt was Question: Describe this Sentinel-2 image, followed by
Answer:. All 30 normal and all 30 shuffled caption generations were empty.
Consequently, all 30 normal/shuffled caption pairs were identical and none
changed under the image shuffle. The raw generated/reference-caption pairs are
preserved in the semantic-audit receipt; no arbitrary caption metric was added.

## Degeneracy

Normal: 34 empty outputs total (30 captions and 4 MCQ), 4 malformed/unparsable
MCQ outputs, no prompt echo, punctuation-only output, excessive repetition,
or abnormal termination.

Shuffled: the same counts—34 empty outputs total and 4 malformed/unparsable
MCQ outputs, with no other listed failure mode.

## Integrity

- Checkpoint SHA-256 verified unchanged:
  e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db.
- Projector state fingerprint verified unchanged:
  db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46.
- Pinned Qwen revision 66285546d2b821cf421d4f5eb2576359d3770cd3 was
  verified; Qwen and projector were unchanged.
- No optimizer was created, no gradient update occurred, test access was zero,
  and Phase 3O.2–3O.6 historical artifacts were unchanged.

## Conservative conclusion and recommendation

The audit did not establish strong dependence of generated answers on the
correct S2 imagery: the binary difference was one response out of 30, MCQ
accuracy was identical, and caption outputs were identically empty under both
conditions. Do not scale the dataset yet. The recommended next authorized
phase is targeted diagnosis of projector/Qwen token integration, attention
masks and visual-token positioning, target masking and answer formatting,
image-token conditioning, and the projector gradient path. No next phase was
started.

Machine-readable evidence:

- artifacts/training/phase3o/phase3o7_validation_panel.json
- artifacts/training/phase3o/phase3o7_image_shuffle_map.json
- artifacts/training/phase3o/phase3o7_task_balanced_semantic_audit.json
