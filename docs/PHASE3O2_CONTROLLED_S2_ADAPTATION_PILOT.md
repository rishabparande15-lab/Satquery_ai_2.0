# Phase 3O.2 — Controlled S2 Adaptation Pilot

Status: **ADAPTATION_PILOT_COMPLETE**

This is an S2 adaptation pilot and is NOT an RSVQA benchmark evaluation. No
benchmark predictions or benchmark scores were generated.

## Integrity and data partition

The frozen five-area train manifest was rehashed successfully:
`972c0f8afbeea3d9fbfc641557dc4296558a1fdf374123d7388ebb25831051aa`.
Every combined raw-S2 checksum matched its inventory record. The five train
records were exact BigEarthNet.txt `VISUAL_ONLY` binary/MCQ records; the one
validation record was independently verified as validation and had a distinct
image ID. No test record was accessed.

## Architecture and configuration

Raw 12-band S2 was bilinearly aligned to B02 and normalized by
`robust_channel_scale_v1`, then passed to the existing S2 projector producing
`[1,16,2048]` finite Qwen-compatible tokens. Qwen revision
`66285546d2b821cf421d4f5eb2576359d3770cd3` was frozen (0 trainable base
parameters); only 5,549,184 projector parameters were optimized.

The run used seed 17, FP16 Qwen, AdamW (`lr=0.001`), batch 1, gradient
accumulation 1, and five optimization steps. No LoRA, QLoRA, RGB adaptation,
SAR, fusion, temporal, grounding, or scientific representation was used.

## Result and reproducibility

Initial train loss was 7.3518242836; final step loss was 2.1016514301.
Validation loss changed from 7.5001544952 before training to 6.8627839088
after training. All losses were finite. Eight projector tensors changed, with
total absolute parameter delta 14922.4267578125 and maximum delta
0.0050189374. The sampled frozen-Qwen state signature remained unchanged.

The dedicated adapter-only checkpoint is
`artifacts/training/phase3o/phase3o2_s2_projector.pt` (22,201,105 bytes,
SHA-256 `f610f4167a202ae0631976d4d6d987822b6334608aaf0b501a17c5cd642945c4`).
A fresh Qwen/projector instance reloaded it and reproduced validation loss
exactly (difference <= 1e-5).

Peak CUDA allocation was 8,151,910,912 bytes; peak RSS was 3,128,954,880
bytes; optimization took 9.3803956000 seconds. These are bounded engineering
receipts, not task-quality measures.

Scientific baseline, Pipeline-3 assets, historical Phase 3G/3G.1 artifacts,
SAR/temporal/grounding states, and all benchmark states remain unchanged.
Phase 3O.3 may proceed only with a separately approved evaluation protocol or
a larger split-preserving adaptation plan; it must not reinterpret this pilot
as benchmark evidence.
