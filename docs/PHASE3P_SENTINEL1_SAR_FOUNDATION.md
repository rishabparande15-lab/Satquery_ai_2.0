# Phase 3P Sentinel-1 / SAR Foundation

Status: `PHASE3P_SAR_FOUNDATION_COMPLETE`. This is a standalone, inference-only
foundation check; it performs no SAR training, Qwen adaptation, or S1+S2 fusion.

The selected existing-architecture path is canonical S1 VV/VH `[2,120,120]`,
existing official CROMA `s1_encoder` tokens `[1,225,768]`, new isolated
`S1SARProjector` tokens `[1,16,2048]`, then the existing Qwen image-token
insertion boundary. This preserves the validated Qwen interface and avoids
inventing a second SAR encoder or fusion path.

Five authoritative non-TEST samples passed loading, canonical VV/VH order,
per-channel mean +/- 2 standard-deviation clipping to `[0,1]`, finite-value,
CRS, and 10 m / 120x120 checks. CROMA representations were finite and varied
between samples (adjacent-pair L2 372.41--390.53; cosine 0.531--0.574).

The SAR adapter output was finite `[1,16,2048]`; frozen Qwen forward passed
with 16 visual-token positions. Correct-versus-shuffled final hidden/logit L2
was 50.78 / 356.62, and correct-versus-zero was 153.75 / 718.61. These prove
numerical pathway sensitivity only, not SAR semantic reasoning. Forward and
generation first-step logits agreed exactly (maximum absolute delta 0.0).

The existing controller correctly remains blocked for SAR VQA and optical-SAR
reasoning; Phase 3P does not enable either route. The next phase, only with
separate authorization, may run a bounded standalone SAR adaptation pilot.
