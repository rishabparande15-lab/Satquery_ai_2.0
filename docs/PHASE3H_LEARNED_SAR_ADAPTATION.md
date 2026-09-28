# Phase 3H — Learned SAR Adaptation for Qwen

Status: **PARTIAL — implementation complete; real-data execution BLOCKED**

## 1. Objective

This phase adds a single-modal learned path from the canonical Sentinel-1
representation to the frozen Qwen language-facing token interface. Optical/SAR
fusion, temporal reasoning, grounding, and changes to the scientific pipeline
are explicitly out of scope.

## 2. Current Qwen interface

The existing Phase 3G inspection is reused: the pinned Qwen2.5-VL-3B
checkpoint consumes image-token slots backed by 2,048-wide post-vision
embeddings. The verified geometry is 14-pixel patches, an aligned 112×112
grid, an 8×8 patch grid, 2×2 spatial merge, and **16 tokens × 2,048**.
The adapter therefore injects learned embeddings through the existing
`inputs_embeds` pathway; it does not pretend SAR is RGB.

## 3. S1 input contract and preprocessing

`src/dataset_loader.py` is the authoritative loader. Source rasters are
single-band GeoTIFFs, read as float32, co-registered to the B02 10 m,
120×120 grid, with VV then VH order. Source nodata is converted to NaN and
the loader rejects any remaining non-finite values. The existing documented
language-facing-compatible normalization is per-channel mean ±2 standard
deviations, clipped to [0,1], with nodata represented as zero. No dB/log or
additional clipping transform is introduced here.

## 4. Learned architecture and integration

`src/eo_vlm/sar_projector.py` validates exact `[B,2,120,120]` input, resizes
to 112×112, applies a lightweight two-channel 14×14 stride convolution,
2×2 token merge, LayerNorm, and a two-layer projection to `[B,16,2048]`.
Only this projector is trainable; Qwen remains frozen. The default projector
has **5,298,304** trainable parameters in the default configuration.

## 5. Dataset linkage and split safety

The annotation artifact records exact dual Sentinel identities (`patch_id`
and `s1_name`) and the existing matching method is `dual_sentinel_exact_v1`.
No SAR VQA pair is accepted from an image ID alone. A real S1 root and an
authoritative manifest mapping those exact identities must be supplied before
training. In this checkout no usable S1 root was available to the Phase 3H
runner, so SAR VQA linkage is **SAR_VQA_LINKAGE_BLOCKED**. No synthetic pair,
question, answer, or split was created.

## 6. Training and results

Stages A (100/50), B (250/100), and C (500/100): **NOT TESTED** because the
linkage gate is blocked. Consequently loss, wall time, VRAM, RAM, qualitative
comparison, and adapter checkpoint hashes are not claimed. The implementation
is ready to reuse the Phase 3G masked answer-token causal-LM objective once
linkage is supplied.

## 7. Provenance and limitations

The projector exposes channel order, input/output shapes, preprocessing, and
scientific-separation provenance. Checkpoint writing and the existing Qwen
token-batch utilities remain reusable, but no checkpoint is emitted without
real linked training data. This phase therefore demonstrates adapter shape
compatibility, not SAR understanding, generalization, benchmark quality, or
scientific prediction.

## 8. Phase 3I readiness

**NOT READY.** The scientific baseline and Phase 3G S2 path were not changed.
Phase 3I must wait for authoritative S1/annotation linkage, staged training,
resource receipts, deterministic-vs-learned validation, and a provenance-complete
SAR adapter checkpoint.
