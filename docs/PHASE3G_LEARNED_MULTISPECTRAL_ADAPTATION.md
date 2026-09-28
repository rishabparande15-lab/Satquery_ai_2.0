# Phase 3G — Learned Multispectral Adaptation for Qwen

Date: 2026-09-21  
Status: **PASS — bounded engineering proof of concept**

## 1. Objective

Phase 3G adds the first learned language-facing adapter for the canonical
12-band Sentinel-2 representation. The scientific Pipeline-3 predictor and
all Phase 2 representations remain outside this path and were not modified.

This phase answers the narrow question: can a small trainable S2-to-Qwen
visual-token projector receive real `[12,120,120]` S2 tensors and optimize a
supervised language-generation loss while Qwen remains frozen?

This is a **proof of concept, not a benchmark result**. No RSVQA/VRSBench
claim, scientific prediction claim, grounding claim, or comparison to the
65.0% scientific baseline is made.

## 2. Qwen architecture inspection

The pinned local checkpoint was inspected from its `config.json` and installed
Transformers implementation. The relevant dimensions are:

| Component | Verified value |
|---|---:|
| Qwen language hidden size | 2,048 |
| Qwen vision input channels | 3 |
| Qwen vision hidden size | 1,280 |
| Qwen vision output hidden size | 2,048 |
| Vision patch size | 14 |
| Spatial merge size | 2 |
| Aligned pilot grid | 112x112 |
| 8x8 patch grid after patchification | 64 patches |
| 2x2-merged visual tokens | 16 |

The installed Qwen forward path accepts `inputs_embeds` and replaces image
token slots with visual embeddings when the native vision path is used. Its
native patch embed is hard-coded for 3 input channels. It does not accept an
arbitrary 12-channel tensor as `pixel_values`.

## 3. Selected adaptation mechanism

Selected mechanism: **projector-only learned multispectral visual-token
adapter** (`phase3g_s2_qwen_token_projector_v1`).

```text
canonical S2 [B,12,120,120]
        ↓ fixed bilinear alignment to 112x112
learned 12-channel patch embedding, 14x14 stride
        ↓ 2x2 spatial merge
learned projection
        ↓ [B,16,2048]
Qwen image-token slots through inputs_embeds
        ↓
frozen Qwen causal language model
```

This is not RGB conversion. All twelve canonical bands are input channels and
the language target is the real annotation answer. The native Qwen vision
tower and all Qwen base weights remain frozen; the adapter targets the exact
2,048-dimensional post-vision representation consumed by Qwen's language
path. This avoids changing the hard-coded 3-channel vision interface and is
the smallest credible mechanism for the available hardware.

LoRA/QLoRA was not introduced. `bitsandbytes` and `peft` are not installed in
the environment, so no quantization or PEFT claim was made. The pilot used
the projector-only path with Qwen in FP16 and no quantization.

## 4. Trainable and frozen parameters

Trainable parameters are only the new projector:

- 12-channel, 14x14-stride patch convolution;
- LayerNorm over the merged patch channels;
- two linear layers with GELU producing 2,048-dimensional tokens.

Measured trainable projector parameters: **5,549,184**.

Qwen total parameters: **3,754,622,976**. Qwen trainable parameters:
**0**. The original Qwen checkpoint was not overwritten or copied into the
adapter checkpoint.

## 5. Real dataset linkage and split policy

The pilot used the existing validated
`artifacts/annotations/image_language/visual_vqa_candidates.jsonl` pool. A
record was eligible only when it was a validated `VISUAL_ONLY` visual-VQA
record, its split was train or validation, and all twelve raw S2 TIFFs were
available under the documented external Pipeline-3 roots.

The scan found 54,873 eligible real train/validation S2+annotation records:

- train: 34,458 binary QA and 18,046 multiple-choice QA;
- validation: 1,552 binary QA and 817 multiple-choice QA.

The deterministic pilot selected the first eligible records in source order:

- train: **100 records**;
- validation: **50 records**;
- test: **0 records**.

No image was moved between splits, no test record was read for optimization or
model selection, and no synthetic labels were used. The complete selected
record, annotation, image, split, and per-area S2 source hashes are stored in
[pilot_manifest.json](../artifacts/phase3g/pilot_manifest.json).

Dataset and provenance fingerprints:

- dataset fingerprint:
  `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`
- split fingerprint:
  `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`
- annotation fingerprint:
  `5562e7976aee140850fede7efe5d1c375934244fe5c0c4a754b76eeb1af5447b`
- pilot representation fingerprint:
  `885782e2d1b58c61a064cf4043cdcaa771106d61b4675bfcfc472b67a69604b5`

## 6. Training configuration

| Setting | Value |
|---|---|
| Objective | standard Qwen causal language-model loss |
| Input | real canonical S2, all 12 bands, normalized by existing robust channel scaling |
| Target | real BigEarthNet.txt visual-VQA answer text |
| Visual/prompt labels | masked with `-100`; only answer tokens supervised |
| Seed | 17 |
| Epochs | 1 |
| Train / validation | 100 / 50 records |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Precision | FP16 Qwen; projector kept trainable on CUDA |
| Quantization | none |
| Maximum answer tokens | 32 |

The exact selected records and configuration are in the pilot manifest and
checkpoint receipt.

## 7. Hardware and resource usage

- GPU: RTX 5060 Laptop, 8,151 MiB reported VRAM;
- Qwen model revision:
  `66285546d2b821cf421d4f5eb2576359d3770cd3`;
- model load: 12.5491 s;
- pilot elapsed time: 383.8696 s;
- peak CUDA allocated: 8,287,460,352 bytes;
- peak CUDA reserved: 8,902,410,240 bytes;
- peak process RSS: 3,553,509,376 bytes.

A single real-sample forward/backward gate reached projector gradients before
the pilot. The bounded pilot completed without OOM. No impossible
configuration was retried.

## 8. Training and validation results

The pilot completed with:

- mean training loss: **4.9798922938**;
- mean validation loss: **2.0006299591**;
- train records: **100**;
- validation records: **50**;
- test records used: **0**.

These losses are engineering receipts for one tiny deterministic pilot. They
do not establish generalization, useful multispectral understanding, or
publication-quality performance.

## 9. Qualitative examples

Generation was run from the saved projector on three validation records after
the pilot. These are qualitative only.

| Image / record | Target | Generated text |
|---|---|---|
| `S2B_MSIL2A_20180506T105029_N9999_R051_T31UER_47_16` / `ilr:00095a30e680217a4d68945f3747fbe7c2fc37162cd3347421e9875d701d6351` | `yes` | `Yes` |
| `S2B_MSIL2A_20180521T100029_N9999_R122_T34WFS_27_70` / `ilr:001b28ab7a092b3073f7638f48dbb1155770a6ae3f5d0ca42a1028be79db39c2` | `yes` | `Yes` |
| `S2A_MSIL2A_20180510T094031_N9999_R036_T35VLC_27_70` / `ilr:001c99101eed8f93c56bac82daf202ee70cd0ee998a183bae6de105c677a92a8` | `d` | empty output |

The first two outputs happen to match their binary targets, but this tiny
sample is not an accuracy evaluation and includes no benchmark claim.

## 10. Checkpoint and provenance

Only the newly trained projector state was saved:

- [s2_multispectral_projector.pt](../artifacts/phase3g/s2_multispectral_projector.pt)
- [s2_multispectral_projector.json](../artifacts/phase3g/s2_multispectral_projector.json)
- [training_summary.json](../artifacts/phase3g/training_summary.json)
- [pilot_manifest.json](../artifacts/phase3g/pilot_manifest.json)

Adapter checkpoint SHA-256:
`46cac19a98b5c4d737eb399d5cae57b616b8e606a6ff2eb8a24cd430ffa0e253`

The sidecar records model identity/revision, adapter identity, canonical band
order, Qwen dimensions, dataset/annotation/split/representation fingerprints,
sample IDs, seed, hyperparameters, device, dtype, quantization, trainable
parameter count, resource summary, and the checkpoint hash.

## 11. Limitations

- This is a one-epoch 100/50 proof of concept, not a benchmark.
- The projector bypasses Qwen's native 3-channel vision tower and targets its
  verified post-vision 2,048-dimensional token interface; native multispectral
  vision encoding remains unimplemented.
- No held-out test evaluation was performed.
- No LoRA/QLoRA comparison was performed.
- The pilot uses answer-token language loss; it does not validate calibrated
  VQA accuracy, spatial grounding, or scientific prediction.
- The high memory watermark leaves little room for larger context, batch size,
  or model expansion on the RTX 5060.

## 12. Future scaling plan

Before scaling beyond this pilot, add a deterministic evaluation harness on a
non-test validation partition, compare against the Phase 3F deterministic RGB
path, and profile a smaller token/context configuration. Only after that
should a separately justified LoRA experiment be considered. The official
test split must remain untouched until the experiment design and evaluator are
frozen.

## 13. Phase 3H readiness

The S2 learned-adaptation gate is technically complete and is ready for the
next scoped modality experiment: **real SAR adaptation**. Phase 3H must retain
the same frozen scientific boundary, split isolation, provenance requirements,
and fail-closed resource policy. The Phase 3F deterministic RGB baseline
remains operational and was not modified.

## 14. Scientific boundary verification

Unchanged by this phase:

- CROMA and its checkpoint;
- `physical_62d`, `joint_croma_gap_768d`, and `hybrid_830d`;
- scientific predictor and checkpoint;
- 5,000-area split and scientific receipts;
- representation catalog;
- Phase 2F spatial semantics;
- raw project scientific artifacts.

The locked scientific baseline remains **65.0% accuracy**, **4.1034286734 pp
MAE**, and **9.5873312123 pp RMSE**. Those values are not Phase 3G VLM
metrics.
