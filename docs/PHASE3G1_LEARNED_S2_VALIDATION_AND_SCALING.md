# Phase 3G.1 — Learned S2 Adapter Validation and Controlled Scaling

Date: 2026-09-21  
Status: **PASS**

## 1. Objective

This phase reproduced the Phase 3G learned 12-band S2 projector and scaled the
same experiment to 250/100 and 500/100 real train/validation records. The
scientific pipeline, test split, CROMA path, representation receipts, and
Phase 2F semantics were not modified.

The result remains an engineering proof of concept. It is not a benchmark
claim, native multispectral-understanding claim, scientific accuracy result,
RSVQA result, or VRSBench result.

## 2. Phase 3G baseline

The fixed adapter is:

```text
real canonical S2 [12,120,120]
  -> learned patch projector
  -> 16 Qwen-compatible visual tokens x 2048
  -> frozen Qwen language model
```

Trainable parameters: **5,549,184**. Qwen parameters: **3,754,622,976**;
Qwen trainable parameters: **0**.

The baseline used seed 17, AdamW, learning rate 0.001, weight decay 0.0001,
batch size 1, gradient accumulation 8, one epoch, FP16 Qwen, no quantization,
and the same masked answer-token causal-LM loss.

## 3. Reproduction

The original 100/50 selection was reloaded from the deterministic source-order
policy and rerun without altering the original Phase 3G checkpoint.

| Receipt | Original Phase 3G | Reproduction | Result |
|---|---:|---:|---|
| Train loss mean | 4.9798922938 | 4.9798922938 | PASS |
| Validation loss mean | 2.0006299591 | 2.0006299591 | PASS |
| Checkpoint SHA-256 | `46cac19a98b5c4d737eb399d5cae57b616b8e606a6ff2eb8a24cd430ffa0e253` | same | PASS |
| Train / validation image overlap | 0 | 0 | PASS |
| Unique train / validation images | 98 / 47 | 98 / 47 | PASS |

The reproduction checkpoint is byte-identical to the original adapter
checkpoint. Its receipts are under
`artifacts/phase3g1/reproduction_100_50/`.

## 4. Dataset scaling

All stages used the existing validated visual-VQA candidate pool, real S2
imagery, exact image/annotation linkage, and source-order selection. No test
record was read for training or model selection. Each stage had zero
train/validation image intersection.

| Stage | Train | Validation | Unique train images | Unique validation images | Train loss | Validation loss | Train time s | Validation time s | Peak CUDA allocated | Peak RAM | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A reproduction | 100 | 50 | 98 | 47 | 4.979892 | 2.000630 | 320.11 | 11.96 | 8,287,460,352 | 3,552,387,072 | `46cac19a…ffa0e253` |
| B | 250 | 100 | 243 | 84 | 3.424620 | 2.803827 | 984.51 | 69.18 | 8,364,978,688 | 4,102,725,632 | `3696a7dd…c3c57ee1` |
| C | 500 | 100 | 481 | 84 | 2.561415 | 1.031613 | 1,953.40 | 39.68 | 8,364,978,688 | 4,084,510,720 | `f9645249…faf2f5c` |

Full hashes and per-record manifests are stored in:

- `artifacts/phase3g1/reproduction_100_50/`
- `artifacts/phase3g1/stage_b_250_100/`
- `artifacts/phase3g1/stage_c_500_100/`

The full Stage B checkpoint hash is
`3696a7dd9d7c9e644e919a6a3a271725f039b70ee18e314634cb5997c3c57ee1`.
The full Stage C checkpoint hash is
`f9645249bdb381116aaccf083f6850af34357fe4576f6f4f6997a87c2faf2f5c`.

The run time increased approximately linearly with record count. Peak model
allocation remained in the same narrow envelope after optimizer state was
established; no concurrent Qwen copy was created and no OOM occurred.

## 5. Training configuration

All three stages used the same configuration:

- optimizer: AdamW;
- learning rate: `0.001`;
- weight decay: `0.0001`;
- batch size: `1`;
- gradient accumulation: `8`;
- epochs: `1`;
- seed: `17`;
- device: CUDA;
- Qwen precision: FP16;
- quantization: none;
- Qwen frozen: yes;
- trainable component: S2 projector only;
- test samples: `0`.

Measured answer-token throughput was approximately 0.625 train tokens/s in
Stage A, 0.508 train tokens/s in Stage B, and 0.512 train tokens/s in Stage C.
Validation throughput was approximately 8.36, 2.89, and 5.04 tokens/s,
respectively; these measurements are diagnostic only because validation uses
no backward pass and has different prompt/answer lengths.

## 6. Loss comparability analysis

The loss implementation is structurally consistent across train and
validation:

- teacher forcing is used by Qwen's standard causal-LM loss;
- visual-token slots and the question/prompt are masked with `-100`;
- only real answer tokens contribute to cross entropy;
- batch size is 1 and each sample contributes one scalar loss to the reported
  arithmetic mean;
- the model is in evaluation mode for both loops, so dropout is not a source
  of difference;
- the same canonical S2 normalization and projector are used.

The means are therefore comparable as implementation receipts, but not as a
strict fixed-model generalization estimate. Training loss is measured online
before each optimizer update, across parameters that change during the epoch;
validation loss is measured afterward using the final adapter. Answer lengths
and task mix also differ between the source-order subsets. The lower
validation loss must not be interpreted as unusually strong generalization.
No loss rewrite was made.

## 7. Resource measurements

The Stage B and C peak CUDA measurements were identical:

- peak allocated: `8,364,978,688` bytes;
- peak reserved: `9,342,812,160` bytes;
- peak RAM: `4,102,725,632` bytes in Stage B and `4,084,510,720` bytes in
  Stage C.

The adapter remains batch-1 only on this RTX 5060 configuration. The run
completed without OOM, but the high watermark leaves little room for longer
contexts, larger batches, or additional trainable modules.

## 8. Deterministic baseline versus learned adapter

The following ablation used the same five validation images and the same
question prompts for both paths:

- deterministic baseline: Phase 3F true-color B04/B03/B02 representation into
  Qwen;
- learned path: real 12-band S2 into the saved Stage C projector.

No subjective score was assigned.

| Sample | Question | Target | Deterministic RGB output | Learned-adapter output | Observed difference |
|---|---|---|---|---|---|
| `ilr:00095a30e680217a4d68945f3747fbe7c2fc37162cd3347421e9875d701d6351` | Is the area of arable lands less than 60%? | `yes` | `To determine if the area of arable` | empty | different response length/content |
| `ilr:001b28ab7a092b3073f7638f48dbb1155770a6ae3f5d0ca42a1028be79db39c2` | Does the satellite view capture mixed forest? | `yes` | `Yes, the satellite view captures mixed forest` | `Yes` | learned response is shorter |
| `ilr:001c99101eed8f93c56bac82daf202ee70cd0ee998a183bae6de105c677a92a8` | Pick the area covered by complex cultivation patterns: a) 60 to 90%, b) 40 to 50%, c) 50 to 60%, d) 0 to 40% | `d` | `c) 50 to 6` | empty | different response content |
| `ilr:004d6364e1f1a59898011e3c3b36d659a5183f42573634c4be49770ead849346` | How much of the image do urban fabrics cover? a) 20 to 30%, b) 30 to 70%, c) 10 to 20%, d) 0 to 10% | `b` | `b) 30 to 7` | `b` | different response length |
| `ilr:005fda2a38e944a083913ba7f15b82f2f75ef1c44529c9b37510a00578844653` | Are at least four connected regions of transitional woodlands or shrubs visible in this image? | `no` | `Yes, there are at least four connected` | empty | different response content |

This is an architectural ablation only. It is too small to support a quality
or superiority claim.

## 9. Checkpoints and provenance

Each stage saved only the adapter-specific projector state, a JSON sidecar,
the stage training summary, and the deterministic sample manifest. No Qwen
full-model copy was saved and the original Qwen checkpoint was not changed.

All receipts record:

- Qwen model ID and revision;
- adapter ID and exact output shape `[1,16,2048]`;
- canonical band order;
- dataset, split, annotation, and stage representation fingerprints;
- selected record and annotation IDs;
- source S2 area hashes;
- seed and hyperparameters;
- device, dtype, quantization, trainable count;
- resource measurements and checkpoint SHA-256.

Common fingerprints remain:

- dataset: `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`;
- split: `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`;
- annotations: `5562e7976aee140850fede7efe5d1c375934244fe5c0c4a754b76eeb1af5447b`.

## 10. Limitations

- The experiments remain tiny proof-of-concept language-loss studies.
- No held-out test evaluation was performed.
- The native Qwen 3-channel vision tower is not a native 12-band encoder;
  the learned projector targets Qwen's verified post-vision token interface.
- Loss is not VQA accuracy and should not be interpreted as benchmark quality.
- The first source-order records are not geography-balanced or task-balanced.
- No LoRA/QLoRA comparison was performed; `bitsandbytes` and `peft` remain
  unavailable.
- The allocator watermark is close to the GPU limit, so larger-scale training
  is not justified under the current configuration.

## 11. Phase 3H readiness

The S2 learned-adaptation validation and controlled scaling gate is **PASS**:

- baseline reproduction: PASS;
- 250/100 scaling: PASS;
- 500/100 scaling: PASS;
- loss behavior understood: PASS;
- resource usage recorded: PASS;
- learned-versus-deterministic ablation: PASS;
- test-set leakage: PASS — zero test records used;
- adapter checkpoints and provenance: PASS;
- scientific pipeline unchanged: PASS;
- validation suite: PASS.

Phase 3H may begin with **real SAR adaptation**, preserving the same frozen
scientific boundary, split isolation, provenance requirements, and resource
fail-closed policy. This report does not promote the learned S2 adapter to a
benchmark or scientific baseline.

## 12. Scientific baseline verification

The locked scientific values remain **65.0% accuracy**, **4.1034286734 pp
MAE**, and **9.5873312123 pp RMSE**. They are unchanged and are not VLM
metrics. CROMA, scientific checkpoints, scientific representations, receipts,
split fingerprints, and Phase 2F spatial semantics were not modified.
