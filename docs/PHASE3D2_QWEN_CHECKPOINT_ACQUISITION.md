# PHASE 3D.2 — Qwen2.5-VL-3B Checkpoint Acquisition and Integrity Gate

**Status:** `PASS` for acquisition, integrity, structural validation, and
minimal CPU model construction.

## 1. Objective

Acquire exactly `Qwen/Qwen2.5-VL-3B-Instruct` through a resumable Hugging Face
path, verify the complete checkpoint and local structure, and perform only a
minimal load gate. No inference, benchmark, training, fine-tuning, LoRA,
QLoRA, or scientific-pipeline operation was performed.

## 2. Previous Download Failure

Phase 3D.1's normal snapshot download stalled after partial metadata and
created two resumable `.incomplete` shard files. The cache was not deleted.
The later acquisition resumed those files rather than restarting from zero.

## 3. Environment

| Component | Result |
| --- | --- |
| Python | 3.13.14 |
| PyTorch | 2.11.0+cu128 |
| CUDA available | `True` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU memory | 8,151 MiB |
| Driver | 616.92 |
| Transformers | 4.51.3 |
| Accelerate | 1.6.0 |
| safetensors | Available |
| bitsandbytes | Not installed; not required for this gate |

## 4. Disk Availability

The checkpoint snapshot consumed **7,520,919,614 bytes**. At the final
measurement, C: reported **28,363,698,176 bytes** available. The checkpoint
remained outside the repository in the Hugging Face cache.

## 5. Hugging Face Authentication Status

`hf auth whoami` reported **Not logged in**. The public target repository was
accessible without authentication. No token was requested, printed, stored in
the repository, or written to source files.

## 6. Acquisition Method

Acquisition used `huggingface_hub`/Hugging Face CLI against the official
repository only. The first resumed snapshot attempt stalled; the two explicit
shard downloads then used the existing cache and resumable byte offsets:

- `model-00002-of-00002.safetensors`: resumed from 3,162,385,742 of
  3,526,688,744 bytes.
- `model-00001-of-00002.safetensors`: resumed from 289,333,755 of
  3,982,649,232 bytes.

No unrelated model, dataset, or training corpus was downloaded.

## 7. Resume Behavior

The cache contained `.incomplete` files and a complete metadata snapshot at
revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. The per-shard CLI path
reused those files and moved each completed shard into the same snapshot. No
partial file was deleted.

## 8. Files Acquired

| File | Bytes | Status |
| --- | ---: | --- |
| `model-00001-of-00002.safetensors` | 3,982,649,232 | `PASS` |
| `model-00002-of-00002.safetensors` | 3,526,688,744 | `PASS` |
| `model.safetensors.index.json` | 65,448 | `PASS` |
| `config.json` | 1,373 | `PASS` |
| `preprocessor_config.json` | 350 | `PASS` |
| tokenizer/configuration files | present | `PASS` |

## 9. Checkpoint Size and Repository Revision

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Source | `https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct` |
| Revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Snapshot total | 7,520,919,614 bytes |
| Model parameter count loaded | 3,754,622,976 |
| License file | present in snapshot; official model-card terms apply |

## 10. SHA256 / Integrity

Computed local SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `model-00001-of-00002.safetensors` | `41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8` |
| `model-00002-of-00002.safetensors` | `365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4` |

The blob names in the Hugging Face cache matched these computed hashes. No
separate public SHA-256 manifest was available to compare against, so the
status is `PASS` for local complete-file hashing and `NOT VERIFIED` for an
independent publisher SHA manifest.

## 11. Cache Location

```text
<machine-local-huggingface-cache>/<qwen-model-cache-entry>
```

The model was not copied into the SatQuery repository and is not Git-tracked.
Windows symlink fallback was reported by Hugging Face; this may use additional
cache space but did not create a project copy.

## 12. Structural Validation

Structural validation **PASS**:

- `config.json` resolved as `Qwen2_5_VLConfig` with model type `qwen2_5_vl`.
- `model.safetensors.index.json` exists and both indexed shards are present.
- Local tokenizer resolved as `Qwen2TokenizerFast` with vocabulary length
  151,665.
- `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`,
  `merges.txt`, and `vocab.json` are present.
- Both shard sizes are complete and plausible for the index.

## 13. Minimal Load Result

`MODEL_LOAD_READY` for a bounded **CPU FP16 construction check**:

```text
Class: Qwen2_5_VLForConditionalGeneration
Load time: 37.749 seconds
RSS before: 638,857,216 bytes
RSS after: 8,235,941,888 bytes
Parameters: 3,754,622,976
Device: CPU
dtype: float16
quantization: none
```

This was a structural model construction only. No image processor execution,
CUDA load, generation, or benchmark was performed.

## 14. Remaining Blockers

- GPU FP16/BF16 feasibility is not yet tested.
- 8-bit and 4-bit modes are not tested; `bitsandbytes` is not installed.
- No image inference or adapter invocation was performed in this acquisition
  phase.
- Publisher-independent SHA manifest comparison is unavailable.

## 15. Phase 3D.3 Readiness

Phase 3D.3 is now **READY TO ATTEMPT** for the target checkpoint. It may test
bounded FP16/BF16 CUDA loading and one RGB smoke prompt through
`src/eo_vlm_adapter.py`. It must continue to avoid S2/SAR claims, training,
quantization claims without execution, and any use of `hybrid_830d` as VLM
input.

## Validation and Scientific Safety

- The scientific baseline remains 65.0% accuracy, 4.1034286734 pp MAE, and
  9.5873312123 pp RMSE.
- No scientific code, datasets, splits, CROMA files, scientific checkpoints,
  representation artifacts, receipts, or Phase 2F documentation were changed.
- No commit or push was performed.
