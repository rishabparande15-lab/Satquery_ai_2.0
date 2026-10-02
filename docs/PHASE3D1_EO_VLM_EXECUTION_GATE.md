# PHASE 3D.1 — EO-VLM Environment Bring-Up and Local Execution Gate

**Status:** `PHASE 3E BLOCKED`

## 1. Objective

Resolve the Phase 3D loading blocker for `Qwen/Qwen2.5-VL-3B-Instruct` and
determine whether it can execute on the target RTX 5060 Laptop GPU with 8 GB
VRAM and 24 GB RAM. This phase performed environment bring-up only. No
training, fine-tuning, LoRA, QLoRA, scientific inference, or artifact
regeneration was performed.

## 2. Environment Before Changes

| Component | Observed status |
| --- | --- |
| Python | 3.13.14 |
| PyTorch | 2.11.0+cu128 |
| `torch.cuda.is_available()` | `True` |
| PyTorch CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Total GPU memory | 8,151 MiB reported by `nvidia-smi` |
| NVIDIA driver | 616.92 |
| Transformers | Not installed before bring-up |
| Accelerate | Not installed before bring-up |
| bitsandbytes | Not installed |
| safetensors | Available |
| Hugging Face Hub | Available |

## 3. Dependency Changes

Installed only the minimal model-loading packages requested for this gate:

- `transformers==4.51.3`
- `accelerate==1.6.0`

`bitsandbytes` was not installed. No repository requirements file, scientific
dependency, CUDA installation, or system-level setting was changed. Pip
reported pre-existing unrelated `paddlex` dependency conflicts; those were not
modified.

## 4. CUDA/GPU Validation

CUDA validation **PASS**:

```text
torch.cuda.is_available() = True
GPU = NVIDIA GeForce RTX 5060 Laptop GPU
VRAM = 8151 MiB
PyTorch CUDA = 12.8
Driver = 616.92
```

No model was loaded, so allocated/reserved pre-load and peak model-load VRAM
were not measured. No permanent CUDA changes were made.

## 5. Model Checkpoint

Target: `Qwen/Qwen2.5-VL-3B-Instruct` from the official Hugging Face model
repository: [model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

The download was started once after installing the minimal runtime. The client
created the local Hugging Face cache and fetched small metadata files, but the
large weight files stalled. The process was stopped after the cache remained at
approximately 11.6 MB and no weight shard completed.

| Field | Result |
| --- | --- |
| Download status | `BLOCKED` / incomplete |
| Local cache | `<machine-local-huggingface-cache>/<qwen-model-cache-entry>` |
| Files present | 15 small metadata/repository files observed |
| Bytes observed | approximately 11,581,678 bytes |
| Complete checkpoint | No |
| Exact revision | `NOT VERIFIED` locally |
| Complete checkpoint size/hash | `NOT AVAILABLE`; no weight shard completed |
| License | Referenced by official model card; local license verification not completed |

No datasets, benchmark corpora, or unrelated models were downloaded.

## 6. FP16 Results

`NOT TESTED`. The checkpoint was incomplete, so no FP16 CUDA load was
attempted.

## 7. BF16 Results

`NOT TESTED`. The checkpoint was incomplete, so no BF16 CUDA load was
attempted.

## 8. 8-bit Results

`NOT TESTED`. `bitsandbytes` was not installed and the checkpoint was
incomplete. No unsupported Windows/GPU configuration was forced.

## 9. 4-bit Results

`NOT TESTED`. `bitsandbytes` was not installed and the checkpoint was
incomplete. No 4-bit support is claimed.

## 10. CPU/Offload Results

`NOT TESTED`. CPU/offload was not attempted because the checkpoint download
never completed. No RAM-exhausting fallback was started.

## 11. VRAM Measurements

| Measurement | Result |
| --- | --- |
| GPU total | 8,151 MiB |
| Before model load allocation | `NOT MEASURED` |
| Before model load reservation | `NOT MEASURED` |
| After model load | `NOT APPLICABLE` — no model loaded |
| Peak during generation | `NOT APPLICABLE` — no generation |

## 12. RAM Measurements

System RAM before/after/peak during model loading was not measured. No model
load was reached and no configuration was started that could intentionally
exhaust system RAM.

## 13. Inference Smoke Test

`BLOCKED`. The fixed prompt was not sent because no complete model checkpoint
was available:

> Describe the observable scene characteristics in this remote-sensing image.
> Do not invent objects or measurements that are not visually supported.

Consequently there is no raw output, normalized output, token count, latency,
or device/dtype generation record. This is not a benchmark result.

## 14. Adapter Integration

The existing isolated adapter remains the only model-facing path:
`src/eo_vlm_adapter.py`.

Adapter contract validation **PASS** through unit tests. It preserves model and
adapter identity, modality, input hash, preprocessing identity, and the
explicit flag that no scientific representation was used. Model invocation
could not be exercised because the runtime/checkpoint was unavailable.

## 15. Modality Support

| Input | Status in this gate |
| --- | --- |
| RGB optical image | Adapter contract `PASS`; model inference `NOT TESTED` |
| S2 12-band tensor `[N,12,120,120]` | `UNSUPPORTED` by the RGB adapter; requires a documented band adapter |
| S1 `[N,2,120,120]` `VV,VH` | `UNSUPPORTED`; requires a SAR adapter |
| Optical + SAR | `UNSUPPORTED`; no native joint fusion established |
| Temporal T1/T2 | `NOT VERIFIED`; no temporal model path implemented |
| CROMA GAP/tokens | `UNSUPPORTED`; never passed to Qwen |
| Physical 62-D / hybrid 830-D | `UNSUPPORTED`; remain scientific-predictor inputs |
| Grounding | `NOT TESTED`; Phase 2F remains fail-closed |

## 16. Failures / Blockers

The exact execution blocker is incomplete checkpoint retrieval. The official
client fetched repository metadata but stalled while fetching weight shards.
Without complete weights, model loading, quantization comparison, inference,
and memory measurement cannot be honestly performed.

Secondary blockers are absent `bitsandbytes` and absent local model-processor
execution evidence. These were not forced because the primary checkpoint
blocker already prevented a meaningful test.

## 17. Reproducibility

Commands and observations were recorded from the repository worktree using:

- Python/PyTorch package inspection;
- `nvidia-smi --query-gpu=name,memory.total,driver_version`;
- pinned installation of Transformers and Accelerate;
- one official Hugging Face metadata/download attempt.

No model revision or complete file hash can be recorded because no weight
shard completed. The local cache path and observed partial size are recorded
above so a future retry can be distinguished from a fresh download.

## 18. Phase 3E Decision Gate

**PHASE 3E BLOCKED.** The required conditions were not met:

- checkpoint loads: **NO**;
- CUDA configuration works for inference: **NOT TESTED**;
- inference succeeds: **NO**;
- resource usage measured: **NO**;
- adapter invocation with model: **NOT TESTED**.

The GPU and CUDA stack are available, and the minimal Python runtime is now
installed. A future controlled retry may resume only the same target model
download, verify the exact revision and hashes, then test FP16/BF16 before any
optional quantized mode. No PEFT/LoRA/QLoRA work should begin until that gate
passes.

## Validation and Scientific Safety

- Focused adapter and relevant tests: **29 passed**; no model test was added
  to the ordinary suite.
- No scientific predictor, CROMA checkpoint, dataset, split, manifest,
  receipt, representation, or Phase 2F document was modified.
- Scientific baseline remains: 65.0% accuracy, 4.1034286734 pp MAE,
  9.5873312123 pp RMSE.
- No commit or push was performed.
