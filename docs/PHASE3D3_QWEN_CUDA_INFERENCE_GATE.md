# PHASE 3D.3 — Qwen2.5-VL-3B CUDA Inference and Resource Gate

**Status:** `PASS` for bounded FP16 CUDA RGB execution; Phase 3E is ready only
for the RGB adapter path.

## 1. Objective

Run the acquired `Qwen/Qwen2.5-VL-3B-Instruct` checkpoint on the RTX 5060,
validate the actual processor and image path, and execute a maximum of three
RGB smoke tests through `src/eo_vlm_adapter.py`. No training, fine-tuning,
quantization training, scientific inference, or benchmark evaluation was
performed.

## 2. Verified Environment

| Field | Value |
| --- | --- |
| Python | 3.13.14 |
| PyTorch | 2.11.0+cu128 |
| CUDA available | `True` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| VRAM | 8,151 MiB |
| Driver | 616.92 |
| Transformers | 4.51.3 |
| Accelerate | 1.6.0 |
| Pillow | 11.3.0 |
| bitsandbytes | Not installed |

The GPU reported BF16 support through `torch.cuda.is_bf16_supported() == True`.

## 3. Checkpoint Identity

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Cache: `<machine-local-huggingface-cache>/<qwen-model-cache-entry>`
- Checkpoint integrity and shard hashes: [Phase 3D.2 report](PHASE3D2_QWEN_CHECKPOINT_ACQUISITION.md)

## 4. GPU Validation

CUDA validation **PASS**. Before FP16 loading, free memory was 7,385,120,768
bytes of 8,518,041,600 bytes; allocated and reserved memory were both zero.

## 5. FP16 Test

`PASS` for model load and one RGB generation.

| Measurement | Result |
| --- | ---: |
| Device | CUDA (`cuda:0`) |
| dtype | `float16` |
| Quantization | none |
| Load time | 21.849 s |
| Allocated after load | 7,520,955,904 bytes |
| Reserved after load | 7,650,410,496 bytes |
| RAM before load | 632,819,712 bytes RSS |
| RAM after load | 1,133,359,104 bytes RSS |

The model loaded with `Qwen2_5_VLForConditionalGeneration` and 3,754,622,976
parameters. The configuration is operational for a tiny RGB generation but
leaves limited VRAM headroom.

## 6. BF16 Test

`NOT TESTED`. The runtime reported BF16 support, but FP16 loaded and generated
successfully; no second full model load was started unnecessarily.

## 7. 8-bit Test

`UNSUPPORTED / NOT TESTED`. `bitsandbytes` is not installed or validated in
this Windows environment. No 8-bit support is claimed.

## 8. 4-bit Test

`UNSUPPORTED / NOT TESTED`. `bitsandbytes` is not installed or validated. No
4-bit support is claimed.

## 9. Processor Validation

`PASS`. The adapter's lazy loader successfully resolved the local Qwen
processor and loaded the model through Transformers. The RGB path converted a
controlled `120×120×3` NumPy image to a PIL RGB image, applied Qwen's chat
template, tokenized text, packed image/text tensors, moved inputs to CUDA, and
called generation.

## 10. RGB Smoke Test

One controlled in-memory RGB smoke test was run because no repository image
files were available in the inspected `experiments`/`data` paths. This was not
a scientific dataset sample and was not used as a benchmark.

Prompt:

> Describe the observable scene characteristics in this remote-sensing image.
> Do not invent objects, counts, measurements, or identities that are not
> visually supported.

Result:

- Status: `PASS`
- Modality: `OPTICAL_RGB`
- Image: controlled 120×120 RGB array through the adapter
- Generation time: 18.333 s
- Generated tokens: 16
- Output: `The image appears to be a satellite or aerial photograph of an area with a mix`
- Scientific representation used: `False`

The output is a smoke-test generation only. It does not establish VQA,
captioning quality, hallucination safety, EO-native understanding, or any
benchmark score.

## 11. Adapter Integration

`PASS` at the tested RGB boundary. The execution path was:

```text
EOInput → Qwen25VLRGBAdapter.validate_inputs
        → prepare_inputs
        → load_model
        → generate
        → provenance-bearing result
```

The result preserved model identity, adapter identity, checkpoint revision,
modality, input SHA-256, preprocessing metadata, dtype, device, quantization
mode, and generation timing. The adapter did not pass CROMA, physical, or
hybrid scientific representations to Qwen.

## 12. VRAM Measurements

| Point | Measurement |
| --- | ---: |
| Before load allocated | 0 bytes |
| Before load reserved | 0 bytes |
| Free before load | 7,385,120,768 bytes |
| Allocated after FP16 load | 7,520,955,904 bytes |
| Reserved after FP16 load | 7,650,410,496 bytes |
| Peak generation allocation | Not reset/captured in this run |
| Cleanup allocation | 9,568,256 bytes |

The successful generation does not mean the configuration has comfortable
headroom. Peak generation statistics were not reset and captured, so they are
reported as not measured rather than inferred.

## 13. RAM Measurements

| Point | RSS |
| --- | ---: |
| Before FP16 load | 632,819,712 bytes |
| After FP16 load | 1,133,359,104 bytes |
| Peak generation RSS | Not separately measured |

## 14. Latency Measurements

- FP16 model load: 21.849 s.
- RGB generation: 18.333 s for 16 generated tokens.
- Preprocessing latency: not separately instrumented; included in the
  generation invocation setup and reported as not measured.

## 15. Failure Analysis

No FP16 CUDA failure occurred. The remaining configuration limitations are:

- BF16: supported by the device but not tested after FP16 success.
- 8-bit/4-bit: blocked because bitsandbytes is not installed or validated.
- VRAM headroom is narrow; larger images, longer prompts, or longer generation
  may require additional measurement or quantization.

## 16. Modality Support

| Modality/input | Status | Boundary |
| --- | --- | --- |
| RGB optical image | `TESTED` | FP16 CUDA generation passed |
| S2 `[N,12,120,120]` | `UNSUPPORTED / ADAPTER REQUIRED` | No native 12-band path tested |
| S1 `[N,2,120,120]`, `VV,VH` | `UNSUPPORTED / ADAPTER REQUIRED` | No SAR path tested |
| Optical + SAR | `UNSUPPORTED / ADAPTER REQUIRED` | No joint fusion path established |
| Temporal pair | `NOT TESTED / ADAPTER REQUIRED` | No temporal model path established |
| CROMA GAP/tokens | `UNSUPPORTED` | Not passed to Qwen |
| Grounding | `NOT VERIFIED` | No grounding output tested; Phase 2F unchanged |

Pseudo-RGB or separate modality prompting must not be described as native
multispectral, SAR, optical-SAR, or temporal reasoning.

## 17. Scientific Separation

The scientific predictor remains separate and unchanged:

```text
physical_62d + joint_croma_gap_768d → hybrid_830d → 19-class predictor
```

Qwen received only the controlled RGB adapter input. No `hybrid_830d`, CROMA
GAP, CROMA tokens, scientific checkpoint, or scientific prediction was used.

## 18. Reproducibility

The run used the local snapshot and the fixed checkpoint revision above. The
adapter result recorded input SHA-256:

`2d78287bdf3f0791c1e88a3b4ec15fd7d495b0c74eaf4a9cc639c7dd884e805d`

The test was one in-memory controlled image, not a benchmark or a dataset
mutation. Model objects were released and CUDA cache cleared after the run.

## 19. Phase 3E Decision Gate

**PHASE 3E: READY FOR RGB-ONLY ADAPTER INTEGRATION.**

The checkpoint loads on CUDA, the processor works, RGB preprocessing works,
generation succeeds, VRAM/RAM load measurements are recorded, and the adapter
preserves provenance. Phase 3E is **not** cleared for S2 multispectral, S1
SAR, optical-SAR fusion, temporal reasoning, grounding, or PEFT until those
separate contracts are implemented and tested.

## Validation and Scientific Safety

- Focused adapter and relevant tests: 29 passed.
- Python compile check: passed.
- `git diff --check`: passed.
- Scientific baseline unchanged: 65.0% accuracy, 4.1034286734 pp MAE,
  9.5873312123 pp RMSE.
- No scientific code, artifacts, datasets, splits, checkpoints, receipts,
  representations, or Phase 2F documents were modified.
- No training, fine-tuning, LoRA, QLoRA, commit, or push was performed.
