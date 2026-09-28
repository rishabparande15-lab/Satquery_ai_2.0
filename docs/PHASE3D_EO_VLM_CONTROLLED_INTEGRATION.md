# PHASE 3D — Controlled EO-VLM Integration and Local Smoke Test

**Status:** complete as a controlled integration attempt; model runtime blocked.

## 1. Executive Summary

Phase 3D reviewed the Phase 3C decision and implemented an isolated,
dependency-light `VisionInputAdapter` boundary for a small local VLM smoke
test. No VLM weights were downloaded, no model was loaded, and no training or
fine-tuning was performed.

The selected primary smoke-test target is **Qwen/Qwen2.5-VL-3B-Instruct**. It
was selected only as the smallest publicly available generic VLM comparison
candidate in the Phase 3C audit, not as the EO-VLM solution. The local runtime
is **BLOCKED / NOT TESTED** because `transformers`, `bitsandbytes`, and model
weights are not present in the repository environment.

The implemented adapter accepts only an explicit RGB image boundary. It
rejects S2 multispectral, S1 SAR, optical-SAR, temporal, CROMA, physical, and
hybrid inputs rather than silently converting them or claiming unsupported
capabilities.

## 2. Phase 3C Decision Basis

Phase 3C identified EarthDial and EarthGPT as the most relevant EO-VLM
research candidates, but their exact local checkpoint/runtime contracts and
8-GB behavior were not verified. Earth-OneVision had no verified runnable
release. Prithvi-EO-2.0 is an encoder, not a VLM. Qwen2.5-VL-3B has a public
checkpoint and is small enough to be a controlled generic comparison target,
while requiring explicit RGB input adaptation.

Primary sources reviewed:

- [Qwen2.5-VL-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen2.5-VL technical report](https://arxiv.org/abs/2502.13923)
- [EarthDial repository](https://github.com/hiyamdebary/EarthDial)
- [EarthGPT repository](https://github.com/wivizhang/EarthGPT)
- [Prithvi-EO-2.0 model family](https://huggingface.co/ibm-nasa-geospatial)
- [Phase 3C compatibility audit](PHASE3C_EO_VLM_COMPATIBILITY_AUDIT.md)

## 3. Candidate Selected for Smoke Test

| Role | Candidate | Reason |
| --- | --- | --- |
| Primary | Qwen2.5-VL-3B-Instruct | Public generic VLM checkpoint; smallest candidate selected for a bounded RGB adapter test |
| Optional secondary | None | No second model was downloaded or tested; avoiding unnecessary model-family downloads |

This is a technical feasibility selection, not a ranking or a claim of EO
fitness.

## 4. Model / Checkpoint Identity

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Source | Official Qwen Hugging Face model card |
| Revision | **NOT RECORDED LOCALLY; checkpoint not downloaded** |
| Expected parameter scale | Public model name indicates approximately 3B class; exact local file size **NOT VERIFIED** |
| Local path | None |
| File hash | Not applicable; no file downloaded |
| Configuration hash | Not applicable |

## 5. License / Source

The official model card is the source of truth for the checkpoint license and
usage terms. A local license verification was not performed because the model
was not downloaded. Status: **NOT VERIFIED LOCALLY**.

## 6. Local Environment and Hardware

The requested target is an NVIDIA RTX 5060 Laptop GPU with 8 GB VRAM and 24 GB
RAM. The repository environment was inspected without installing packages.

Observed runtime availability:

| Component | Status |
| --- | --- |
| PyTorch | Available |
| Transformers | Not installed |
| bitsandbytes | Not installed |
| PIL | Not installed in the active Python environment |
| Candidate weights | Not present |
| CUDA/model runtime measurement | Not performed |

No configuration was started that could exhaust VRAM or RAM.

## 7. Installation / Dependency Changes

No dependency was installed and no requirements file was changed. There is no
local model runtime in this phase. This keeps the scientific and ordinary test
environments independent of Hugging Face-specific objects.

## 8. Model Loading Results

**BLOCKED / NOT TESTED.** Loading could not be attempted responsibly because
the processor/runtime package and checkpoint were absent. No repeated retries
were made.

Recorded blocker:

- Required: candidate checkpoint, Transformers processor/model runtime, and a
  compatible quantization path if using 4-bit/8-bit.
- Available: PyTorch only; no checkpoint or model runtime.
- Resolution for Phase 3E: obtain an approved checkpoint, record its exact
  revision/license/size, install an isolated runtime, then run at most ten
  controlled inputs.

## 9. Quantization Results

No FP16/BF16, 8-bit, or 4-bit run occurred. All modes are **NOT TESTED**.
`Qwen25VLRGBAdapter.estimate_resources()` intentionally returns an estimate
with null peak-memory fields rather than inventing measurements.

## 10. Input Test Results

| Test | Status | Result |
| --- | --- | --- |
| Single RGB optical image | PASS at adapter-contract level | Validated and packaged as H×W×3 uint8-compatible input |
| S2 `[N,12,120,120]` | NOT SUPPORTED | No silent band selection or pseudo-RGB conversion implemented |
| S1 `[N,2,120,120]`, `VV,VH` | NOT SUPPORTED | No SAR adapter implemented |
| Optical + SAR pair | NOT SUPPORTED | No native joint input or fusion claimed |
| Temporal pair T1/T2 | NOT VERIFIED | No temporal model or two-image prompt path implemented |
| CROMA GAP `[768]` | NOT SUPPORTED | Not passed to the VLM |
| CROMA tokens `[225,768]` | NOT SUPPORTED | Not passed to the VLM |
| Physical 62-D / hybrid 830-D | NOT SUPPORTED | Remain scientific-tool representations |

The RGB preparation is explicitly an **input adapter**, not native EO,
multispectral, SAR, or optical-SAR reasoning.

## 11. Optical-SAR and Temporal Tests

No optical-SAR generation test was run. The adapter returns
`NOT SUPPORTED` for `OPTICAL_SAR`; it does not generate two independent
answers and call them fusion.

No temporal test was run. The repository's temporal contract requires T1, T2,
ordering, metadata, and validated spatial correspondence. The adapter rejects
`TEMPORAL`; no change labels, temporal embeddings, or synthetic pair were
created.

## 12. Grounding Test

**GROUNDING BLOCKED / NOT TESTED.** No model was loaded and no coordinate
contract was established. The adapter produces no bbox, point, mask, polygon,
region reference, or coordinate-token output. Phase 2F remains unchanged:
BigEarthNet.txt source geometry is preserved but not mapped to pixels or CROMA
tokens.

## 13. CROMA Compatibility

CROMA is not directly consumable by this adapter. No projection from
`[225,768]`, `[768]`, `62-D`, or `830-D` to language-model inputs was added.
The scientific predictor remains separate:

```text
scientific predictor: physical_62d + joint_croma_gap_768d → hybrid_830d → coverage output
language model:       explicit model-facing visual adapter → language output
```

Any future CROMA use would require a separately trained and provenance-aware
token/vector projection adapter, with a task contract that does not silently
reuse scientific predictor semantics.

## 14. Adapter Architecture

Added: `src/eo_vlm_adapter.py`.

`Qwen25VLRGBAdapter` provides:

- `validate_inputs(...)`
- `prepare_inputs(...)`
- `get_capabilities(...)`
- `get_provenance(...)`
- `estimate_resources(...)`

The implementation is model-isolated and does not import Transformers, load a
tokenizer, access the web application, or alter `src/image_language_foundation.py`.
It preserves the conceptual adapter boundary while leaving the existing
scientific and image-language contracts intact.

## 15. Provenance Handling

Prepared RGB inputs preserve:

- model identity;
- adapter identity;
- modality;
- input SHA-256;
- explicit `scientific_representation_used: false`;
- temporal metadata, when supplied;
- preprocessing identity.

No model text was produced. If a future runtime generates text, it must remain
typed as **MODEL TEXT**, distinct from scientific prediction, image/pixel
evidence, source metadata, and temporal evidence.

## 16. Resource Measurements

No model load or inference occurred. Therefore the following are all
**NOT MEASURED**:

- model load time;
- download size;
- peak VRAM;
- peak RAM;
- preprocessing latency;
- generation latency;
- output length;
- CUDA/offload behavior;
- FP16/BF16/8-bit/4-bit behavior.

The report intentionally contains no fabricated resource values.

## 17. Failures / Blockers

1. The checkpoint is not present locally.
2. Transformers and bitsandbytes are not installed.
3. No model-specific processor contract was executed.
4. RGB-only preparation cannot establish multispectral, SAR, optical-SAR,
   temporal, or grounding support.
5. 8 GB feasibility remains unmeasured.

## 18. Limitations

This phase proves the adapter boundary and validation behavior only. It does
not prove VQA, captioning, EO understanding, hallucination behavior,
grounding, temporal reasoning, benchmark compatibility, or local model
performance. A successful RGB adapter unit test is not a VLM smoke-test result.

## 19. Phase 3E Readiness Gate

| Question | Phase 3D result |
| --- | --- |
| Can the selected VLM run locally? | **NOT VERIFIED / BLOCKED** |
| Usable precision/device mode? | **NOT TESTED** |
| Usable modalities? | RGB contract only; runtime not tested |
| Required adapters? | RGB processor boundary now; multispectral, SAR, fusion, temporal, grounding, and evidence adapters still required |
| Joint optical-SAR representation? | **NOT SUPPORTED** by this adapter |
| Honest temporal representation? | Contractually rejected until a temporal-capable adapter exists |
| Grounding? | **GROUNDING BLOCKED** |
| Isolated behind `VisionInputAdapter`? | **PASS** at contract/unit-test level |
| PEFT/LoRA/QLoRA next phase? | Technically possible in principle, but **not approved** until checkpoint loading and resource measurements pass |

Phase 3E may proceed only after an isolated runtime setup records exact
checkpoint identity, license, quantization support, actual VRAM/RAM behavior,
and bounded tests for every supported modality. No adaptation should begin
before that gate.

## 20. Validation Results

Focused adapter tests: **6 passed**.

The existing scientific baseline remains unchanged:

- 5,000 areas; split 4,600 / 200 / 200.
- Accuracy: 65.0%.
- MAE: 4.1034286734 pp.
- RMSE: 9.5873312123 pp.

No scientific code, CROMA artifacts, predictor checkpoint, dataset, split,
manifest, receipt, representation, or Phase 2F document was modified. No
training, fine-tuning, LoRA, QLoRA, synthetic grounding labels, commit, or
push was performed.
