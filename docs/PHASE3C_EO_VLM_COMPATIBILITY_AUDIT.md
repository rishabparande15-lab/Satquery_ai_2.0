# PHASE 3C — EO-VLM Compatibility and Integration Feasibility Audit

**Status:** audit complete; no model selected, installed, trained, or fine-tuned.

**Scope:** public-source model compatibility, hardware feasibility, input/output
contracts, adapter design, benchmark readiness, and a safe architecture gate.
No scientific pipeline, checkpoint, dataset split, receipt, representation
artifact, or Phase 2F spatial policy was changed.

## 1. Executive Summary

SatQuery's validated scientific baseline remains a separate tool:

```text
S2 [N,12,120,120] + S1 [N,2,120,120]
  → frozen CROMA → physical_62d + joint_croma_gap_768d
  → hybrid_830d → frozen linear 19-output softmax probe
```

The future language-facing system should use an RS-adapted VLM/MLLM with a
model-specific input adapter and lightweight parameter-efficient adaptation
only after the adapter and benchmark contracts are proven. The audit does not
justify making `hybrid_830d` a universal VLM input: it is a scene-level
scientific predictor representation.

EarthDial and EarthGPT have the clearest public evidence of multisensor
remote-sensing language capability. EarthDial additionally documents
multispectral, SAR, temporal, and grounding-oriented tasks, but its published
4B variants and custom runtime are not verified here on an RTX 5060 Laptop GPU
with 8 GB VRAM. EarthGPT documents optical, SAR, and infrared comprehension,
but exact released checkpoints and local resource behavior remain
**UNKNOWN / NOT VERIFIED**. Earth-OneVision has a public paper record, but no
public runnable repository or checkpoint was verified in this audit. Prithvi-
EO-2.0 is a useful multispectral/temporal encoder, not a VLM. Qwen2.5-VL-3B
is a reasonable generic comparison baseline for RGB-style smoke testing, but
it does not natively consume SatQuery's 12-band/SAR contract.

The decision gate is therefore: design an adapter-swappable RS-VLM interface,
keep the scientific predictor independent, and defer model selection and any
LoRA/QLoRA work until a tiny reproducible inference test proves the exact
candidate, quantization, and evidence-output contract on the target machine.

## 2. SatQuery Requirements and Invariants

The current scientific inputs are:

| Object | Verified contract |
| --- | --- |
| Sentinel-2 | `[N,12,120,120]`, bands `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12` |
| Sentinel-1 | `[N,2,120,120]`, channels `VV,VH` |
| CROMA spatial output | `[N,225,768]` optical, SAR, and joint latent token arrays |
| CROMA scene output | `[N,768]` GAP vector |
| Physical representation | `physical_62d`, materialized |
| Scientific representation | `hybrid_830d = physical_62d || joint_croma_gap_768d`, materialized |
| Current scientific result | 65.0% accuracy; 4.1034286734 pp MAE; 9.5873312123 pp RMSE |

The eventual language layer must support single optical/multispectral input,
single SAR input, co-registered optical+SAR, bi-temporal corresponding pairs,
and eventually sequences. It must also support VQA, captioning, joint
optical-SAR reasoning, change description, evidence-linked answers, and
agent-selected task execution. These are requirements, not claims of current
implementation.

BigEarthNet.txt remains fail-closed for spatial supervision. Its source boxes
and points are preserved, but no authoritative mapping to the 120×120 grid or
225-token grid is established. No grounding labels or learned grounding score
may be derived from them.

## 3. Candidates Audited

The audit used public primary sources and model cards where available:

| Candidate | Primary evidence | Audit status |
| --- | --- | --- |
| EarthDial | [official repository](https://github.com/hiyamdebary/EarthDial), [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Soni_EarthDial_Turning_Multi-sensory_Earth_Observations_to_Interactive_Dialogues_CVPR_2025_paper.pdf), [paper record](https://arxiv.org/abs/2412.15190) | Audited; public code and named HF weights are documented |
| EarthGPT | [official repository](https://github.com/wivizhang/EarthGPT), [paper](https://arxiv.org/abs/2401.16822) | Audited; public code and dataset evidence, exact checkpoint contract not verified |
| Earth-OneVision | [paper](https://arxiv.org/abs/2606.10819) | Audited; public paper found, runnable release/checkpoint not verified |
| Prithvi-EO-2.0 | [official model family](https://huggingface.co/ibm-nasa-geospatial), [paper](https://arxiv.org/abs/2412.02732) | Audited as a vision encoder, not a VLM |
| Qwen2.5-VL-3B-Instruct | [official model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [technical report](https://arxiv.org/abs/2502.13923) | Generic RGB/image-text comparison baseline only |

No candidate was downloaded. Consequently, no local load, VRAM, RAM, or
latency measurement is claimed.

## 4. Architecture Comparison

`Yes` means supported by public model evidence for some input/task; it does not
mean supported through a SatQuery adapter or benchmarked here. `Unknown` means
the property was not established from a primary source or exact released
checkpoint.

| Model | RS-native | Optical | Multispectral | SAR | Optical+SAR | Temporal | Change | VQA | Captioning | Grounding | CROMA compatibility | Raw S1/S2 compatibility | Adapter complexity | 4-bit feasibility | RTX 5060 8GB | License | Weights available | Local inference tested | Main limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EarthDial | Yes | Yes | Yes | Yes | Claimed by paper/repo | Yes | Yes | Yes | Yes | Yes; grounded outputs reported | No direct contract | No | High | Unknown | Unverified; likely constrained | **UNKNOWN / NOT VERIFIED** | Public HF variants documented | No | Exact 4B runtime, memory, and SatQuery tensor adapter unverified |
| EarthGPT | Yes | Yes | Yes/IR | Yes | Yes, claimed | Unknown | Unknown | Yes | Yes | Yes; repository describes grounding | No direct contract | No | High | Unknown | Unknown | **UNKNOWN / NOT VERIFIED** | Public repo; exact weights not verified | No | Exact checkpoint, license, and deployment contract incomplete |
| Earth-OneVision | Claimed | Unknown | Unknown | Unknown | Claimed by paper title | Unknown | Unknown | Unknown | Unknown | Unknown | No direct contract | No | High | Unknown | Blocked pending release | **UNKNOWN / NOT VERIFIED** | Not verified | No | Public paper but no runnable release verified |
| Prithvi-EO-2.0 | EO encoder | HLS optical | Yes; six HLS bands | No | No native language fusion | Yes; temporal variants | Downstream-task dependent | No | No | None native | Encoder output could be projected | No; six-band HLS contract | Medium | Not applicable as VLM | Smaller variants potentially feasible; unmeasured | Apache-2.0 on official cards | Yes; 5M/100M/300M/600M families listed | No | Encoder only; not a language-facing VLM |
| Qwen2.5-VL-3B | No | Yes, RGB/image | No | No | No native EO fusion | Video, not EO temporal reasoning | No native EO change task | Yes | Yes | General visual localization capabilities; not EO grounding contract | No | No | Medium | Public quantized variants exist; exact target path unmeasured | Potentially feasible only after measurement | **UNKNOWN / NOT VERIFIED** | Yes, public generic ecosystem; not tested here | Generic RGB model; spectral/SAR semantics must be adapted |

The matrix deliberately avoids a score or winner. “Available model” means a
public artifact is referenced; it does not mean SatQuery can safely use it.

## 5. Hardware Feasibility

Target hardware is an RTX 5060 Laptop GPU with 8 GB VRAM and 24 GB system
RAM. Exact candidate memory numbers were not available from the audited primary
sources, so the following are feasibility hypotheses, not measurements:

| Precision | What can be stated safely |
| --- | --- |
| FP32 | 4B language weights alone require roughly 16 GB before runtime state and vision modules; therefore full-GPU FP32 is not feasible on 8 GB. This is a parameter-count estimate, not a measured candidate result. |
| FP16/BF16 | 4B weights are roughly 8 GB before KV cache, activations, vision tower, and framework overhead; full-GPU fit is therefore not established and likely requires offload or lower precision. |
| 8-bit | Weight storage is roughly 4 GB for 4B parameters before overhead; actual fit and speed are unknown for each candidate. |
| 4-bit | Weight storage is roughly 2 GB for 4B parameters before quantization scales, vision tower, KV cache, and runtime overhead. 4-bit inference is technically plausible for some 3B/4B models, but candidate-specific support and quality are unknown. |
| 24 GB RAM | CPU offload may be possible for some small candidates, but no candidate load or latency measurement was run. |

The numbers above are simple parameter-count estimates (`parameters × bytes`)
and must not be reported as measured VRAM. Exact model size, processor memory,
KV-cache behavior, and offload performance are a Phase 3D smoke-test gate.
LoRA/QLoRA is a future adaptation option, not performed here.

## 6. Input and CROMA Compatibility

No audited candidate directly accepts SatQuery's raw 12-band S2 and 2-channel
S1 tensors under the current scientific preprocessing contract. For each
candidate, compatibility is therefore an adapter question:

| Input | Candidate status | Required interface |
| --- | --- | --- |
| Raw S2 12-band tensor | Not directly verified for any language candidate | `MultispectralBandGroupingAdapter` or a native multispectral processor; preserve band names, scale, grid, dtype, and provenance |
| Raw S1 VV/VH tensor | EarthDial/EarthGPT report SAR capability, but exact tensor contract is unknown | `SARInputAdapter`; establish normalization, channel order, polarization metadata, and image-to-token contract |
| Optical RGB | Likely supported by EarthDial, EarthGPT, and Qwen-style VLMs | `RGBImageAdapter`; this is lossy relative to 12-band science |
| SAR pseudo-RGB | Not native learned SAR semantics unless candidate explicitly provides it | `SARToModelInputAdapter`; adaptation is not native optical-SAR fusion |
| CROMA GAP `[768]` | No direct language-input contract verified | `CromaProjectionAdapter`; project/vector-tokenize only after a model-specific training contract |
| CROMA tokens `[225,768]` | No direct language-input contract verified | `CromaTokenProjectionAdapter` plus positional and provenance metadata |
| Physical 62-D | Not a visual input | `StructuredEvidenceAdapter`; pass as tool/evidence data, not an image embedding |
| Hybrid 830-D | Scientific predictor input only | Keep behind `ScientificPredictorTool`; do not inject into VLM by default |

The required adapter categories are: raw-image preprocessing, multispectral
band grouping, SAR input conversion, optical-SAR fusion, CROMA token
projection, temporal multi-image packaging, spatial grounding, and
evidence/provenance binding. They are specified here only; none is implemented.

## 7. Optical-SAR and Temporal Compatibility

The language layer must preserve modality identity and enable cross-modal
fusion. Two independent prompts or two unrelated image answers are not
optical-SAR reasoning. A future model adapter must expose either a native
cross-modal input contract or an explicit learned/engineered fusion module.
Pseudo-RGB conversion is input adaptation, not evidence of native SAR
understanding or optical-SAR fusion.

Temporal input must be a first-class object:

```text
TemporalPair {
  T1: image/representation + acquisition metadata
  T2: image/representation + acquisition metadata
  optional_metadata: sensor, CRS, transform, resolution, provenance
  spatial_correspondence: validated contract, otherwise BLOCKED
  temporal_order: explicit T1 < T2
}
```

The output must preserve `T1 evidence`, `T2 evidence`, explicit
`difference/change evidence`, and the language answer. Asking a single-image
VLM about two separately rendered images is not a validated temporal model.
EarthDial publicly claims bi-temporal/multi-temporal and change capabilities;
for the other audited candidates, true EO temporal reasoning is
**UNKNOWN / NOT VERIFIED** or absent. SatQuery's own temporal representations
remain deferred.

## 8. Grounding Compatibility

Grounding has three separate layers:

1. source annotation geometry;
2. deterministic mapping into a validated image/token coordinate frame;
3. learned visual grounding output.

EarthDial reports grounded responses and region-oriented output; its exact
coordinate schema must be verified before use. EarthGPT describes visual
grounding, but its exact output schema is **UNKNOWN / NOT VERIFIED** here.
Earth-OneVision and Prithvi-EO-2.0 are **UNKNOWN**/not applicable from the
audited evidence. Qwen2.5-VL has general visual capabilities but no
SatQuery-compatible EO grounding contract.

Possible native output types are `bbox`, `point`, `mask`, `polygon`,
`coordinate tokens`, or `none`; no candidate was accepted as a SatQuery
grounding provider in this phase. In particular, no BigEarthNet.txt geometry
was mapped and no grounding labels were created.

## 9. Benchmark Compatibility

| Benchmark | Data in repository | Model | Evaluator | Implemented evaluation |
| --- | --- | --- | --- | --- |
| BigEarthNet.txt | Available; source-preserving foundation records | Not implemented | Not implemented for learned VLM | Not implemented |
| RSVQA | Not available locally | Not implemented | Not implemented | Not implemented |
| VRSBench | Not available locally | Not implemented | Not implemented | Not implemented |
| CDVQA | Not available locally; temporal contract utilities only | Not implemented | Not implemented | Not implemented |
| Pipeline 3 coverage | Available frozen 5,000-area scientific artifacts | Frozen CROMA + linear probe | Implemented | Verified, but not a VLM metric |

Dataset existence, model availability, evaluator availability, and an
implemented score are intentionally reported as separate states.

## 10. Adapter Interface Specification

The following is implementation-neutral and model-swappable:

```python
class VisionInputAdapter(Protocol):
    def validate_inputs(self, request: VisionRequest) -> ValidationReport: ...
    def prepare_inputs(self, request: VisionRequest) -> PreparedVisionInput: ...
    def get_capabilities(self) -> VisionCapabilities: ...
    def get_provenance(self, prepared: PreparedVisionInput) -> ProvenanceRecord: ...
    def estimate_resources(self, prepared: PreparedVisionInput) -> ResourceEstimate: ...
```

`VisionRequest` may contain one image, an optical/SAR pair, or a validated
temporal sequence. `PreparedVisionInput` must retain modality, band/polarization
semantics, shape, dtype, normalization, spatial metadata, and source hashes.
`VisionCapabilities` must state supported modalities, tasks, temporal behavior,
grounding output types, and unsupported inputs. `ResourceEstimate` must label
values as measured, estimated, or unknown.

The agent/controller depends only on this interface, so EarthDial, EarthGPT,
Earth-OneVision, or a later EO-VLM can be swapped without redesigning query
planning. The scientific predictor remains a separate tool returning
land-cover evidence.

## 11. Proposed VLM Integration Architecture

```text
USER QUERY
    ↓
QUERY UNDERSTANDING
    ↓
TASK PLANNER
    ↓
INPUT / GEO VALIDATOR
    ↓
REPRESENTATION SELECTOR
    ↓
VISION INPUT ADAPTER
    ↓
RS-ADAPTED VLM / MLLM
    ↓
TASK OUTPUT
    ↓
EVIDENCE RECONCILER
    ↓
ANSWER GENERATOR
    ↓
AUDIT / PROVENANCE

Scientific Predictor → land-cover evidence
VLM                → language / visual reasoning
```

The Evidence Reconciler must not silently turn a VLM assertion into a
scientific prediction. It should preserve source, representation, model,
prompt, timestamp, coordinate frame, uncertainty/calibration state, and
whether a claim is deterministic, learned, or unavailable.

## 12. Smoke-Test Results

No smoke test was run. No candidate was installed or downloaded, and no
candidate was locally runnable without first changing the environment and
obtaining model artifacts. This is recorded as **NOT RUN / BLOCKED BY AUDIT
SCOPE**, not as a performance result. There are consequently no measured
values for model load, download size, VRAM, RAM, preprocessing time, inference
time, output quality, CUDA/offload behavior, or warnings.

## 13. Risks and Blockers

- Exact checkpoint identities, licenses, and quantized runtime paths are
  incomplete for some candidates.
- Public capability claims do not establish a SatQuery tensor contract.
- 8 GB VRAM feasibility is unmeasured; parameter-count arithmetic is not a
  substitute for runtime measurement.
- BigEarthNet.txt grounding remains blocked by unresolved spatial semantics.
- Current benchmark datasets/evaluators do not establish VQA, captioning,
  grounding, or change-reasoning scores.
- The 830-D scientific vector is not a universal language-facing embedding.
- Temporal representations and learned optical-SAR fusion remain deferred.

## 14. Decision Gate for Phase 3D

Technically viable directions are conditional, not selected models:

- EarthDial: technically promising for the required modality/task envelope,
  pending exact checkpoint, resource, and adapter tests.
- EarthGPT: technically promising for optical/SAR language tasks, pending
  exact public checkpoint and input/output verification.
- Qwen2.5-VL-3B: technically useful as a lightweight generic RGB baseline,
  not as the EO solution.
- Prithvi-EO-2.0: useful as a multispectral/temporal vision encoder only.

Blocked or unverified directions:

- Earth-OneVision: blocked for integration by missing verified runnable release.
- Any full-precision 4B-class VLM: blocked from assumed full-GPU deployment by
  the 8 GB target until measured quantization/offload behavior is available.
- BigEarthNet.txt learned grounding: blocked by Phase 2F semantics.

Required next gate: acquire one approved small checkpoint, run a 5–10 image
smoke test on the target machine, verify structured output and provenance,
then separately validate one optical, SAR, paired optical-SAR, and temporal
adapter contract. Only after that should a PEFT/LoRA/QLoRA experiment be
considered. No training is approved by this document.

## 15. Reproducibility and Provenance

Repository evidence inspected includes:

- `docs/PHASE3A_BENCHMARK_RERUN_AUDIT.md`
- `docs/PHASE3B_MODEL_LANDSCAPE_AUDIT.md`
- `docs/PHASE0_POST_PHASE2_REPRODUCIBILITY.md`
- `docs/PHASE2H_REPRESENTATION_COMPUTE_STORAGE_POLICY.md`
- `src/phase1_foundation.py`
- `src/croma_adapter.py`
- `src/pipeline3_scene_probe.py`
- `src/image_language_foundation.py`
- `src/temporal.py`
- `src/region_grounding.py`

External evidence is linked in Section 3. No external model weights or
datasets were downloaded. The report records capability evidence separately
from local execution evidence.

## 16. Validation Results

- No implementation files were added.
- No model weights, checkpoints, scientific artifacts, datasets, splits, or
  receipts were modified.
- No Phase 2F spatial semantics were modified.
- No training, fine-tuning, LoRA, QLoRA, or synthetic label generation was
  performed.
- The locked scientific metrics remain 65.0% accuracy,
  4.1034286734 pp MAE, and 9.5873312123 pp RMSE.
- Existing staged work and the prior README change were preserved.
- `git diff --check` is required as the final repository validation command.

**Final status:** Phase 3C compatibility audit complete; no VLM selected and
no VQA implementation started.
