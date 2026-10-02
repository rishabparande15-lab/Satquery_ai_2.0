# SatQuery AI v1 — Final Project Package

## A. Report-ready content

### 1. Title

**SatQuery AI: A Paired Sentinel-1 and Sentinel-2 Multimodal Visual-Language System for Satellite Scene Queries**

### 2. Abstract

SatQuery AI v1 is a multimodal remote-sensing visual-language system for answering semantic questions about paired Earth-observation imagery. The primary route accepts a BigEarthNet Sentinel-1 VV/VH SAR patch of shape `[2,120,120]` and a co-registered Sentinel-2 twelve-band multispectral patch of shape `[12,120,120]`. It validates the pair, obtains frozen joint CROMA tokens `[1,225,768]`, maps them through a trained `CromaJointProjector` to Qwen-compatible visual tokens `[1,16,2048]`, and uses frozen Qwen2.5-VL-3B-Instruct to produce Binary, multiple-choice, or caption-style structured responses. The final end-to-end fixed panel contained 30 non-TEST BigEarthNet records and completed 30/30 requests. The system is technically validated and reproducible, including an exact fresh-process projector reload. However, the bounded comparison did not show fusion improvement over the S2-only baseline; S2-only was stronger on the small Binary/MCQ panel, modality identity weakly affected generated decisions, and final caption outputs were empty. SatQuery v1 is therefore presented as a functioning multimodal architecture with explicit scientific limitations, not as a benchmark or state-of-the-art claim.

### 3–7. Introduction, problem, motivation, objectives, and scope

Remote-sensing questions often require information from complementary sensors. Sentinel-1 contributes radar observations that can operate independently of illumination and cloud conditions, while Sentinel-2 provides rich multispectral optical information. SatQuery AI investigates a practical way to connect paired Earth-observation representations to a language-model interface.

The problem is to build a robust paired-input system that can accept satellite patches and return structured answers without silently dropping a modality. The project objectives were to: (1) validate paired S1/S2 inputs, (2) produce a joint CROMA representation, (3) adapt that representation to Qwen’s visual-token interface, (4) support Binary, MCQ, and caption-style responses, (5) preserve provenance and fail closed on invalid input, and (6) measure reproducibility and limitations. The scope is limited to BigEarthNet paired patches and a small fixed non-TEST panel. Temporal reasoning, grounding, new datasets, benchmark claims, and generalization claims are out of scope.

### 8. Technology background

- **Sentinel-1:** a Synthetic Aperture Radar (SAR) mission. Its VV and VH polarizations capture microwave backscatter properties of the surface.
- **Sentinel-2:** a multispectral optical mission. SatQuery uses 12 bands in canonical order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, and B12.
- **BigEarthNet:** the project’s only implemented final dataset source, providing paired Sentinel-1/Sentinel-2 patches and associated scene metadata.
- **CROMA:** the frozen paired Earth-observation representation model used to obtain joint S1+S2 latent tokens. CROMA was not trained in this project.
- **Qwen2.5-VL-3B-Instruct:** the frozen vision-language/language model used to generate the final response.
- **Vision-language model (VLM):** a model interface that combines visual representations with language prompts and produces text or structured language answers.

### 9–18. Proposed system, architecture, preprocessing, and model integration

The final system is deliberately multimodal:

```text
Sentinel-1 VV/VH [2,120,120] ─┐
                               ├─ paired validation/preprocessing
Sentinel-2, 12 bands [12,120,120] ─┘
                 ↓
     official paired frozen CROMA
     joint tokens [1,225,768]
                 ↓
  trained CromaJointProjector
     visual tokens [1,16,2048]
                 ↓
 frozen Qwen2.5-VL-3B-Instruct
                 ↓
 Binary / MCQ / Caption structured response
```

The loader validates paired BigEarthNet identity, CRS, footprint/bounds, common processing-grid metadata, shapes, channel counts, and finite values. Sentinel-1 is represented by VV/VH; Sentinel-2 is represented by the canonical twelve-band tensor. The official paired CROMA encoder produces joint 225-token, 768-wide features. `CromaJointProjector` is the learned adaptation layer: it pools the 15×15 joint-token grid to 16 Qwen-facing tokens and projects width 768 to 2048. Qwen is pinned at revision `66285546d2b821cf421d4f5eb2576359d3770cd3` and remains frozen.

The primary implementation is `src/satquery_v1.py`, class `SatQueryV1Controller`, method `run_satquery(...)`. It returns status, task type, generated response, parsed answer when applicable, route, modalities used, patch identity, CROMA mode, joint-projector SHA, Qwen revision, warnings, error code, provenance, and diagnostics.

### 19–23. Tasks, controller, validation, and error handling

Binary questions parse `yes` or `no`; MCQ parses `a`, `b`, `c`, or `d`; captions return the generated text without inventing a quality score. The normal route is `MULTIMODAL_S1_S2` and requires both modalities. S2-only is a separately callable baseline/ablation; S1-only is research-only. There is no automatic modality selection and no silent S2 fallback.

Missing S1 returns `MISSING_S1_FOR_MULTIMODAL_ROUTE`; missing S2 returns `MISSING_S2_FOR_MULTIMODAL_ROUTE`; mismatched identities return `MISMATCHED_PATCH_ID`. Wrong shapes, NaN/Inf, incomplete spatial provenance, unavailable assets, invalid tasks, and unapproved checkpoints fail closed.

### 24–29. Training and experimental methodology

Only the final joint projector was adapted for the final multimodal path; CROMA and Qwen were frozen. The approved final projector checkpoint SHA is `bedcb8dc60d3375878b6fe732cb90b568283c7d5565148fb1a52eae0d1b12eaa` and its fingerprint is `8b3e05aa2346c24f662d747ab59d47fee8c2a96dc1728e87d2dce6b30a4d1a69`.

S2 work established a twelve-band projector path and diagnosed a generation-interface issue. SAR work established a technically valid S1 representation path, but standalone SAR semantic reasoning was not demonstrated because labels were not shown to be SAR-inferable (`SAR_LABEL_MODALITY_MISMATCH`). Fusion work compared correct, shuffled, and zeroed modalities on the same fixed panel. It found `OPTICAL_SAR_TECHNICALLY_VALID_S2_DOMINANT`: the joint route works, but no clear S1+S2 semantic advantage was established.

### 30–35. Results, resources, reproducibility, and key challenge

**Final Phase 3R E2E panel (30 non-TEST records):**

| Task | Successful | Parseable | Correct / nonempty |
| --- | ---: | ---: | ---: |
| Binary | 10/10 | 7/10 | 4/10 correct |
| MCQ | 10/10 | 9/10 | 1/10 correct |
| Caption | 10/10 | — | 0/10 nonempty |

**Fixed S2-only versus joint comparison:**

| Task | S2-only | Joint S1+S2 | Joint wins/losses/ties |
| --- | --- | --- | --- |
| Binary | 5/10 correct; 1.00 parseability | 4/10 correct; 0.70 parseability | 3 / 4 / 3 |
| MCQ | 3/10 correct; 1.00 parseability | 1/10 correct; 0.90 parseability | 1 / 3 / 6 |
| Caption | 1 unique nonempty mode | 1 unique nonempty mode | No clear advantage |

The final route took 52.310 seconds total, 1.744 seconds/record, with peak VRAM of 8,389,146,112 bytes and peak RAM of 4,851,982,336 bytes on CUDA. This is profiling, not a speed-superiority claim.

Fresh-process reload reproduced the precise validation mean exactly: original `2.3361381590366364`, reload `2.3361381590366364`, absolute difference `0.0`. Qwen, CROMA, historical S1/S2 projectors, and the final joint projector were unchanged.

A major engineering challenge was Hugging Face generation with `inputs_embeds`: placeholder `input_ids` could be dropped during prefill. A scoped generation wrapper restores those IDs for the first step. Direct-forward and generation first-step logits then matched with L2 difference `0.0`.

### 36–39. Advantages, limitations, future work, and conclusion

Advantages include a real paired-sensor architecture, validated modality contracts, frozen-model reproducibility, structured provenance, and explicit failure handling. Limitations are equally important: the panel is small and pilot-scale; fusion did not improve over S2-only; generated decisions showed zero changes under shuffled/zeroed controls despite candidate-score changes; captions collapsed to empty output in the final E2E panel; and standalone SAR semantic reasoning was not established. No benchmark, SOTA, or generalization claim is made, and TEST was not accessed.

Future work may investigate modality-aware objectives, stronger decision-level dependence on both sensors, caption recovery, larger controlled evaluation, carefully authorized Qwen adaptation/LoRA, stronger fusion, reintroduction of hybrid physical/GEE features, temporal/change detection, grounding, and external evaluation. These are future work only.

In conclusion, SatQuery AI v1 demonstrates a complete, reproducible S1+S2 remote-sensing VLM architecture. Its technical integration is successful; its scientific effectiveness remains limited and is reported transparently.

### 40. Reference suggestions

Use only sources you have verified and cite them in the required institutional style. Suitable categories are: official ESA Sentinel-1 and Sentinel-2 mission documentation; the official BigEarthNet publication/documentation; the CROMA paper and official implementation; Qwen2.5-VL technical documentation/model card; Hugging Face Transformers generation documentation; and general VLM or multimodal-learning surveys. Do not invent bibliographic entries.

## B. Presentation plan (15 slides)

| Slide | Title and bullets | Suggested visual | What to say |
| --- | --- | --- | --- |
| 1 | **SatQuery AI v1**; paired S1+S2 VLM; BigEarthNet; final project | Satellite collage plus title | “This project connects paired radar and multispectral imagery to a language interface.” |
| 2 | Problem: EO data are multimodal; queries need a usable interface; robust pairing matters | Radar/optical contrast | “The challenge is not merely loading images; it is preserving both modalities through a validated route.” |
| 3 | Objectives: pair validation; joint representation; structured answers; reproducibility | Four-icon objective diagram | “My goal was a technically defensible multimodal system, not an unsupported benchmark claim.” |
| 4 | BigEarthNet; S1 VV/VH `[2,120,120]`; S2 12 bands `[12,120,120]` | Input tensor cards | “S1 captures radar backscatter, while S2 provides spectral information.” |
| 5 | Primary route requires S1+S2; S2-only baseline; S1-only research | Architecture policy diagram | “The final architecture is multimodal; baseline results do not silently change the route.” |
| 6 | CROMA joint `[1,225,768]`; projector `[1,16,2048]`; frozen Qwen | Tensor-shape flow | “The projector is the bridge from remote-sensing tokens to Qwen’s embedding width.” |
| 7 | Research journey: S2 diagnostics; SAR diagnostic; fusion controls; integration | Timeline | “The system was built through bounded experiments and fail-closed gates.” |
| 8 | S2 and S1 findings; S2 stronger on fixed QA; SAR semantic mismatch | Two-column comparison | “A numerically valid path is not automatically semantic evidence.” |
| 9 | Fusion controls: shuffled/zero S1/S2; full shuffle; no output changes | Ablation grid | “I tested whether identities mattered rather than assuming different embeddings prove conditioning.” |
| 10 | Final controller: validation; CROMA; projector; Qwen; provenance | Controller flow | “Missing or mismatched modalities fail closed; there is no silent fallback.” |
| 11 | 30/30 E2E; Binary 4 correct; MCQ 1 correct; captions empty | Results table | “These are pilot integration results, not benchmark scores.” |
| 12 | Demo flow: inputs; route; response; provenance | Screenshot/terminal JSON | “The demo visibly proves that both S1 and S2 entered the CROMA joint route.” |
| 13 | Limitations: no fusion gain; weak decision sensitivity; caption collapse | Warning/limitations card | “Negative results are part of the contribution and guide the next work.” |
| 14 | Future work: stronger objectives; captions; larger evaluation; LoRA under approval | Roadmap | “These are proposals, not implemented features.” |
| 15 | Conclusion: complete reproducible multimodal architecture; transparent limits | Summary graphic | “SatQuery v1 is technically complete, while its scientific claims remain deliberately conservative.” |

## C. Architecture diagram specification

Draw four horizontal layers:

1. **Input/validation layer:** S1 VV/VH `[2,120,120]` and S2 12-band `[12,120,120]`; arrows meet at “paired identity, CRS, footprint, grid, finite-value validation.” Draw red fail-closed branches for missing S1, missing S2, mismatched identity, bad shape, and NaN/Inf.
2. **Representation layer:** “Frozen official paired CROMA” with output “joint tokens `[1,225,768]`.”
3. **Language layer:** “trained `CromaJointProjector`: `[1,225,768] → [1,16,2048]`” then “frozen Qwen2.5-VL-3B-Instruct.”
4. **Response/provenance layer:** Binary/MCQ/Caption response plus route `MULTIMODAL_S1_S2`, `CROMA_JOINT`, patch ID, projector SHA, Qwen revision, warnings, and error code.

Place “S2-only baseline/ablation” and “S1-only research/ablation” as dashed side paths, never as replacements for the central route.

## D. 3–5 minute demo script

“SatQuery AI v1 answers satellite-scene questions from a paired BigEarthNet sample. I provide a Sentinel-1 VV/VH tensor, a Sentinel-2 twelve-band tensor, the shared patch identity and spatial metadata, plus a task and question. The controller first checks that both modalities are present, shaped correctly, finite, and associated with the same patch. If S1 or S2 is missing, it fails closed rather than silently switching to an S2-only route.

Next, the validated pair enters the official CROMA joint encoder. CROMA returns `[1,225,768]` joint tokens. The trained CromaJointProjector converts these to `[1,16,2048]` Qwen-compatible tokens. Frozen Qwen then generates a task-specific response. The final JSON shows `MULTIMODAL_S1_S2`, `CROMA_JOINT`, both modalities used, patch identity, checkpoint SHA, Qwen revision, the generated response, and parsed answer.

If an output is incorrect, I would not claim it is correct. I would show the provenance, identify the task limitation, and explain that the project’s small fixed-panel study did not establish fusion accuracy gains. The demo proves the end-to-end architecture and validation behavior, not perfect semantic accuracy.”

## E. Viva preparation (40 concise questions)

1. **What is SatQuery AI?** A paired S1+S2 remote-sensing VLM interface for structured scene queries.
2. **Why use both S1 and S2?** They provide complementary radar and multispectral observations.
3. **What is SAR?** Active microwave sensing that measures backscatter from the Earth’s surface.
4. **Why VV and VH?** They are Sentinel-1 polarization channels with different scattering information.
5. **Why Sentinel-2?** Its twelve bands provide rich optical/spectral scene information.
6. **Why BigEarthNet?** It provides paired Sentinel-1/Sentinel-2 Earth-observation patches used consistently in v1.
7. **What is CROMA?** A frozen Earth-observation representation model that produces optical, SAR, and joint features.
8. **Was CROMA trained here?** No.
9. **Why Qwen2.5-VL?** It provides the frozen language-generation interface after visual-token adaptation.
10. **Why freeze Qwen?** To limit trainable scope, GPU risk, and unintended model drift.
11. **What is the projector?** The learned layer that maps CROMA joint features into Qwen’s visual embedding space.
12. **Why 768 to 2048?** CROMA joint tokens are width 768; the Qwen interface expects width 2048.
13. **Why 225 to 16 tokens?** CROMA has a 15×15 token grid; Qwen’s configured visual interface uses 16 projected tokens.
14. **What is multimodal fusion?** Combining information from more than one sensor modality.
15. **Did fusion outperform S2-only?** Not on the fixed pilot panel; S2-only was stronger on Binary and MCQ.
16. **Why keep fusion as the final architecture?** The project requirement is a technically validated paired multimodal system; limitations are documented honestly.
17. **Why did captions fail?** Final E2E captions were empty, consistent with previously observed prefix-lock-in/mode-collapse behavior.
18. **What is mode collapse here?** Generation repeatedly collapses to a narrow or empty output pattern rather than meaningful varied captions.
19. **What generation bug was found?** Placeholder IDs could be dropped when generation used `inputs_embeds`.
20. **How was it fixed?** A scoped wrapper restored placeholder IDs during generation prefill.
21. **How was the fix checked?** Direct-forward and generation first-step logits matched with L2 difference 0.0.
22. **Why was standalone SAR difficult?** Existing labels were not proven inferable from SAR alone.
23. **What is label-modality mismatch?** Targets depend on information a modality may not contain sufficiently to support the task.
24. **What does shuffled-modality testing show?** Whether changing modality identity changes observable output behavior.
25. **What is an ablation?** A controlled comparison that removes or isolates a component.
26. **What is TEST leakage?** Using held-out test information during development or model selection.
27. **Why keep TEST untouched?** To preserve evaluation integrity; v1 development used no TEST records.
28. **What happens if S1 is missing?** `MISSING_S1_FOR_MULTIMODAL_ROUTE` is returned.
29. **What happens if S2 is missing?** `MISSING_S2_FOR_MULTIMODAL_ROUTE` is returned.
30. **What if patches are mismatched?** The controller fails closed with `MISMATCHED_PATCH_ID`.
31. **What is the final output?** A structured response with answer, parsing, route, provenance, warnings, and error state.
32. **What is the main technical contribution?** A reproducible paired CROMA-to-Qwen end-to-end route with strict modality validation and provenance.
33. **How much GPU memory was used?** Peak measured VRAM was 8,389,146,112 bytes for the final route.
34. **How reproducible is it?** The final projector reload exactly matched the original validation mean.
35. **What does EXACT_MATCH mean?** Original and fresh-process reload means were numerically identical, with difference 0.0.
36. **What is CLC?** Corine Land Cover, a land-cover nomenclature commonly associated with BigEarthNet context; it is not a new v1 dataset claim.
37. **What is a VLM versus an LLM?** An LLM handles language; a VLM combines visual representations with language processing.
38. **Why not fully fine-tune Qwen?** The project bounded adaptation to the projector to avoid large, uncontrolled fine-tuning.
39. **What are the key limitations?** No demonstrated fusion gain, weak decision-level modality sensitivity, empty final captions, and a small pilot panel.
40. **What would you improve next?** Controlled modality-aware objectives, caption generation, larger evaluations, then separately authorized Qwen adaptation and broader tasks.

## F. Short explanations

**30 seconds:** “SatQuery AI v1 is a multimodal satellite-query system. It takes paired Sentinel-1 radar and Sentinel-2 multispectral patches from BigEarthNet, combines them with frozen CROMA, maps the joint representation into frozen Qwen, and returns structured Binary, MCQ, or caption-style outputs. The pipeline is technically validated, but I clearly report that its small pilot evaluation did not prove fusion accuracy gains.”

**1 minute:** Add the tensor path, strict paired validation, no silent S2 fallback, 30/30 E2E success, and the caption limitation.

**2 minutes:** Add the projector’s `[1,225,768] → [1,16,2048]` role, frozen Qwen/CROMA policy, controlled shuffled-modality study, and exact reload result.

**5-minute technical explanation:** Use slides 4–13 in order: inputs, contracts, CROMA, projector, Qwen interface, generation fix, ablations, E2E metrics, resource profile, and limitations.

**For a professor:** Emphasize controlled validation, frozen components, reproducibility, TEST exclusion, and the difference between technical integration and semantic performance.

**For a recruiter:** Emphasize end-to-end systems engineering: multimodal validation, model-interface adaptation, generation debugging, structured provenance, testing, reproducibility, and transparent reporting.

**For a non-technical person:** “It is a satellite assistant that reads two complementary kinds of satellite data together, checks they belong to the same place, and tries to answer a question while showing how the answer was produced.”

## G. Resume content

**Title:** SatQuery AI v1 — Multimodal Satellite Visual-Language System

**One line:** Built a paired Sentinel-1/Sentinel-2 BigEarthNet query pipeline using frozen CROMA, a learned joint projector, and frozen Qwen2.5-VL-3B-Instruct.

- Implemented a fail-closed S1+S2 CROMA-joint → Qwen inference route with structured provenance, modality validation, and no automatic S2 fallback.
- Developed and validated a 768-to-2048 joint-token projector interface; exact fresh-process reload reproduced the recorded validation mean.
- Built controlled S2/S1/joint ablations, generation-interface diagnostics, 30-record end-to-end integration checks, pytest coverage, and a paired-input demo.

**Stack:** Python, PyTorch, Hugging Face Transformers, CUDA, pytest, Git/GitHub, Sentinel-1, Sentinel-2, BigEarthNet, CROMA, Qwen2.5-VL-3B-Instruct.

## H. GitHub copy

**Description:** Paired Sentinel-1 + Sentinel-2 BigEarthNet visual-language system using frozen CROMA, a joint projector, and frozen Qwen.

**Opening paragraph:** SatQuery AI v1 is a technically validated paired remote-sensing VLM route. It requires both S1 VV/VH SAR and S2 multispectral inputs, validates their identity and spatial metadata, produces CROMA joint features, and returns structured language responses with provenance.

**Features:** strict paired validation; CROMA joint fusion; Qwen-compatible projector; Binary/MCQ/Caption interfaces; fail-closed missing-modality handling; checkpoint provenance; reproducibility receipt; explicit S2-only baseline.

**Results:** 30/30 non-TEST E2E requests succeeded. This is an integration result, not benchmark performance. The bounded comparison did not establish fusion improvement over S2-only.

**Limitations and future work:** Link to `docs/SATQUERY_V1_LIMITATIONS.md`; state that caption generation and decision-level modality dependence require future research.

## I–J. Stack and contributions

| Area | Technology / contribution |
| --- | --- |
| Programming | Python |
| Deep learning | PyTorch |
| VLM | Hugging Face Transformers; frozen Qwen2.5-VL-3B-Instruct |
| Remote sensing | Sentinel-1, Sentinel-2, BigEarthNet, frozen CROMA |
| Geospatial work | Existing project raster, pairing, and feature-processing components |
| Testing and delivery | pytest; Git/GitHub; CUDA |
| Architecture | Primary paired S1+S2 CROMA-joint route |
| Data integration | Strict shape, identity, metadata, and finite-value validation |
| Representation adaptation | `CromaJointProjector` maps CROMA tokens to Qwen tokens |
| Generation engineering | Fixed `inputs_embeds` placeholder-ID prefill behavior |
| Scientific diagnostics | S2/S1/joint controls with shuffle and zero ablations |
| Reproducibility | Checkpoint SHA/fingerprint and exact reload mean |
| Robustness | Structured errors and no silent modality fallback |

## K. Additional report tables

| Component | Trainable in final adaptation? | Frozen in final route? |
| --- | ---: | ---: |
| Official CROMA | No | Yes |
| Qwen2.5-VL-3B-Instruct | No | Yes |
| CromaJointProjector | Yes, during bounded adaptation only | Yes, in final inference |
| Historical S1 projector | No | Yes |
| Historical S2 projector | No | Yes |

| Capability | v1 status |
| --- | --- |
| Multimodal Binary | Supported interface |
| Multimodal MCQ | Supported interface |
| Multimodal Caption | Supported interface; empty-output limitation observed |
| S2-only | Baseline/ablation only |
| S1-only | Research/ablation only |
| Temporal change | Unsupported |
| Grounding | Unsupported |

## L. Future work

Future work should remain separate from v1 claims: improve modality-aware training objectives; increase decision-level S1/S2 dependence; recover caption generation; evaluate larger controlled validation and later test sets; investigate carefully authorized Qwen LoRA/adaptation; improve fusion; revisit hybrid physical/GEE features; and add temporal/change detection, spatial grounding, and larger-scale external evaluation only after their own validation gates.
