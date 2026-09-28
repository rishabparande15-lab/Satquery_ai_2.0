# Phase 3B - EO Model Landscape Audit

## 1. Scope and research guardrails

This phase is a source-based audit of the current Earth-observation model landscape.
It does not select or implement a learned VLM, install weights, or change the
scientific Pipeline 3. It is intentionally limited to public-source evidence and
repository metadata.

The active project constraints remain in force:

- Fail-closed spatial semantics: BigEarthNet.txt source geometry remains
  preserved but unmapped. No source annotation is treated as an analysis-grid or
  CROMA-token supervision target.
- Scientific representation boundary: `physical_62d`, `joint_croma_gap_768d`, and
  `hybrid_830d` remain scientific features, not universal VLM inputs.
- Local resource envelope: RTX 5060 with 8 GB VRAM and 24 GB RAM.
- No benchmark claim: no VLM score, VQA metric, or grounding score is claimed
  from this repository or this audit.

This is a research-only compatibility review and not a deployment decision.

## 2. Evidence basis and confidence levels

The findings below were derived from official model cards, public repositories,
conference papers, and public project READMEs. The primary evidence set includes:

- EarthMind: public repo and README references, plus the arXiv paper cited there.
- GeoChat: official repo with model weights/demo references.
- EarthDial: public paper and repository references.
- Prithvi-EO-2.0: IBM/NASA HF and project documentation.
- CROMA and DOFA: official project/repo references.
- RemoteCLIP: public repo and weights.
- SkySense V2 and related encoders: public paper evidence.

Confidence is intentionally conservative:

- High: public code or public weights or explicit model card plus task statement.
- Medium: public paper and project repo mention the claim, but local deployment or
  code compatibility is not yet validated in this repository.
- Low: task claim exists in the literature but no verified contract or adapter
  path is established for the current scientific foundation.

## 3. Model families and current verdict

### 3.1 EO encoders and foundation models

| Model | Type | Main task focus | Public code/weights evidence | Compatibility with current project | Verdict |
| --- | --- | --- | --- | --- | --- |
| CROMA | Encoder / multimodal representation model | Scene-level optical+SAR embeddings, token/grid outputs | Public repo and revision metadata available; not a VLM | Useful as scientific representation source, but not a language model and not a direct VLM adapter | Important but not a selected VLM |
| Prithvi-EO-2.0 | Foundation encoder | Temporal multispectral remote sensing encoding | Public project/model card evidence; encoder family, not VLM | Strong foundation encoder, but not a generative grounded vision-language model | Important encoder; not a VLM candidate |
| DOFA / DOFA+ | Foundation encoder | General remote-sensing representation learning | Public model evidence exists; encoder family | Potential representation source only; no current language adapter | Important encoder; not direct language model |
| SkySense V2 | Foundation encoder | Large-scale multisensor remote sensing feature extraction | Public paper and project evidence | Strong encoder but not generative VLM | Encoder only |
| RemoteCLIP | Contrastive VLM | Retrieval / zero-shot classification | Public weights and repo exist | Useful as a contrastive baseline; not a generative VQA/grounding model in the local benchmark stack | Baseline only |

These families are relevant to representation design, but they do not resolve the
core project issue: no verified source-to-grid mapping and no model-specific
language task stack for BigEarthNet.txt, RSVQA, VRSBench, or CDVQA.

### 3.2 Generative EO VLM candidates

| Model | Type | Tasks reported | Public code/weights evidence | Local 8 GB feasibility | Compatibility with current pipeline | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| EarthMind | EO VLM / MLLM | Multi-sensor and multi-granularity optical/SAR reasoning | Public repo with release info, model weights/data/code references; Apache-2.0 in repo | Not proven for 8 GB local deployment; likely heavier than this project envelope | Strongest public EO-VLM candidate, but still requires proven adapter and benchmark contract | Research candidate only |
| GeoChat | EO VLM / grounded MLLM | Region-based visual grounding, captioning, VQA | Public repo and demo/weights references | Likely too heavy for local 8 GB GPU if fully loaded; still more feasible than large multi-sensor VLMs | Strong public grounding baseline for RGB/region tasks, but not aligned with fail-closed BigEarthNet.txt geometry | Candidate for RGB grounding direction, not current project default |
| EarthDial | EO VLM / domain-specific MLLM | Classification, detection, captioning, VQA, grounding, change detection | Public paper exists; repo references also exist | Heavy; project is not aligned to this deployment profile | Broad task coverage but high complexity and no current adapter contract | High-potential but not immediate fit |
| RS-InternVL / InternVL-based EO models | EO VLM family | General remote-sensing visual-language tasks | Public project URLs and model cards exist in some cases; status varies by exact model | Usually heavy and not local-friendly for 8 GB VRAM | Potentially relevant but not yet proven for the project’s current contract stack | Needs explicit adapter and benchmark proof |
| SkyEyeGPT | EO VLM / MLLM candidate | VLM-style reasoning around remote sensing | Some literature references exist, but evidence is mixed and some URLs were invalid or mismatched | Not yet locally feasible or benchmark-proven | Not validated for current contract path | No approval |

## 4. Detailed findings by candidate

### EarthMind

EarthMind is currently the strongest public EO-VLM candidate in this audit.
Evidence summary:

- Public repo is present and indicates a release with code, data, model weights,
  training/evaluation references, and an Apache-2.0 license.
- The repository cites the arXiv paper `2506.01667`.
- The model is described as a multi-sensor, multi-granularity vision-language
  system for Earth observation, including optical and SAR-related reasoning.

The important caveat is that this remains a research candidate, not an approved
adapter for this repository. The current project still requires:

- a verified input contract for optical/SAR data;
- a clear mapping from model inputs to the existing representation layer;
- a benchmark standard that matches the project’s source-preservation rules;
- local runtime feasibility under the 8 GB VRAM requirement.

Verdict: high-interest research candidate, but no selection yet.

### GeoChat

GeoChat has strong public evidence as a remote-sensing VLM with grounding and
region-aware multimodal interactions.

Evidence summary:

- Public repository is available.
- README references model weights and demo availability.
- It is strong on region captioning and grounding tasks.

The key issue is that it is not yet proven compatible with the current fail-closed
spatial policy. The project cannot accept a learned grounding model that assumes
source boxes, pixels, or token coordinates are already authoritative when the
BigEarthNet.txt source geometry remains unmapped and unresolved.

Verdict: promising RGB/grounding baseline, but not approved for use in the
current scientific stack without a separate, validated grounding contract.

### EarthDial

EarthDial is a broad remote-sensing VLM with reported support for multi-spectral,
multi-temporal, multi-resolution imagery and tasks including classification,
detection, captioning, VQA, grounding, and change detection.

Evidence summary:

- Public paper exists.
- Public repo references are present.
- Training and model design appear large-scale and resource-intensive.

The project position is that this is a strong research direction but not a
lightweight local match. It exceeds the local deployment plan and would require
explicit contract-level evaluation before adoption.

Verdict: promising but not local-resource aligned and not yet adoptable.

### RemoteCLIP

RemoteCLIP is relevant as a contrastive remote-sensing VLM, but it is not a
replacement for the current generative or grounded task stack.

Evidence summary:

- Public weight/model evidence exists.
- It is useful for zero-shot classification and retrieval-style tasks.

The project does not treat it as a VQA, captioning, or grounding model.

Verdict: baseline/representation reference only; not a project-ready language
model.

### Prithvi-EO-2.0, CROMA, DOFA, SkySense V2

These are essential representation-layer references, not direct VLM candidates.
They are important because they show that the remote-sensing ecosystem has strong
encoders and multi-sensor feature extractors. However, they do not satisfy the
project’s requirement for a proven visual-language reasoning stack or a validated
benchmark contract.

Verdict: essential scientific references, not decision-ready VLMs.

## 5. Compatibility matrix against this project

| Dimension | EarthMind | GeoChat | EarthDial | CROMA / Prithvi / DOFA / SkySense | RemoteCLIP |
| --- | --- | --- | --- | --- | --- |
| Public code/weights evidence | High | High | Medium | High for encoder families | High |
| Optical-SAR reasoning | High | Low / mostly RGB | High | High for encoders | Medium |
| Grounded region understanding | Medium / likely | High | High | Not direct VLM | Low |
| Temporal reasoning | Possible | Limited | High | High for temporal encoders | Low |
| Benchmark compatibility with local stack | Unknown | Unknown | Unknown | Not direct benchmark stack | Not direct |
| Local 8 GB fit | Unclear / likely heavy | Unclear but likely heavy | Poor fit | Usually encoder-focused and more feasible in isolation | Moderate but not a VQA stack |
| Alignment with fail-closed spatial policy | Not yet proven | Not yet proven | Not yet proven | Not a language model | Not a project fit |
| Direct use as current repository adapter | No | No | No | No | No |
| Research-only recommendation | Yes | Yes | Yes | Yes | Yes as baseline only |

## 6. Recommendations

### Recommended direction

The strongest current research-only recommendation is:

1. Keep the existing scientific baseline and invariants unchanged.
2. Treat EarthMind as the strongest public EO-VLM candidate for future research
   review, but only after explicit model-specific adapter and benchmark contracts
   are proven.
3. Treat GeoChat as the strongest public grounded RGB baseline for future
   region-aware research, but not as a direct replacement for the current source
   geometry policy.
4. Keep EarthDial in the long-list as a heavy and broad task model, but not a
   near-term deployment candidate.
5. Treat CROMA, Prithvi-EO-2.0, DOFA, SkySense V2, and RemoteCLIP as
   representation and baseline references rather than production VLM paths.

### Explicit non-selection

No model is selected for implementation in this repository at this stage.
No project code, benchmark, or adapter is approved to consume BigEarthNet.txt
source geometry or the current scientific pipeline features as a universal VLM
input.

This remains a research-only audit and not a scientific deployment.

## 7. Final decision

The landscape audit does not support immediate VLM selection. The evidence is
strong enough to say that the ecosystem contains multiple viable EO-VLM and
foundation-model research directions, but it is not strong enough to justify a
project integration or benchmark claim under the current fail-closed geometry and
scientific-preservation requirements.

The current decision remains:

- No benchmark-grade VLM is selected.
- No learned model is installed.
- No benchmark score is claimed.
- No source geometry is mapped to the analysis grid or CROMA tokens.
- Future model work remains contingent on explicit contracts and evidence.

## 8. Source references used in this audit

- EarthMind public repository and paper references: public repo README and arXiv
  paper cited in the project README.
- GeoChat public repository: public project repo with weights/demo references.
- EarthDial public paper and repo references: public arXiv paper and linked repo.
- Prithvi-EO-2.0 official project / model card references: IBM/NASA HF and
  project documentation.
- CROMA public repository and revision metadata: official project references.
- RemoteCLIP public weights and model repo.
- SkySense V2 public paper references.

All of these were reviewed as public-source evidence only. No local benchmark or
model deployment was performed in this phase.
