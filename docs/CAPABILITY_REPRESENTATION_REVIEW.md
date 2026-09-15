# Capability and Representation Architecture Review

## 1. Executive summary

Pipeline 3 remains the sole production core and `src.analysis_engine.run_analysis` remains its sole analysis entry. This review changes no runtime, model, dataset, preprocessing, evidence, interpretation, API, or contract code.

The Phase 6A.3 contracts are sufficient to exercise the first end-to-end capability boundary, provided that the first capability is the **existing deterministic Pipeline 3 scene analysis**. They are intentionally permissive and are not yet sufficient to guarantee the stronger inputs, model identity, output typing, evidence attachment, or temporal pairing required by learned VQA, captioning, grounding, or change prediction.

The current system has scientifically useful representations, but their meanings differ:

- physical statistics and spatial feature maps are deterministic and interpretable;
- optical, SAR, and joint CROMA tensors are pretrained latent representations, not task answers;
- the 2,304-value pooled vector is concatenated optical/SAR/joint CROMA output;
- the 192-value `HybridFusion` result uses seeded but **untrained** projection weights and is representation-only;
- spatial evidence is deterministic, traceable, and descriptive;
- no live task prediction or calibrated confidence exists for sample `61_39`.

The first executable capability should wrap the current feature/evidence/interpretation behavior. This validates `TaskRequest -> CapabilityRegistry -> SceneBundle -> Pipeline 3 -> TaskResult -> Evidence + Provenance` without adding a model and dataset at the same time.

## 2. Phase 6A.3 contract review

### SceneBundle

Strengths:

- One immutable scene identity and input-identifier boundary.
- Optical-only, SAR-only, joint, and absent future representations are valid.
- CRS, bounds, resolution, dimensions, georeferencing, and co-registration have explicit locations.
- Existing spatial evidence and provenance can pass through without translation.

Limitations:

- `temporal_context` is an open mapping, not a validated T1/T2 contract.
- `RepresentationSet` fields are `Any`; shape, dtype, artifact identity, model/checkpoint, spatial grid, and validation status are not enforceable.
- Raw modality arrays have no explicit representation slots. They can be referenced through generic fields/provenance, but that is ambiguous.
- `croma_scene` currently receives the legacy `features.deep` object. For a joint run that object contains a concatenated 2,304-value vector and shape metadata, not one uniquely defined scene embedding.
- The live adapter does not populate optical/SAR token representations because `run_analysis` retains only their shapes and pooled values in the public result.

Conclusion: sufficient for scene-level compatibility; partially ready for learned or token-level capabilities.

### RepresentationSet

The field set anticipates the right categories, but semantics are underspecified. Before a capability requires a representation, add a versioned `RepresentationDescriptor` or equivalent reference containing at least representation kind, modality, shape, dtype, spatial grid, semantic status, learned/deterministic status, validation status, artifact reference/hash, producer/model version, and parent inputs. This should be additive; the current fields need not be removed.

### TaskRequest

Strengths:

- Generic task type, scene reference, query, modalities, temporal context, parameters, constraints, and execution metadata.
- Detectable invalid VQA, optical/SAR, and temporal combinations fail early.

Limitations:

- No request identifier, requested output types, representation requirements, evidence requirements, or accepted contract versions.
- Only a small hard-coded set of task/modality constraints is structural.
- Open parameter and constraint mappings defer semantic validation to a future capability.

Conclusion: adequate for the first bounded capability; add request/output/representation requirements before multiple heterogeneous capabilities are registered.

### TaskResult

Strengths:

- Distinguishes success, partial, failure, and unavailable.
- Evidence can be absent.
- Numeric confidence is rejected unless explicitly declared calibrated.
- Existing output, artifacts, diagnostics, and provenance can pass through unchanged.

Limitations:

- `output` is untyped and there is no `output_type`, result identifier, limitations field, or per-output evidence relation.
- Confidence is one scalar rather than a versioned record tied to a model, calibration dataset, target, and calibration metric.
- `evidence` is one mapping rather than typed references between individual outputs/claims and supporting items.

Conclusion: safe for a compatibility result with `confidence=None`; insufficient for strong learned-output guarantees without additive descriptors.

### Capability and CapabilityRegistry

Strengths:

- Declarative metadata, deterministic listing, duplicate rejection, and task/modality/input/temporal matching.
- No dynamic code execution or planner coupling.

Limitations:

- A capability declares supported modalities but cannot distinguish required, optional, or alternative modality sets.
- It cannot declare required representations, evidence schema versions, accepted contract versions, output types, model/checkpoint identity, calibration artifacts, resource limits, or explicit limitations.
- Matching cannot yet reject a scene that has the modality but lacks the representation required by the implementation.
- One capability describes one task type; aliases or multiple task types require multiple entries.

Conclusion: sufficient to register one scene-analysis capability. Before VQA or temporal work, add capability requirements and representation-aware matching. No Phase 6A.3 code change is blocking now.

## 3. Current Pipeline 3 representation inventory

Dimensions below come from implementation, tests, and the verified `61_39` receipts. `N=1` in the live single-scene path.

| Representation | Source and exact shape | Modality / spatial meaning | Semantic and scientific status | Persistence / cost | Plausible reuse |
|---|---|---|---|---|---|
| Raw optical tensor | Raster assembly; `[12,120,120]`, float32 | Sentinel-2 canonical band order; 10 m common grid for `61_39` | Measured input after grid assembly; deterministic, not learned | Ephemeral live; persisted in final-audit artifact. Raster I/O/resampling cost not separately benchmarked | Physical analysis, CROMA, future optical models, visualization |
| Raw SAR tensor | Raster assembly; `[2,120,120]`, float32 | Sentinel-1 `VV,VH`; co-registered grid | Measured input; deterministic, not learned | Ephemeral live; persisted in final-audit artifact | SAR statistics, CROMA, optical-SAR analysis, future SAR models |
| CROMA-normalized inputs | `_croma_normalize`; `[1,12,120,120]` or `[1,2,120,120]` | Per-channel mean +/- 2 standard deviations, clipped to `[0,1]` | Deterministic preprocessing, not a task representation | Ephemeral; CROMA dominates live compute and records per-run runtime | CROMA encoders only |
| Optical CROMA tokens | Official pretrained CROMA; `[N,225,768]` | Optical latent token for each row-major 15x15 grid cell (8x8 pixels at 120x120) | Learned latent; dimensions have no assigned semantics. Shape/repeatability validated | Live report stores shape only; `[225,768]` array persisted in final-audit artifact | Classification heads, grounding research, future question-conditioned models |
| SAR CROMA tokens | Official pretrained CROMA; `[N,225,768]` | SAR latent tokens on the same token grid | Learned latent; not a prediction | Shape live; array persisted in final-audit artifact | SAR classification, multimodal reasoning, grounding research |
| Joint CROMA tokens | Official joint CROMA call; `[N,225,768]` | Pretrained joint optical-SAR latent tokens | Learned multimodal representation, but not a task-specific semantic output | Shape live; array persisted in final-audit artifact | Joint classification and future learned multimodal heads |
| Optical/SAR/joint GAP | Official CROMA; three `[N,768]` tensors | Scene-level latent vectors per view | Learned pooled representations; no class semantics by themselves | Final-audit artifact persists each `[768]`; live joint response combines them | Scene-level heads and comparisons |
| Pooled CROMA vector | `pooled_croma_features`; `[2304]` for a joint live scene | Concatenation of optical, SAR, and joint GAP vectors | Deterministic concatenation of learned latents; not new learned fusion | Values persisted in live report and final-audit artifact | Existing hybrid projection, scene-level task heads |
| Physical feature vector | `modality_features.extract`; joint `[62]`, optical `[52]`, SAR `[9]` | Scene statistics and indices | Deterministic, interpretable. Joint schema is `legacy_joint_62`; documented clipped legacy SAR relationship limitation | Values persisted in live report; `[62]` final-audit artifact. Cost not separately benchmarked | Numerical analysis, evidence, conventional task heads |
| Seeded hybrid vector | `HybridFusion(62,2304)`; `[192]` | Scene-level projection of physical and pooled CROMA branches | Neural projection and concatenation with seeded **untrained** weights; representation-only, no prediction validity | Values persisted in live report and final-audit artifact | Architecture experiments only until a trained/evaluated task use exists |
| Pixel/local feature maps | `pixel_features`; each `[120,120]` | Optical indices/bands, SAR values/difference, local standard deviation, edge strength | Deterministic measured/derived spatial features | Seventeen maps plus mask persisted in pixel-feature probe; live evidence computes an 11-feature subset. Cost not separately benchmarked | Numerical maps, region proposals, evidence |
| Token physical summaries | `aggregate_tokens`; each selected map `[225,8]` | Mean, standard deviation, min, max, p25, median, p75, valid ratio per 8x8 block | Deterministic token-aligned statistics | Seventeen `[225,8]` arrays persisted in pixel probe; live evidence embeds selected mean records | Grounded numerical analysis, feature/model comparisons |
| Spatial evidence tokens | `spatial_evidence_v1`; 225 objects | Pixel bounds, projected bounds, grid position, selected feature records, latent refs, optional predictions | Deterministic and validated; CROMA remains `LATENT`, predictions are `UNAVAILABLE` without a head | Complete tokens persisted under `artifacts/spatial_evidence/61_39`; public live response currently omits token list | Evidence queries, future grounding/evidence reconciliation |
| Evidence regions | Four-connected threshold grouping; variable count, 17 for `61_39` | Groups of token indices with projected extents | Deterministic evidence regions, explicitly not learned segmentation | Persisted in spatial artifact and live report | Constrained interpretation, region references, future proposal baselines |
| Evidence claims | Three for `61_39` | Vegetation, water, and SAR polarization-contrast evidence | Inferred from derived features; descriptive strength, not prediction/confidence | Persisted and exposed by API | Current scene analysis, evidence-aware future tasks |
| Geospatial/scene metadata | Per modality: CRS, affine, bounds, resolution, shape, acquisition date, bands, availability | Establishes spatial identity and co-registration | Deterministic metadata validation | Persisted in live report and artifacts | Every spatial or temporal capability |
| Constrained interpretation | `constrained_interpretation_v1`; structured answer plus claims | Text rendering of allowlisted evidence | Deterministic evidence presentation, not arbitrary VQA or captioning | Persisted in live report and interpretation artifacts; measured median 0.390 ms, p95 0.926 ms in a 200-run adapter benchmark | First scene-analysis capability only within allowlist |

Evidence construction plus interpretation measured 0.1402 seconds in the documented `61_39` API run. CROMA and total runtime are environment/device dependent and recorded per execution; the repository does not establish one portable latency guarantee.

## 4. Capability-representation matrix

Readiness means architectural and scientific readiness in the current repository, not theoretical feasibility.

| Capability | Inputs / representations | Output | Trustworthy evidence and confidence now | Provenance | Readiness |
|---|---|---|---|---|---|
| A. Deterministic scene analysis | Optical and/or SAR; physical features; optional pooled CROMA; regions and existing evidence; metadata | Structured features and constrained text | Pixel/token/region/claim evidence available; descriptive strength only; confidence `None` | Existing chain is sufficient | **READY** within current allowlist |
| A2. Learned scene classification | Optical, SAR, or joint CROMA scene/tokens; trained head; metadata | Class/coverage scores | Evaluated research heads exist, but live `61_39` has no task prediction evidence and no calibrated confidence | Research provenance is strong; live adapter linkage is incomplete | **PARTIALLY READY** |
| B. Single-image VQA | Raw view and/or CROMA tokens; question representation; question-conditioned fusion; metadata | Text or structured answer | Current evidence cannot validate arbitrary answer claims; confidence unavailable | Model/question/answer provenance fields are not enforced | **NOT READY** |
| C. Scene captioning | Raw view or tokens plus learned language decoder | Free text or structured caption | No caption model, caption evidence policy, or benchmark; current claims cover only an allowlist | Base provenance can carry refs but model linkage is absent | **NOT READY** |
| D. Grounding | Raw imagery, token embeddings, grid metadata, learned/query-conditioned representation | Token set, bbox, mask, or polygon | Deterministic regions exist but are not learned grounding; no grounded task evaluation | Spatial provenance is strong; task/model relation missing | **PARTIALLY READY** as evidence infrastructure; **NOT READY** as learned capability |
| E. Optical-SAR analysis | Separate optical/SAR physical and CROMA views; joint CROMA; metadata | Structured comparison, features, or later learned output | Sensor-separated current evidence is trustworthy; no joint semantic claim or calibrated confidence | Existing provenance preserves sensor/source | **PARTIALLY READY**; deterministic subset is ready |
| F. Temporal/change analysis | Explicit T1/T2 scenes, matching modality representations, acquisition/grid/co-registration metadata | Structured deltas, score, change map, regions | Pair validation and vector distances exist; no validated change prediction/evidence/confidence | Base provenance can hold parents but temporal lineage is untyped | **NOT READY** as production prediction |
| G. Change VQA | Everything required by VQA and temporal/change | Text plus temporal/spatial evidence | Neither parent capability has sufficient evidence controls or evaluation | Requires two-scene, change-model, question, and answer lineage | **NOT READY** |
| H. Numerical/physical analysis | Raw optical/SAR, deterministic maps/statistics, geospatial metadata | Values, units, summaries, possibly mapped regions | Strongest current evidence: pixel/token/region/source; no probability needed | Existing provenance is sufficient | **READY** within implemented formulas |
| I. Future extensible capability | Declared by capability-specific requirements | Typed task-specific output | Depends on registered evidence and calibration contract | Open mappings can carry it, but enforcement must be added | **PARTIALLY READY** as architecture only |

## 5. Representation bottleneck

CROMA extraction solves representation learning, not downstream tasks. A 768-dimensional token or scene vector has no automatic mapping to a class, phrase, answer, location, or change. Each such mapping requires a task-specific trained and evaluated component.

- Semantic VQA requires question encoding, question-conditioned interaction with image tokens, an answer decoder/head, unsupported-question behavior, and evidence linking the answer to spatial/model outputs.
- Captioning requires a trained language decoder and a claim policy. The constrained interpretation adapter cannot be relabelled as a caption model.
- Grounding requires a learned or otherwise validated query-to-location mapping. Current regions are deterministic feature-threshold groups, not object masks.
- Optical-SAR reasoning may reuse separate and joint CROMA views, but task-specific claims still require a validated head or constrained deterministic rule.
- Temporal reasoning requires paired representations with explicit T1/T2 lineage and a validated change operator/head; vector subtraction alone is not a semantic change model.

The immediate storage bottleneck is that live `run_analysis` returns pooled CROMA values and token shapes, not live token arrays or immutable token artifact references. That is acceptable for scene analysis but blocks trustworthy token-consuming capabilities.

## 6. Multimodal fusion readiness

There are three distinct mechanisms:

1. **Official CROMA joint output:** passing real optical and SAR tensors to pretrained CROMA produces `joint_encodings [N,225,768]` and `joint_GAP [N,768]`. This is a learned multimodal latent representation. The repository does not turn it into a semantic task output.
2. **Pooled combination:** `pooled_croma_features` concatenates optical, SAR, and joint GAP vectors into 2,304 values. Concatenation itself is not additional learned fusion.
3. **Physical/CROMA hybrid:** `HybridFusion` separately projects 62 physical and 2,304 CROMA values, concatenates the projections, then applies an output projection to 192 values. In the live runtime its seeded weights are untrained and explicitly representation-only.

At evidence level, optical and SAR claims remain separate and the interpretation says they are complementary. No joint semantic claim is manufactured.

Independent modality vectors are suitable for sensor-specific tasks and ablations. Existing joint CROMA can support a future trained scene head. Question-conditioned cross-attention or another evaluated fusion is required for semantic multimodal VQA. Physical-feature fusion is appropriate when formulas are relevant and evaluated. Evidence-level fusion should reconcile already-supported claims; it must not promote co-occurrence into a new learned claim.

## 7. Temporal readiness

Current infrastructure:

- deterministic parsing of temporal intent;
- before/after file requirements and grid checks;
- `TemporalRequest` date/range contracts in Phase 3 orchestration;
- `validate_temporal_pair` for identities, ordered dates, CRS/grid correspondence;
- `compare_representations` for L2 and mean absolute difference;
- an untrained `ChangeAnalysisHead` that returns `after - before`.

Production behavior is deliberately unavailable: the API validates a pair but emits no change map, area, prediction, or confidence. Historical/orchestration utilities are not a validated production change capability.

`SceneBundle.temporal_context` can carry T1/T2 data conceptually, but it does not validate two scene identities, acquisition timestamps, modalities, parent artifact hashes, co-registration method/residuals, or derived representation lineage. Before temporal implementation, introduce a typed two-scene temporal descriptor and temporal evidence references. Current optical/SAR/physical/CROMA representations can be reused per timepoint only after preprocessing equivalence and co-registration are explicit.

## 8. Evidence readiness

The current trusted path is:

```text
input pixels -> deterministic feature -> token -> region -> claim
             -> source artifact/version -> evidence provenance -> interpretation
```

- Scene and numerical analysis can reuse it directly.
- Learned classification must attach each predicted output to its head/checkpoint and the spatial or scene representation used. Reference labels cannot substitute for predictions.
- Grounding must distinguish deterministic proposal regions from learned query grounding.
- Temporal results must cite both parent scenes, the alignment operation, changed values/regions, and the change model or deterministic formula.
- VQA needs answer-to-claim relations. Every factual answer unit must either cite supported measured/derived/predicted evidence or be marked unsupported/unavailable. Fluent text is never evidence.

The existing evidence provenance can be embedded in future results, but arbitrary outputs need typed evidence references and a reconciliation validator. Until that exists, VQA answers cannot be made trustworthy.

## 9. Provenance readiness

Current artifacts already carry input/sample identity, source files and hashes, bands, CRS/grid, preprocessing profile, physical schema, CROMA checkpoint details in scientific runs, representation shapes/hashes, evidence versions, execution identifiers, and interpretation provenance. The compatibility adapter shares one provenance mapping between `SceneBundle` and `TaskResult`.

Gaps are enforcement rather than storage: open mappings do not require a live checkpoint hash, task adapter version, calibration artifact, output-to-evidence link, or temporal parent graph. Add requirements at capability registration/artifact creation rather than reconstructing them at final rendering.

## 10. Confidence readiness

These concepts remain separate:

| Concept | Current status |
|---|---|
| Model confidence / class score | No live trained task head for `61_39`; unavailable |
| Probability | A sigmoid/softmax score in research code is not automatically calibrated |
| Calibrated confidence | No current live calibration artifact; canonical value is `None` |
| Heuristic score | Representation distances or thresholds may be reported with their formula, never as confidence |
| Evidence strength | `STRONG/MODERATE/WEAK/UNAVAILABLE` is descriptive threshold support, not probability |
| Linguistic certainty | Presentation style has no scientific confidence meaning |

`TaskResult` correctly rejects non-null numeric confidence unless marked calibrated. Before a learned capability uses confidence, a richer record should identify target, model/checkpoint, calibration method/dataset/version, metric, and applicability limits.

## 11. Capability ranking

Scores are 0–5 for: scientific maturity (M), existing validated implementation (I), architecture-validation value (A), representation reuse (R), evidence compatibility (E), provenance compatibility (P), confidence compatibility (C), implementation risk where 5 is lowest risk (K), evaluation availability (V), and extensibility value (X). Total is out of 50.

| Rank | Capability | M | I | A | R | E | P | C | K | V | X | Total |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Deterministic scene analysis | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 5 | **49** |
| 2 | Numerical/physical analysis | 5 | 5 | 4 | 5 | 5 | 5 | 5 | 5 | 4 | 4 | **47** |
| 3 | Deterministic optical-SAR analysis | 4 | 4 | 4 | 5 | 4 | 5 | 5 | 4 | 3 | 5 | **43** |
| 4 | Learned scene classification | 4 | 4 | 5 | 5 | 3 | 4 | 2 | 3 | 5 | 5 | **40** |
| 5 | Learned grounding | 2 | 1 | 4 | 4 | 2 | 3 | 1 | 1 | 2 | 4 | **24** |
| 6 | Temporal/change analysis | 2 | 1 | 5 | 3 | 1 | 3 | 1 | 1 | 2 | 5 | **24** |
| 7 | Single-image VQA | 1 | 0 | 5 | 3 | 1 | 2 | 1 | 1 | 1 | 5 | **20** |
| 8 | Scene captioning | 1 | 0 | 3 | 3 | 1 | 2 | 1 | 1 | 1 | 3 | **16** |
| 9 | Change VQA | 0 | 0 | 5 | 2 | 0 | 1 | 0 | 0 | 0 | 5 | **13** |

The ranking favors a capability that validates the contracts with already-validated science. It does not choose the smallest coding task; it chooses the smallest increase in scientific uncertainty.

## 12. Recommended first executable capability

### Identity

- Capability name: `pipeline3_scene_analysis`
- Task type: `scene_analysis`
- Version: `1.0.0`
- Nature: deterministic compatibility capability over current Pipeline 3; not learned classification

### Required SceneBundle fields

- `scene_id`, `analysis_id`, `input_type= satellite_scene`
- at least one of `optical` or `sar` in `available_modalities`
- dimensions and validated georeferencing for spatial evidence
- input identifiers and provenance run/source information
- `representations.physical_features` with computed status

### Required and optional representations

Required: current physical feature report. Optional: pooled CROMA/deep report, untrained hybrid report labelled representation-only, spatial regions, spatial evidence, and constrained interpretation. The capability must degrade to `partial` when CROMA is unavailable but physical features are valid.

### TaskRequest

```text
task_type: scene_analysis
scene_id: required
query: optional, limited to the existing constrained interpretation surface
requested_modalities: non-empty subset of optical, sar
parameters:
  interpretation_mode: simple | technical | none
constraints:
  require_spatial_evidence: boolean, default false
execution_metadata: request/run correlation only
```

Unknown parameters, temporal context, unsupported modalities, arbitrary VQA intent, or a query outside the constrained interpretation contract must be rejected or return unavailable—not silently reinterpreted.

### TaskResult

```text
status: success | partial | failure | unavailable
task_type: scene_analysis
output:
  modalities_processed
  physical_feature_summary
  croma_representation_status
  hybrid_representation_status
  constrained_interpretation (optional)
evidence: exact spatial_evidence_v1 object or None
confidence: None
confidence_calibrated: false
provenance: shared Pipeline 3 provenance
artifacts: existing input/feature/evidence references
diagnostics: validation, warnings, unavailable reasons, legacy status
```

### Evidence, provenance, failures, and artifacts

- Claims may only be the existing validated allowlisted claims with their exact sensor, token, region, feature, source artifact, and descriptive strength.
- No CROMA dimension may receive a semantic label.
- Confidence is always `None`; evidence strength remains descriptive.
- Provenance must include run ID, input identifiers/source, preprocessing steps, physical schema, CROMA source when used, evidence provenance, task name/version, and execution trace.
- Fail on malformed bundle/request, unavailable required modality, invalid geospatial metadata, failed physical computation, non-finite representations, or contract mismatch.
- Return `partial` when physical analysis succeeds but optional CROMA/evidence does not.
- Expected artifacts are references to the unchanged legacy report, feature statistics, and spatial evidence when available. Large arrays remain references, not duplicated JSON.

### Evaluation strategy

1. Golden adapter tests prove physical/CROMA/hybrid values, evidence, interpretation, status, warnings, and provenance are unchanged.
2. Optical-only, SAR-only, joint, missing-CROMA, malformed-grid, unsupported-query, and no-evidence cases.
3. Capability registry match/rejection tests.
4. Exact normalized scientific fingerprint and `61_39` API/report round-trip.
5. No new accuracy metric: this capability exposes existing analysis and does not claim classification.

New scientific assumptions: **none** beyond the already documented deterministic feature/evidence thresholds and pretrained latent-representation status.

## 13. Requirements before VQA

Before implementing VQA:

1. Add immutable artifact descriptors for raw views, separate CROMA scene vectors, and token arrays; include shapes, grids, hashes, producer/checkpoint, and validation state.
2. Make live token representations available by bounded artifact reference rather than embedding large arrays in API JSON.
3. Define supported question classes and an explicit unsupported-question result.
4. Select and evaluate question encoding and question-conditioned visual/multimodal fusion. CROMA alone is not VQA.
5. Define answer output types and answer-to-claim/evidence links.
6. Add a validator that prevents unsupported answer claims and distinguishes predicted answers from derived measurements and constrained interpretation.
7. Decide optical-only, SAR-only, and joint policies; never manufacture a missing modality.
8. Define spatial grounding requirements for questions whose answers imply location, count, extent, or object identity.
9. Keep confidence unavailable unless a calibration study produces a versioned artifact.
10. Extend capability metadata with required representations, evidence versions, model/checkpoint identity, and limitations.
11. Implement a benchmark adapter and immutable split/provenance contract for RSVQA or another deliberately selected dataset; prevent train/test geographic leakage and label-vocabulary drift.
12. Evaluate answer accuracy, unsupported-question rejection, evidence faithfulness, grounding where applicable, modality ablations, calibration, and failure behavior before API exposure.

## 14. Risks

- Treating latent CROMA features as semantic answers.
- Treating the seeded untrained hybrid vector as a trained multimodal model.
- Registering capabilities based only on modality while required artifacts are absent.
- Returning fluent text without evidence support.
- Confusing evidence strength, raw model score, probability, and calibrated confidence.
- Duplicating large token arrays in contracts/API instead of using immutable references.
- Divergence between research artifacts and the live `run_analysis` path.
- Temporal claims without explicit two-scene identity and co-registration lineage.
- Reusing reference labels as predictions.
- Breaking legacy API/report consumers while exposing new contracts.

## 15. Architectural decisions

1. Pipeline 3 and `run_analysis` remain canonical.
2. No Phase 6A.3 contract modification is required in this review.
3. Representation and capability descriptors should be strengthened additively before token-dependent or temporal capabilities.
4. The first capability is deterministic scene analysis, not learned classification or VQA.
5. Existing `spatial_evidence_v1` and constrained interpretation remain the only trusted semantic pathway today.
6. Confidence remains unavailable unless tied to an explicit calibration artifact.
7. Learned classification may follow after a live model/evidence/calibration decision; VQA requires broader shared infrastructure first.

## Final recommendation

**THE FIRST EXECUTABLE CAPABILITY SHOULD BE: PIPELINE 3 DETERMINISTIC SCENE ANALYSIS.**

It should be first because its scientific behavior, evidence, provenance, failure modes, and regression fixtures already exist. It reuses input validation, physical features, optional CROMA representations, deterministic spatial evidence, constrained interpretation, the compatibility adapter, and immutable reports. It introduces no new model or dataset and no new accuracy or confidence claim. It validates every common architecture boundary before model development begins. After it is implemented and evaluated, the team should add representation-aware capability requirements and select either bounded numerical analysis or a production model/evaluation plan for learned scene classification; VQA should wait for the requirements above.
