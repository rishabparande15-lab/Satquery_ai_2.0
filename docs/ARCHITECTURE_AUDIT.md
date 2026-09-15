# SatQuery AI Architecture Audit

Date: 2026-09-13

Status: forensic documentation only. This document records the implemented system and proposes boundaries for future work. It does not change scientific behavior, model weights, datasets, splits, prediction logic, API routes, or frontend behavior.

> Current-status note (2026-09-15): this forensic snapshot predates the implemented architecture contracts, deterministic capability wrapper, optional representation persistence, receipt catalog, and completed exact 5,000-area scientific checkpoint. Those additions preserve the runtime boundary audited here. See [Current Project Status](CURRENT_PROJECT_STATUS.md) for the authoritative present state.

## 1. Executive Finding

SatQuery AI currently contains two related but distinct execution paths:

1. The live local web product path, entered through `src/api.py`, runs `src.analysis_engine.run_analysis`. It is a deterministic, fail-closed feature and evidence service. It does not run a trained task head, does not emit a prediction, does not calculate calibrated confidence, and does not implement temporal change inference.
2. The typed Phase 1-3 scientific path, centered on `src/phase3_orchestration.py`, `src/phase1_foundation.py`, and `src/phase2_integration.py`, validates source-independent scene data and packages CROMA, physical features, identities, hashes, and provenance for experiments and verification. It is not the live web controller.

The most important architectural risk is therefore not missing algorithms. It is the boundary between these paths: they have overlapping concepts but different contracts, provenance depth, and runtime ownership. Future task modules should attach through a typed task/representation boundary and should not rewrite the validated web analysis path.

## 2. Current Architecture

### Live product components

- `src/api.py`: loopback HTTP server, request routing, upload registry, size limits, host/origin checks, serialized analysis lock, and report download.
- `src/static/app.js` and `src/static/index.html`: browser UI, form state, upload flow, availability search, local history, report rendering, and explicit unavailable-state messaging.
- `src/analysis_engine.py`: live request validation, query planning, retrieval, input validation, raster assembly, physical features, optional CROMA, deterministic hybrid representation, spatial evidence, constrained interpretation, confidence/status fields, and report payload construction.
- `src/query_interpreter.py`: deterministic `QueryPlan` parser. The LLM provider interface exists, but configured providers intentionally return no provider in the local proof-of-concept.
- `src/data_orchestrator.py`: local sample discovery, user-upload routing, and explicit external-provider-unavailable response.
- `src/input_validation.py`, `src/raster_inputs.py`, `src/multimodal_cube.py`: input and raster contracts.
- `src/modality_features.py`, `src/gee_features.py`: physical feature extraction.
- `src/croma_adapter.py`: dynamic loading of the official CROMA implementation and checkpoint; per-channel normalization and optical/SAR/joint inference.
- `src/hybrid_fusion.py`: deterministic seeded representation fusion. The live path labels it representation-only; it is not a trained prediction model.
- `src/evidence_schema.py`: `spatial_evidence_v1`, 225 row-major tokens, deterministic feature maps, four-connected threshold regions, claims, and provenance.
- `src/interpretation_adapter.py`: `constrained_interpretation_v1`, allowlisted claim routing, deterministic templates, semantic hash, and fail-closed validation.
- `src/tool_selector.py`: availability-aware selection metadata. This is a local capability summary, not an agent controller.
- `src/config.py`: environment-based settings for dataset, feature, CROMA source, and checkpoint paths.

### Scientific and experiment components

- `src/phase1_foundation.py`: typed identities, normalization, prepared batches, CROMA shape validation, frozen CROMA foundation, label targets, and evaluation metrics.
- `src/phase2_integration.py`: typed bridge from Phase 1 output into the existing advanced pipeline, including checkpoint hash and source hashes.
- `src/phase3_orchestration.py`: typed AOI, temporal, source, resolution, scene selection, acquisition, raster, spatial-pair, and processing contracts.
- `src/scientific_artifacts.py`: hashed artifact bundles and receipts.
- `src/feature_cache.py`, `src/prepare_training.py`, `src/train_landcover.py`, `src/evaluate_landcover.py`, and `src/hybrid_pipeline.py`: experiment/training infrastructure, not live API services.
- `experiments/` and `docs/`: reproducible evaluation, leakage, evidence, and Pass 5D records.

## 3. Actual Runtime Data Flow

The implemented web path is:

`browser input -> HTTP origin/body validation -> request validation -> deterministic query plan -> data retrieval selection -> raster/input validation -> raster assembly -> multimodal cube summary -> physical features -> optional CROMA -> optional representation-only fusion -> spatial evidence -> constrained interpretation -> report persistence -> JSON response -> frontend rendering`

### Step-by-step trace

1. `src/static/app.js` collects a query, mode, AOI, dates, local sample or uploaded GeoTIFFs. Uploads go to `/api/upload`; analysis goes to `/api/analyze`.
2. `src/api.py` enforces local host/origin, JSON or multipart content type, body limits, upload role restrictions, temporary upload ownership, and the one-analysis lock.
3. `/api/analyze` calls `validate_request` and then `run_analysis`. The request is rejected for malformed fields, unsupported modes, invalid file-token maps, or unavailable cloud filtering.
4. `interpret_query` creates a deterministic `QueryPlan`: task, modality, dates, datasets, temporal requirement, required tools, expected outputs, and constraints.
5. `DataOrchestrator.retrieve` selects user files, a local development sample, or reports that an external provider is required. It does not download imagery or call GEE.
6. `validate_inputs` checks AOI/date/source/raster requirements. `assemble` loads canonical optical and/or SAR arrays and metadata. Local bands may be aligned to a reference grid; uploaded grids are checked.
7. `MultimodalCube.summary()` records the assembled data shape and metadata. It is a summary object, not the shared representation contract needed by future tasks.
8. `modality_features.extract` computes optical-only, SAR-only, or the legacy joint physical schema. The joint schema is a fixed feature vector; it is not a learned semantic cross-modal prediction.
9. If the official source and checkpoint exist and the raster is 120x120, `_deep_features` reuses a process-local `CROMAAdapter` keyed by source, checkpoint path, checkpoint mtime, and device. It computes CROMA encodings/GAP values and, for paired inputs, the seeded untrained `HybridFusion` representation.
10. The live path calls `build_spatial_evidence` with aligned raw arrays and raster metadata. It currently does not pass the computed live CROMA outputs or task predictions into that call, so the live spatial evidence can contain deterministic physical token evidence but does not claim CROMA-backed or task-predicted token semantics.
11. `interpret_evidence` accepts only allowlisted evidence claims and emits deterministic text with feature, token, region, and artifact traceability. It never calls an LLM/VLM.
12. `run_analysis` constructs a JSON report with `schema_version: "2.0"`, status, validation, preprocessing, feature state, model-result state, evidence, confidence status, temporal state, execution trace, and an explicit no-prediction explanation.
13. `save_report` writes an immutable JSON file under `experiments/outputs/web_reports`. `/api/report/{id}` returns that persisted report. The frontend renders it and stores only lightweight report history in browser localStorage.

## 4. Existing Reusable Components

Already implemented and suitable for reuse behind stable adapters:

- Canonical optical bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12.
- Canonical SAR bands: VV, VH.
- AOI, date, source, coverage, CRS, transform, shape, finite-value, and optical/SAR grid validation.
- Per-channel CROMA normalization and official pretrained CROMA loading.
- CROMA shape contracts: 225 spatial tokens and 768-dimensional pooled vectors in the typed foundation.
- Optical, SAR, and legacy joint physical feature extraction with explicit schema names and limitations.
- Deterministic 15x15 token mapping, feature aggregation, region grouping, evidence claims, and evidence validation.
- Allowlisted interpretation with explicit inferred-from-derived-evidence epistemic status and uncalibrated confidence wording.
- Typed scientific identity, split, normalization, source hash, checkpoint hash, and artifact receipt patterns.
- Immutable report persistence and bounded temporary upload lifecycle.
- Strong regression tests for validation, evidence, interpretation, API failures, upload cleanup, and scientific contracts.

## 5. Bottlenecks and Readiness

### A. Representation bottlenecks

CROMA is produced in `analysis_engine._deep_features` for the live path and by `FrozenCromaFoundation`/the Phase 2 adapter in the typed scientific path. The typed path already exposes separate optical, SAR, and joint encodings plus GAP vectors. The live report currently serializes pooled values and shapes but does not expose a reusable typed representation artifact containing all encodings, normalization metadata, checkpoint hash, and source hashes.

The shared artifact that should become canonical is a versioned `SceneBundle`/`AnalysisContext` with references to raw/validated inputs, physical features, CROMA scene vectors, CROMA spatial tokens, regions, metadata, and provenance. It should be created once per validated scene and passed by reference to task plugins. It should avoid copying large tensors into every task result.

CROMA is not currently recomputed by every live stage because `_deep_features` reuses the process-local adapter, but raw physical/evidence calculations are separate passes. A future bundle may cache their outputs by an input/preprocessing/representation fingerprint. That optimization must be additive and must preserve current numerical behavior.

### B. Modality and fusion bottlenecks

Optical and SAR are separately validated, normalized, and encoded. The live physical joint vector and seeded `HybridFusion` concatenate/use both modality representations, but the live code correctly labels the fusion as a representation-only untrained output. The typed Phase 2 contract carries separate encodings and a pooled concatenation; it does not prove that a learned multimodal task head exists.

Future cross-modal fusion should plug in after validated modality representations and before a task plugin. It must declare whether it is concatenation, learned fusion, cross-attention, or another operation, and must not call a concatenated vector a semantic joint prediction without an evaluated task head.

### C. Evidence bottlenecks

The current evidence chain is concrete:

`raw pixel arrays -> derived maps -> 225 token aggregates -> deterministic threshold regions -> inferred allowlisted claims -> constrained text`

Evidence is stored in `spatial_evidence_v1` JSON within reports and in standalone spatial evidence artifacts. CROMA token slots and prediction attachment fields exist in the evidence schema, but live `run_analysis` does not currently supply CROMA or predictions to `build_spatial_evidence`. Evidence therefore supports physical/evidence claims, not learned segmentation or learned grounding.

Evidence v2 should preserve v1 fields and add an explicit epistemic type for every claim/support item: `MEASURED`, `DERIVED`, `PREDICTED`, or `INFERRED`. It should require source artifact/model references for predicted and inferred claims and retain the existing `confidence_status` distinction from calibrated probability.

### D. Provenance bottlenecks

The typed Phase 1-3 path can carry scene identity, acquisition metadata, normalization, CROMA checkpoint SHA-256, feature schema, source hashes, and experiment configuration. The live report carries analysis ID, retrieval source, metadata, evidence schema/version, and execution trace, but live CROMA provenance currently does not include checkpoint SHA-256 and live reports do not have a single typed provenance object spanning all stages.

A future provenance record must identify: request, scene/input artifact hashes, acquisition metadata, preprocessing profile/version, physical-feature schema/version, representation model/version/checkpoint hash, fusion version, task adapter/version/checkpoint/calibration, evidence artifact/version, execution environment/device, and parent artifact IDs. Paths should remain private in public reports.

### E. Task integration bottlenecks

The API has a fixed `run_analysis` workflow and a broad `MODES` allowlist, but several modes are explicitly unavailable. `ToolSelection` is metadata, not a plugin execution interface. There is no common typed `TaskRequest`/`TaskResult` or model adapter protocol in the live path. Adding VQA, captioning, grounding, or change models directly to `analysis_engine` would create conditional logic and make provenance inconsistent.

The missing boundary is a task registry plus adapter protocol that consumes a validated `SceneBundle` and returns a structured result. The existing analysis should remain the compatibility path until an adapter-backed implementation is verified against its tests and report contract.

### F. Controller/agent bottlenecks

There is a deterministic query parser and `select_tools`, but no controller that discovers capabilities, validates task support, selects a versioned tool, executes it with structured input, or reconciles multiple outputs. The optional `LLMProvider` is an interpretation boundary only and is intentionally unconfigured in the local runtime.

A future controller needs a read-only capability registry, structured request validation, explicit unsupported responses, deterministic tool selection rules, execution trace, and no ability to invent evidence or confidence.

### G. Temporal bottlenecks

The web path recognizes temporal language and validates the need for before/after inputs, but `run_analysis` returns temporal unavailable and emits no change map or area. `phase3_orchestration.TemporalRequest` and spatial-pair validation are stronger typed foundations, but they are not connected to a temporal representation or change task in the live API.

A future temporal bundle should preserve T1/T2 as separate scene references, acquisition dates, co-registration diagnostics, preprocessing fingerprints, and a derived change representation. Temporal evidence must cite both parent scenes and the transformation used.

### H. Confidence bottlenecks

Current reports deliberately state `model_confidence: None`, `calibration_status: uncalibrated`, and no accuracy claim. Evidence strength is a descriptive threshold category, not statistical confidence. Future confidence must be supplied by the task adapter and tied to a model/calibration artifact; the interpretation layer and any LLM must not synthesize it.

## 6. Proposed Target Architecture

`USER QUERY -> TASK PLANNER -> INPUT/GEO VALIDATOR -> DATA PROCESSING -> SCENE BUNDLE -> FUSION INTERFACE -> TASK PLUGIN -> EVIDENCE RECONCILER -> RESULT/REPORT -> PROVENANCE/AUDIT`

The boundaries are:

1. **Planner**: converts a request into a validated task request; it may use the existing deterministic parser, but it cannot claim unsupported capability.
2. **Input/GEO validator**: owns AOI, dates, source, raster, bands, CRS, grid, coverage, and temporal pair checks.
3. **Data processing**: materializes canonical optical/SAR/T1/T2 arrays and metadata without task-specific prediction logic.
4. **Shared representation layer**: creates the reusable `SceneBundle` and representation artifacts once.
5. **Fusion interface**: receives declared modality/temporal/text representations and returns a versioned fusion artifact; it must state whether it is learned and evaluated.
6. **Task plugin**: classification, VQA, captioning, grounding, temporal/change, or future task. It only consumes contracts and returns `TaskResult`.
7. **Evidence reconciler**: validates claim epistemic type, support artifact, spatial/temporal references, and compatibility with task output.
8. **Report/provenance layer**: produces the existing compatible report plus typed provenance and artifact links.
9. **Capability registry/controller**: discovers plugins and refuses unsupported or incompatible requests before execution.

## 7. Required Contracts

### 7.1 SceneBundle / AnalysisContext

Minimum conceptual shape:

```text
SceneBundle {
  bundle_id, schema_version,
  scene_identity,
  acquisition_metadata,
  geospatial_metadata,
  source_artifacts,
  optical: ModalityView | unavailable,
  sar: ModalityView | unavailable,
  temporal: TemporalContext | not_requested,
  physical_features: FeatureArtifact | unavailable,
  croma: RepresentationArtifact | unavailable,
  fusion_inputs: references only,
  spatial_tokens: TokenArtifact | unavailable,
  regions: RegionArtifact | unavailable,
  evidence: EvidenceArtifact | unavailable,
  provenance: ProvenanceRecord
}
```

A `ModalityView` contains canonical band order, shape, dtype, finite/mask status, acquisition identity, grid/CRS, artifact hash, and preprocessing profile. A representation artifact contains model/version/checkpoint hash, input artifact IDs, feature keys, shapes, dtype/device, normalization profile, and artifact hash.

The bundle is immutable after construction. Task plugins receive references and read-only arrays/artifacts; they do not mutate the validated scene.

### 7.2 TaskRequest

```text
TaskRequest {
  request_id,
  task,                 # classification | vqa | captioning | grounding | change | ...
  query,
  scene_ref or bundle_ref,
  modality,             # optical | sar | optical_sar | temporal | auto
  temporal_context,
  parameters,
  constraints,
  requested_outputs,
  capability_requirements,
  provenance_parent_refs
}
```

Unknown tasks, missing modalities, unsupported temporal context, and incompatible constraints must fail before model execution.

### 7.3 TaskResult

```text
TaskResult {
  result_id, status, task,
  answer_or_output,
  output_type,
  evidence_refs,
  confidence: ConfidenceRecord | unavailable,
  provenance: ProvenanceRecord,
  artifacts,
  diagnostics,
  limitations
}
```

`status` must distinguish `COMPLETED`, `PARTIAL`, `UNAVAILABLE`, `REJECTED`, and `FAILED`. An unavailable confidence is a valid value and must not be converted into a number.

### 7.4 Capability / Tool Registry

Each capability declaration should include:

- stable tool name and version;
- supported task types;
- supported modalities and input artifact types;
- temporal capability and co-registration requirements;
- accepted SceneBundle schema versions;
- output type;
- evidence capability and evidence schema versions;
- confidence/calibration capability;
- model/checkpoint identity;
- deterministic/resource constraints;
- explicit limitations.

The registry is declarative. The controller selects only a capability whose requirements match the request and bundle.

### 7.5 Evidence Contract v2

Preserve `spatial_evidence_v1` and `constrained_interpretation_v1` for compatibility. Evidence v2 should add:

```text
EvidenceItem {
  evidence_id,
  epistemic_type: MEASURED | DERIVED | PREDICTED | INFERRED,
  feature_or_output_id,
  value,
  unit,
  sensor_or_temporal_view,
  spatial_refs,
  temporal_refs,
  source_artifact_refs,
  model_ref: optional,
  support_refs,
  confidence_status,
  limitations
}
```

An inferred claim must cite the derived or predicted support it uses. A predicted item must cite its task/model artifact. All items must be traceable to input or parent artifacts. Evidence strength remains descriptive unless a calibrated confidence artifact is attached.

### 7.6 TemporalContext

```text
TemporalContext {
  status: NOT_REQUESTED | SINGLE | PAIR | INVALID,
  t1_scene_ref, t2_scene_ref,
  t1_acquisition, t2_acquisition,
  co_registration: status, method, residuals, artifact_ref,
  change_representation_ref,
  temporal_evidence_refs,
  limitations
}
```

No change result is valid without explicit T1/T2 identity and co-registration status.

### 7.7 ModelAdapter

```text
ModelAdapter {
  capability() -> CapabilityDescriptor
  validate(bundle, request) -> Diagnostics
  run(bundle, request) -> TaskResult
  provenance() -> ModelProvenance
}
```

Adapters must be deterministic with respect to declared configuration where claimed, must not load undeclared checkpoints, must return structured outputs, and must never fabricate evidence or confidence.

## 8. What Exists, What Changes, What Waits

### Already exists and can be reused

- Input/raster contracts, canonical bands, AOI/date validation, physical features, CROMA loading, typed Phase 1-3 identity/provenance, spatial evidence v1, constrained interpretation, immutable reports, and regression tests.

### Exists but needs refactoring or an adapter

- Unify the live report and typed scientific provenance without changing output semantics.
- Package computed physical/CROMA/token artifacts as an immutable SceneBundle.
- Replace fixed task conditionals with a registry behind compatibility wrappers.
- Make live CROMA checkpoint hash and representation artifact identity explicit.
- Connect supplied CROMA/prediction references to evidence only when a task actually produces them.
- Generalize evidence's unavailable-prediction wording; current code contains a sample-specific phrase.

### Missing but required before VQA

- A stable immutable SceneBundle/AnalysisContext contract.
- A validated TaskRequest/TaskResult contract and adapter registry.
- A real VQA model adapter with explicit supported modalities and output/evidence behavior.
- Evidence reconciliation for text answers, including measured/derived/predicted/inferred typing.
- Model and checkpoint provenance plus confidence semantics for the VQA model.
- A deliberate policy for whether VQA answers may be emitted when evidence is unavailable.

### Missing but can wait

- Captioning, grounding, object detection, learned segmentation, external provider acquisition, multitemporal models, change maps, learned cross-modal fusion, and a general agent controller can follow the foundation.

### Scientific-risk changes that must not happen now

- Do not replace CROMA or alter its normalization, image resolution, channel order, or checkpoint.
- Do not change the current hybrid predictor/representation, physical feature formulas, datasets, splits, or validated metrics.
- Do not convert evidence regions into learned segmentation claims.
- Do not call concatenated optical/SAR features a learned joint semantic result.
- Do not add free-form LLM claims, invented confidence, fabricated imagery, or unverified provider data.
- Do not route the live API to the typed Phase 3 path until compatibility and regression tests prove unchanged behavior.

## 9. Risks and Migration Strategy

1. **Contract drift**: version SceneBundle, evidence, task, and provenance schemas; keep v1 readers and the existing report shape.
2. **Two-path divergence**: first build an adapter that packages the live path's existing outputs into the typed bundle. Do not duplicate extraction logic.
3. **Provenance gaps**: require hashes and parent references at artifact creation, not at final text rendering.
4. **Capability overclaiming**: registry entries must explicitly declare unavailable calibration, grounding, temporal, and prediction support.
5. **Memory and latency**: keep large arrays/artifacts out of JSON responses; use immutable references and bounded caches.
6. **Backward compatibility**: retain existing routes and frontend fields; add typed internals and optional report fields before considering a major report version.
7. **Security and privacy**: retain private filesystem paths only in local diagnostics; public reports should use opaque artifact IDs and hashes.

Recommended migration sequence:

1. Freeze the contracts in `ARCHITECTURE_FREEZE.md`.
2. Add contract-only dataclasses/validators in a new boundary module, with no runtime routing change.
3. Write an adapter from the existing `run_analysis` result and evidence into a SceneBundle receipt.
4. Add a registry that initially exposes only the current feature/evidence capability and explicitly unavailable future capabilities.
5. Add a no-op/task-result compatibility wrapper and contract tests.
6. Only then implement a VQA adapter as a separate, evaluated task plugin.

## 10. Validation and Baseline

This audit is documentation-only. Existing tests were run after documentation planning and remain the source of truth for regression verification. The architecture proposal does not modify prediction results, datasets, benchmark splits, model weights, preprocessing, API routes, or frontend behavior.

Baseline facts to preserve:

- Live reports explicitly have no trained task prediction and no calibrated confidence.
- Temporal requests remain unavailable rather than producing a fabricated change result.
- Spatial evidence remains deterministic and descriptive.
- Pass 5D remains a completed pixel/local/token explainability-sidecar comparison; it does not become a new live predictor.
