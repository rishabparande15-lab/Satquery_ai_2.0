# SatQuery AI Architecture Freeze

Date: 2026-09-13

Status: recommended architecture boundary for future work. This is a freeze proposal, not an implementation plan that changes the current scientific or web runtime.

## Freeze Decision

Keep the current validated analysis behavior intact and place future capabilities behind typed, versioned boundaries. The live API remains responsible for compatibility and deterministic feature/evidence reports. Future VQA, captioning, grounding, temporal/change, calibration, and agentic orchestration must consume a shared immutable scene context through task adapters.

The current system is not to be represented as a completed multimodal assistant. It is a validated local feature-and-evidence pipeline with optional pretrained CROMA representations, representation-only hybrid fusion, deterministic spatial evidence, and constrained interpretation.

## Frozen Boundaries

### 1. Request and planning boundary

Current owner: `src/api.py`, `src/analysis_engine.py`, `src/query_interpreter.py`.

Canonical future contract: `TaskRequest`.

The planner may parse a query and select a task, but it must only request capabilities declared by the registry. It must preserve the current deterministic parser as a fallback and must return an explicit unsupported result when a task is not installed.

```text
TaskRequest {
  request_id,
  task,
  query,
  scene_ref or bundle_ref,
  modality,
  temporal_context,
  parameters,
  constraints,
  requested_outputs,
  capability_requirements,
  provenance_parent_refs
}
```

### 2. Input and geospatial validation boundary

Current owners: `src/input_validation.py`, `src/raster_inputs.py`, `src/phase3_orchestration.py`.

This boundary owns AOI, dates, source, coverage, CRS, transform, dimensions, finite/mask checks, canonical band order, optical/SAR compatibility, and future T1/T2 co-registration checks. It must produce structured diagnostics and must not silently repair scientifically meaningful incompatibilities.

The existing web validation behavior remains compatible. The typed Phase 3 contracts are the preferred foundation for future acquisition and temporal work, but adoption into the web path requires a compatibility adapter and regression proof.

### 3. Data processing boundary

Current owners: `src/data_orchestrator.py`, `src/raster_inputs.py`, `src/multimodal_cube.py`, `src/dataset_loader.py`.

This boundary materializes validated inputs and metadata. It does not decide a task answer. External providers remain explicit adapters; local development samples must remain labelled as local and must not be described as GEE retrieval.

### 4. Shared representation boundary

Current owners: `src/modality_features.py`, `src/gee_features.py`, `src/croma_adapter.py`, `src/phase1_foundation.py`, `src/phase2_integration.py`, `src/hybrid_fusion.py`.

Future canonical object: immutable `SceneBundle` / `AnalysisContext`.

```text
SceneBundle {
  bundle_id, schema_version,
  scene_identity,
  acquisition_metadata,
  geospatial_metadata,
  source_artifacts,
  optical_view,
  sar_view,
  temporal_context,
  physical_feature_artifact,
  croma_representation_artifact,
  fusion_artifact_refs,
  spatial_token_artifact,
  region_artifact,
  evidence_artifact,
  provenance
}
```

Rules:

- The bundle is immutable after validation/materialization.
- Large arrays are artifacts or read-only references, not duplicated in every JSON result.
- Each artifact carries schema/version, shape, dtype, input references, preprocessing profile, model/checkpoint identity where applicable, and SHA-256 when persisted.
- Optical, SAR, and joint representations remain distinguishable.
- CROMA encodings and pooled vectors remain separate from physical features.
- A concatenation or seeded untrained fusion vector is a representation, not a semantic task prediction.
- Existing live extraction and scientific extraction remain unchanged until an adapter is verified.

### 5. Fusion boundary

Future owner: a versioned fusion adapter, not `analysis_engine` conditionals.

```text
FusionAdapter {
  capability() -> CapabilityDescriptor
  validate(inputs, request) -> Diagnostics
  run(inputs, request) -> FusionArtifact
}
```

A descriptor must state modality inputs, temporal inputs, whether fusion is learned, training/evaluation status, output shapes, checkpoint identity, and limitations. Optical-SAR, T1-T2, and image-text fusion are separate capability declarations.

### 6. Task plugin boundary

Future owner: a task registry plus `ModelAdapter` implementations.

```text
ModelAdapter {
  capability() -> CapabilityDescriptor
  validate(bundle, request) -> Diagnostics
  run(bundle, request) -> TaskResult
  provenance() -> ModelProvenance
}
```

Every task plugin must consume a `SceneBundle` and return a `TaskResult`; it must not reach into HTTP globals, upload registries, or frontend state. It must not load undeclared weights or silently substitute an unavailable modality.

```text
TaskResult {
  result_id,
  status,                 # COMPLETED | PARTIAL | UNAVAILABLE | REJECTED | FAILED
  task,
  answer_or_output,
  output_type,
  evidence_refs,
  confidence,
  provenance,
  artifacts,
  diagnostics,
  limitations
}
```

### 7. Evidence boundary

Current owners: `src/evidence_schema.py`, `src/interpretation_adapter.py`.

Keep `spatial_evidence_v1` and `constrained_interpretation_v1` backward compatible. Future Evidence Contract v2 may be additive and must distinguish:

- `MEASURED`: directly observed input/metadata value;
- `DERIVED`: deterministic calculation from measured data;
- `PREDICTED`: output of a declared evaluated model/task;
- `INFERRED`: claim derived from measured, derived, or predicted support.

```text
EvidenceItem {
  evidence_id,
  epistemic_type,
  feature_or_output_id,
  value,
  unit,
  sensor_or_temporal_view,
  spatial_refs,
  temporal_refs,
  source_artifact_refs,
  model_ref,
  support_refs,
  confidence_status,
  limitations
}
```

Rules:

- Every claim has support references.
- Predicted claims identify task/model/checkpoint artifacts.
- Inferred claims identify the evidence items from which they were formed.
- Spatial regions remain evidence regions unless learned segmentation is separately trained and evaluated.
- Evidence strength is descriptive and cannot be serialized as calibrated confidence.
- The interpretation adapter may render only validated, allowlisted claims.
- An LLM/VLM may verbalize accepted evidence only under a future constrained adapter; it may not invent claims, locations, confidence, or provenance.

### 8. Temporal boundary

Future owner: `TemporalContext` plus temporal processing/task adapters. Current web behavior remains unavailable for temporal inference.

```text
TemporalContext {
  status,
  t1_scene_ref,
  t2_scene_ref,
  t1_acquisition,
  t2_acquisition,
  co_registration,
  change_representation_ref,
  temporal_evidence_refs,
  limitations
}
```

A temporal result is invalid without explicit T1/T2 identity, acquisition metadata, co-registration status, and a declared change representation. Temporal evidence must cite both parent scenes and the transformation/model that generated it.

### 9. Provenance and audit boundary

Current owners: live report construction, `phase1_foundation`, `phase2_integration`, and `scientific_artifacts`.

Canonical future `ProvenanceRecord` fields:

```text
ProvenanceRecord {
  request_id,
  bundle_id,
  parent_artifact_refs,
  input_artifact_refs_and_hashes,
  acquisition_and_geospatial_metadata,
  preprocessing_profile_and_version,
  feature_schema_and_version,
  representation_model_and_checkpoint_hash,
  fusion_model_and_checkpoint_hash,
  task_model_and_checkpoint_hash,
  calibration_artifact_ref,
  evidence_schema_and_artifact_refs,
  execution_environment,
  device,
  timestamps,
  software_revision
}
```

Public reports must use opaque artifact IDs and hashes rather than server filesystem paths. The existing live report shape remains supported while typed provenance is introduced additively.

### 10. Capability registry and controller boundary

Future owner: declarative `CapabilityRegistry` and a fail-closed controller.

A capability descriptor must declare tool name/version, task types, modalities, input artifact types, temporal requirements, accepted bundle schema, output type, evidence support, confidence/calibration support, model identity, and limitations.

The controller may:

1. discover registered capabilities;
2. validate a `TaskRequest` against a capability and `SceneBundle`;
3. select a compatible version using deterministic rules;
4. execute the adapter;
5. collect structured `TaskResult`, evidence, provenance, and diagnostics.

It may not invent unsupported tools, bypass validation, synthesize confidence, or merge incompatible evidence. The current `select_tools` function remains a compatibility metadata layer until this registry exists.

## Backward Compatibility Rules

- Preserve current API routes, request fields, response fields, and frontend state behavior.
- Preserve `schema_version: "2.0"` reports unless a compatibility layer is introduced and tested.
- Preserve current physical feature formulas and dimensions, including the explicitly versioned legacy joint schema.
- Preserve CROMA normalization, canonical bands, 120x120 input requirement, representation keys, and validated checkpoint behavior.
- Preserve current spatial evidence and constrained interpretation schemas.
- Preserve explicit unavailable states for task prediction, temporal change, and calibration.
- Preserve existing experiment splits, metrics, artifact receipts, and Pass 5D results.
- Add new fields or sidecar artifacts before changing existing meanings.
- Do not route the live API through the typed Phase 3 path solely for architectural neatness; first prove output and regression equivalence.

## Minimum Foundation Before VQA

Before implementing VQA, implement only the following architectural foundation:

1. A versioned, immutable `SceneBundle`/`AnalysisContext` dataclass and validator.
2. An adapter that packages the existing live analysis outputs into that bundle without changing extraction or report semantics.
3. Typed `TaskRequest`, `TaskResult`, `CapabilityDescriptor`, and `ProvenanceRecord` contracts.
4. A registry initially containing current feature/evidence capabilities and explicit unavailable entries for VQA and other future tasks.
5. Evidence reconciliation that preserves v1 and supports epistemic typing for future task outputs.
6. Contract/regression tests proving the current API report and scientific metrics remain unchanged.

Do not add VQA training or a VQA model in the architecture-freeze step.

## Explicitly Deferred

The following are outside this freeze and must not be implied as implemented:

- VQA model/training and open-ended answer generation;
- captioning;
- learned grounding or segmentation;
- object detection/counting;
- temporal/change prediction;
- learned cross-modal or image-text fusion;
- calibrated confidence;
- external provider acquisition;
- a general-purpose agentic controller;
- automatic migration of live API execution to Phase 3 orchestration.

## Migration Guardrails

Any future implementation must include:

- a compatibility test against the current report contract;
- focused tests for schema validation and unsupported capability behavior;
- artifact/provenance verification;
- explicit scientific-risk review if preprocessing, model loading, feature formulas, evidence thresholds, or evaluation data are touched;
- no dataset or split changes as part of interface work;
- no unreviewed changes to validated prediction outputs.

The next implementation task should be the contract-only SceneBundle/TaskRequest/TaskResult foundation plus an adapter around the existing `run_analysis` result. It should stop short of VQA and should be validated against the current API, evidence, and scientific regression tests.
