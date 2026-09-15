# Canonical Architecture Contracts

## Purpose and boundary

Pipeline 3 is the only canonical SatQuery production architecture. The production entry point remains `src.analysis_engine.run_analysis`; its GEE-style physical features, CROMA representations, deterministic spatial evidence, constrained interpretation, API response, and persisted report are unchanged.

The contracts in `src.architecture_contracts` create typed boundaries around that runtime so later capabilities can exchange scene context, requests, results, evidence, and provenance without redesigning Pipeline 3. They are Python-native frozen dataclasses with explicit validation. They do not load plugins, select executable code, or run new models.

The name **SceneBundle** was selected because the object bundles the inputs, geospatial metadata, optional representations, evidence, and provenance belonging to one scene. `AnalysisContext` could imply mutable execution state; the implemented contract is an immutable boundary object.

```text
                    PIPELINE 3
                        |
                        v
                   SceneBundle
                        |
             +----------+----------+
             v                     v
        TaskRequest           Existing Analysis
             |                     |
             v                     v
       Capability Registry   Compatibility Adapter
             |                     |
             +----------+----------+
                        v
                    TaskResult
                        |
                        v
                Evidence + Provenance
                        |
                        v
                 Future Answer Layer
```

VQA, captioning, learned grounding, temporal/change models, new VLMs, multimodal fusion models, planners, LLM reasoning, prediction models, and training changes are **not implemented** by this foundation.

## SceneBundle

`SceneBundle` is the one reusable scene context. It includes:

- identity: `scene_id`, optional `analysis_id`, and input identifiers;
- input state: input type, available modalities, and explicit optical/SAR/temporal availability;
- optional temporal context;
- geospatial context: CRS, bounds, resolution, dimensions, georeferencing status, and co-registration status;
- a `RepresentationSet` for physical, optical, SAR, joint, CROMA scene, spatial-token, and region representations;
- the existing spatial-evidence object and other task evidence references;
- a provenance mapping.

Only `scene_id` is required for a minimal bundle. Optical-only and SAR-only bundles are valid. Temporal context, joint features, CROMA, tokens, regions, and evidence may be absent. Availability flags must agree with `available_modalities`; dimensions, resolution, and bounds are validated when present.

The representation fields intentionally accept the existing representation objects. The contract does not copy, reinterpret, normalize, or assign semantics to latent CROMA dimensions.

## TaskRequest

`TaskRequest` provides one generic request interface with:

- `task_type`;
- either `scene_id` or a structured `scene_reference`;
- optional `query`;
- requested modalities and optional temporal context;
- open mappings for parameters, constraints, and execution metadata.

The request validates structure without pretending every future task is known. It rejects empty identifiers, malformed queries, duplicate modalities, VQA without a query, `optical_sar_analysis` without both optical and SAR, and `temporal_change` without temporal context. Those checks describe detectable impossibilities only; they do not implement or advertise the corresponding tasks.

## TaskResult

`TaskResult` provides one generic result interface:

```text
status -> task_type -> output
                    -> evidence (optional)
                    -> confidence (optional, calibrated only)
                    -> provenance
                    -> artifacts
                    -> diagnostics
```

Canonical statuses are `success`, `partial`, `failure`, and `unavailable`. Evidence is `None` when a task produced none. Confidence is `None` unless a task supplies a finite value from zero to one and explicitly marks it calibrated. An uncalibrated numeric value is rejected; adapters and future answer layers therefore cannot silently manufacture confidence.

## Capability and registry

`Capability` is discovery metadata: name, task type, version, supported modalities and input types, temporal/evidence/confidence support, and input/output contract names. It contains no executable code.

`CapabilityRegistry` supports registration, lookup by name, deterministic listing, matching, and a boolean support check. Matching considers task type, requested modalities, temporal support, and—when a `SceneBundle` is supplied—the scene input type and actually available modalities. Duplicate names are rejected. The registry is deliberately an in-memory abstraction, not a plugin loader or planner.

## Compatibility adapter

`adapt_run_analysis_output(result, request)` creates a `SceneBundle` and `TaskResult` view of an already-computed legacy result. `run_analysis_with_contracts(request)` is a convenience wrapper with this exact flow:

```text
src.analysis_engine.run_analysis(request)
    -> unchanged legacy result
    -> adapt_run_analysis_output(...)
    -> AdaptedAnalysis(legacy_result, SceneBundle, TaskResult)
```

`run_analysis` remains the one canonical computation entry point. The adapter does not invoke feature extraction, CROMA, evidence generation, interpretation, prediction, or metrics. Existing feature, evidence, interpretation, and artifact objects are attached directly to the contracts, preserving their values and epistemic labels.

Legacy status names are mapped only in the contract view: `completed` to `success`, `partial` to `partial`, `rejected` to `failure`, and `unavailable` to `unavailable`. The original status remains in diagnostics and the legacy result is retained unchanged.

## Evidence integration

The established `spatial_evidence_v1` structure remains canonical. The pixel-to-token-to-region-to-claim-to-provenance chain is neither replaced nor flattened:

```text
pixel -> token -> region -> claim -> provenance
                    |
                    +-> SceneBundle.spatial_evidence
                    +-> TaskResult.evidence
```

Legacy evidence references are exposed as scene task-evidence references and result artifacts. When `spatial_evidence.status` is not `AVAILABLE`, canonical evidence is `None`. No semantic evidence, learned grounding, or future-task evidence is inferred.

## Provenance propagation

The adapter provides one shared provenance mapping to the scene and task result. It identifies attachment points for:

- `run_id` and execution timestamp/trace;
- input identifiers and retrieval source;
- existing preprocessing steps;
- physical schema and CROMA source/version information when supplied by the runtime;
- task type and legacy schema/task version;
- the complete existing spatial-evidence provenance object.

This preserves the current evidence provenance rather than introducing a competing evidence format. Future capabilities should add their task/model versions to the task section and pass the same provenance forward to their `TaskResult` and evidence artifacts.

## Backward compatibility

The API still returns the exact dictionary produced by `run_analysis`, and `save_report` still persists that dictionary. No contract field is injected into `/api/analyze`, so the frontend and downloaded-report schema do not need to migrate. Callers that need the new view can use the adapter explicitly.

The contracts are additive. A future API version may serialize `SceneBundle.to_dict()` or `TaskResult.to_dict()` under a new endpoint or additive field only after its response contract is reviewed.

## Future extension points

A future capability can plug in conceptually as:

```text
SceneBundle
    -> TaskRequest
    -> Capability metadata and registry match
    -> separately implemented task
    -> TaskResult
    -> existing evidence + shared provenance
```

Before implementing any such capability, review its scientific validity, modality requirements, confidence calibration, evidence semantics, and API exposure. The next phase is an architectural decision point; it is not automatically a VQA phase.
