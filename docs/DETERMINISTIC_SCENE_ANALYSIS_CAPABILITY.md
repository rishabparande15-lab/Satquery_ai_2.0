# Deterministic Scene Analysis Capability

## 1. Purpose

`DeterministicSceneAnalysisCapability` is the first executable use of the Phase
6A architecture contracts. It exposes the existing deterministic SatQuery scene
analysis through `TaskRequest`, `SceneBundle`, `CapabilityRegistry`, and
`TaskResult`. It adds no model, formula, label, claim, or scientific inference.

## 2. Why this is the first executable capability

Pipeline 3 is already the validated production runtime and already produces
physical features, CROMA representations, deterministic spatial evidence, and
constrained interpretation. Wrapping that mature path proves that the generic
contracts can execute real work before any new task family is considered.

## 3. Relationship to Pipeline 3

The capability delegates exactly once to `src.analysis_engine.run_analysis`.
That function remains authoritative for input assembly, preprocessing, physical
features, CROMA execution, hybrid representation, evidence, interpretation, and
the legacy result. Pipeline 1 and Pipeline 2 remain reproduction-only.

```text
TaskRequest
     ↓
Capability Registry
     ↓
Deterministic Scene Analysis
     ↓
Pipeline 3 / run_analysis
     ↓
TaskResult
     ├── output
     ├── evidence
     ├── provenance
     └── confidence = unavailable
```

## 4. Capability metadata

| Field | Value |
|---|---|
| Name | `deterministic_scene_analysis` |
| Task type | `deterministic_scene_analysis` |
| Version | `1.0.0` |
| Input contract | `TaskRequest` |
| Output contract | `TaskResult` |
| Input types | `satellite_scene` |
| Modalities | optical, SAR, or their joint combination |
| Temporal support | false |
| Spatial evidence support | true |
| Confidence support | false |

Evidence support refers only to the existing deterministic evidence pipeline.
It does not mean learned grounding. Confidence is not inferred from evidence
strength and always remains unavailable.

## 5. TaskRequest

The task type must be `deterministic_scene_analysis`. A non-empty query and a
usable scene reference are required. A scene may be identified by `scene_id`
(mapped to the existing local `sample_id`) or by `scene_reference`, including an
existing upload `files` map. Requested modalities may be empty, optical, SAR,
or optical plus SAR. Fixed modality combinations map to the corresponding
existing analysis mode; this is contract adaptation, not semantic planning.

Wrong task types and malformed required input return a failure `TaskResult`.
Unsupported modalities, input types, or temporal requests return an unavailable
`TaskResult`. No unsupported request silently falls through to another task.

## 6. SceneBundle usage

The capability accepts an existing `SceneBundle` or creates a minimal canonical
scene context from the task request. It validates scene identity, input type,
and declared modality availability before calling Pipeline 3. The existing
compatibility adapter then creates the populated post-run bundle with physical,
CROMA, hybrid, region, evidence, geospatial metadata, and provenance references.

Pipeline 3 currently accepts raster/request inputs rather than stored contract
representations. Consequently this phase does not pretend it can bypass CROMA
or physical computation from an arbitrary pre-populated bundle. Adding such a
cache boundary would require separate scientific identity tests; rewriting the
engine to do so is intentionally out of scope.

## 7. Adapter boundary

`Pipeline3AnalysisAdapter` deep-copies the request-side scene reference, adds the
query and fixed requested-modality mode, and calls `run_analysis`. It never
modifies the caller's request, `SceneBundle`, or returned legacy result. The
existing `adapt_run_analysis_output` function creates contract views after the
scientific run; it performs no feature extraction or interpretation itself.

## 8. TaskResult mapping

The legacy result remains directly available as `output.analysis`. Existing
`model_results`, `features`, `interpretation`, and explanation objects are also
referenced in `output`. Legacy completion states map to the generic result
status. Validation, warnings, legacy confidence state, and errors remain in
diagnostics. Existing artifact references remain in `artifacts`.

The task type on the capability result is the architectural task type, while
the unchanged Pipeline 3 parser task remains inside `output.analysis` and the
provenance task record.

## 9. Evidence handling

When Pipeline 3 reports spatial evidence as available, the same evidence object
is attached to `TaskResult.evidence` and `SceneBundle.spatial_evidence`. Claims,
regions, sensor views, and provenance are neither copied into a new schema nor
expanded. The authoritative chain remains pixel → token → region → claim →
provenance. No semantic evidence or learned grounding is generated.

## 10. Provenance handling

The mapped provenance retains the run identifier, public input identifiers and
source, existing preprocessing receipt, physical/CROMA representation sources,
Pipeline 3 task/schema information, timestamp and execution trace, and existing
evidence provenance. This is a view over the current provenance, not a second
provenance system.

## 11. Confidence semantics

`TaskResult.confidence` is `None` and `confidence_calibrated` is false. The
legacy confidence object remains diagnostic context and continues to state that
there is no trained task head or calibrated confidence. Evidence strength is
not converted into probability or confidence.

## 12. API compatibility

Phase 6A.5 does not change `src.api`, `/api/analyze`, `/api/report`, the saved
report schema, or frontend code. The API continues returning and persisting the
exact dictionary produced by `run_analysis`. The capability is an additive
internal/programmatic path.

## 13. Testing

`tests/test_deterministic_scene_analysis.py` covers sole registration, exact
metadata, lookup by name and task type, rejection of unknown capabilities,
request and scene validation, unsupported modality and temporal handling, the
registry-led execution path, fixed mode adaptation, input immutability, output
identity, evidence claim/region counts, provenance, and unavailable confidence.
The complete Python and frontend suites, compilation, API health, and a real
`61_39` execution are required release checks.

## 14. Scientific invariants

The normalized before/after fingerprint includes physical feature values,
CROMA values and shapes, hybrid values, spatial scene/sensor/claim/region
evidence, rendered interpretation and claims, model-result state, confidence,
and temporal state. It excludes generated IDs, timings, CROMA cache state, and
semantic hashes derived from run-specific provenance. Generated run UUIDs are
normalized recursively, including IDs embedded in evidence artifact strings.

The representative `61_39` run must remain completed with physical dimension
62, pooled CROMA dimension 2304, hybrid dimension 192, 17 evidence regions,
three claims, interpretation status `ANSWERED`, and confidence unavailable.

## 15. Known limitations

This capability is single-scene and deterministic. It has no VQA, captioning,
learned grounding, temporal/change prediction, trained task head, calibrated
confidence, autonomous planning, or dynamic routing. It exposes current live
representations; it does not reconstruct live CROMA token arrays that the
legacy response does not retain.

## 16. Future capability integration

Future scientifically validated capabilities can implement the same boundary:
declare honest `Capability` metadata, validate a `TaskRequest` against a
`SceneBundle`, execute a task-specific adapter, and return a `TaskResult` with
explicit evidence, provenance, and confidence semantics. They should be added
to the registry only after their own computation and scientific contracts are
validated. VQA is not implied or initiated by this phase.
