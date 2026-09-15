# Capability Roadmap

## Guiding order

Capability delivery is organized by dependency, not by a promise that every task follows linearly. Architecture, model development, dataset/evaluation, and integration can progress in parallel only when their prerequisites are stable.

```text
Phase 6A contracts and review
        |
        v
Deterministic Pipeline 3 scene analysis
        |
        +--------------------+
        v                    v
Numerical/physical      Representation and
analysis               capability descriptors
        |                    |
        +----------+---------+
                   v
       Learned scene classification decision
                   |
          +--------+---------+
          v                  v
  Grounding/evidence     VQA model + benchmark
  infrastructure        research (not API-ready)
          |                  |
          +--------+---------+
                   v
          Evidence-faithful VQA gate

Temporal context + paired evidence is a separate shared workstream.
Change VQA depends on both temporal prediction and VQA validation.
```

## Stage 1 — first executable capability

### ARCHITECTURE

- Specify and register `pipeline3_scene_analysis` / `scene_analysis`.
- Exercise `TaskRequest`, registry matching, `SceneBundle`, compatibility adapter, `TaskResult`, evidence, and provenance.
- Preserve the legacy API response and report schema.

### MODEL DEVELOPMENT

- None. Use existing Pipeline 3 computations only.

### DATASET/EVALUATION

- Golden `61_39`, optical-only, SAR-only, joint, missing-CROMA, rejected-input, and unavailable-evidence fixtures.
- Exact scientific fingerprint and evidence/interpretation equality.

### INTEGRATION

- Add an internal capability execution boundary after specification approval.
- Keep `/api/analyze` behavior unchanged until compatibility tests pass.

Exit gate: the complete common-contract path works without changing scientific output or inventing confidence.

## Stage 2 — bounded numerical and optical-SAR capabilities

### ARCHITECTURE

- Add representation descriptors with artifact identity, shape, dtype, modality, grid, producer/version, and semantic status.
- Add required/optional representation declarations and accepted schema versions to capability metadata.
- Define typed structured-value outputs with units.

### MODEL DEVELOPMENT

- None for formula-backed numerical analysis.
- No new fusion model for deterministic optical-SAR reporting.

### DATASET/EVALUATION

- Formula and unit verification, missing-band behavior, sensor-specific tests, grid/CRS cases, and numerical tolerances.
- Evaluate whether deterministic optical-SAR comparisons produce useful, non-overclaimed outputs.

### INTEGRATION

- Expose only bounded structured values and existing evidence.
- Maintain explicit sensor separation and `confidence=None`.

Exit gate: requirements-based registry matching and typed outputs are proven on at least two distinct capabilities.

## Stage 3 — learned scene-classification decision

### ARCHITECTURE

- Define task-model adapters, output types, per-output evidence links, checkpoint identity, and calibration records.
- Decide whether the production representation is joint CROMA GAP, token CROMA, physical features, or an evaluated combination.

### MODEL DEVELOPMENT

- Select one trained head supported by the intended target and population.
- Do not promote the live seeded untrained `HybridFusion` vector into a predictor.

### DATASET/EVALUATION

- Freeze accessible train/validation/test splits and leakage controls.
- Reproduce metrics, class support, baselines, modality ablations, calibration, and artifact hashes.
- Decide whether existing Pass 3/Pass 5 results match the production task and population.

### INTEGRATION

- Register only the evaluated modalities and representation versions.
- Attach model predictions to evidence and provenance; keep confidence unavailable until calibrated.

Exit gate: a production-approved checkpoint, evaluation report, evidence policy, and rollback-compatible API adapter exist.

## Stage 4 — shared grounding and answer-evidence infrastructure

### ARCHITECTURE

- Persist live CROMA tokens through bounded immutable artifact references.
- Add answer/output-to-evidence relations and an evidence reconciliation validator.
- Distinguish deterministic regions, learned grounding, bbox, mask, polygon, and scene-level support.

### MODEL DEVELOPMENT

- Evaluate query-conditioned grounding only after its target and spatial output are defined.

### DATASET/EVALUATION

- Select spatially annotated data with leakage controls and metrics appropriate to each output type.
- Test evidence faithfulness and unsupported localization.

### INTEGRATION

- Never relabel deterministic threshold regions as learned masks.
- Fail closed when an output implies spatial support that the capability cannot produce.

Exit gate: learned and deterministic spatial evidence are typed, evaluated, and visibly distinct.

## Stage 5 — VQA research and readiness gate

### ARCHITECTURE

- Define supported question taxonomy, question representation, modality policies, task output schema, unsupported-question behavior, and provenance requirements.
- Require representation-aware registry matching and answer-level evidence attachment.

### MODEL DEVELOPMENT

- Select a question-conditioned architecture; evaluate optical-only and joint optical-SAR variants deliberately.
- Do not assume CROMA embeddings alone answer questions.

### DATASET/EVALUATION

- Build a versioned benchmark adapter, potentially for RSVQA after dataset/license/task review.
- Freeze geographic splits, vocabularies, metrics, baselines, and modality ablations.
- Evaluate answer accuracy, evidence faithfulness, grounding, hallucination/unsupported rejection, and calibration.

### INTEGRATION

- Expose VQA only after the evidence and unsupported-answer gates pass.
- Keep numeric confidence `None` unless the calibration artifact applies to the deployed task and population.

Exit gate: the VQA capability can reject unsupported questions and every emitted factual answer has valid task/model/input provenance and the required evidence status.

## Stage 6 — captioning

### ARCHITECTURE

- Define structured caption claims and evidence relations rather than accepting unrestricted prose.

### MODEL DEVELOPMENT

- Train or adapt a caption model only against an approved remote-sensing caption target.

### DATASET/EVALUATION

- Evaluate factuality, evidence support, coverage, hallucination, sensor attribution, and human usefulness—not text-overlap alone.

### INTEGRATION

- Reuse VQA answer-evidence controls and provenance; keep constrained interpretation separately labelled.

Exit gate: captions cannot introduce claims absent from validated predicted/derived evidence.

## Parallel temporal workstream

### ARCHITECTURE

- Add a typed T1/T2 context with identities, acquisitions, modality correspondence, co-registration method/status, parent hashes, and temporal evidence references.
- Define temporal representation and change-result descriptors.

### MODEL DEVELOPMENT

- Select a change operator/model only after target semantics are fixed. Vector subtraction is a diagnostic, not automatically a change predictor.

### DATASET/EVALUATION

- Freeze temporal pairs, spatial leakage controls, change labels, no-change baselines, alignment tolerances, and region/map metrics.

### INTEGRATION

- Continue returning unavailable from the live API until a validated capability exists.
- Attach every change claim to both scenes and the alignment/model lineage.

Exit gate: temporal/change analysis has a validated prediction target, paired evidence, and honest confidence semantics.

## Final dependent capability — change VQA

Change VQA is not a near-term stage. It begins only after both the VQA readiness gate and temporal/change exit gate pass. It requires question-conditioned temporal fusion, answer-level temporal/spatial evidence, dual-scene provenance, unsupported-question behavior, and a dedicated evaluation protocol.

## Decision checkpoints

1. Approve the deterministic scene-analysis specification before implementation.
2. Approve additive representation/capability descriptor changes before registering token-dependent capabilities.
3. Review the production classification target and existing experiment applicability before selecting a checkpoint.
4. Review dataset, licensing, leakage, evidence, and calibration plans before VQA model work.
5. Review temporal target semantics and pair construction independently of VQA.

This roadmap deliberately does not make VQA the automatic next implementation.
