# Current Project Status

Date: 2026-09-15

Status: **READY FOR NEXT RESEARCH PHASE**

## Canonical architecture

Pipeline 3 is the only canonical production/scientific architecture. The supported runtime is `src.api` → `src.analysis_engine.run_analysis` → validated Pipeline 3 processing → evidence and constrained interpretation → API/frontend.

- Pipeline 1: **ARCHIVED** — historical So2Sat FusionCNN experiment.
- Pipeline 2: **ARCHIVED** — historical GEE-only experiment.
- Pipeline 3: **CANONICAL** — current GEE + CROMA production/scientific architecture.

CROMA is one pretrained remote-sensing representation model. Its optical, SAR, joint, pooled/GAP, and spatial-token outputs are representations, not separate predictive models. Deterministic feature extraction supplies 62 physical dimensions. The final evaluated scene predictor concatenates those features with the 768-D joint CROMA GAP to create an 830-D vector for a linear softmax probe.

The separate 192-D `HybridFusion` output remains seeded/untrained representation infrastructure and is not the final trained predictor.

## Authoritative scientific checkpoint

- Dataset: BigEarthNet v2 selected 5,000-area verified multimodal dataset.
- Complete multimodal areas: 5,000/5,000.
- Split: 4,600 train, 200 validation, 200 held-out test.
- Test MAE: **4.1034 pp**.
- Test RMSE: **9.5873 pp**.
- Dominant-class accuracy: **65.0%**.
- Dataset fingerprint: `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`.
- Split fingerprint: `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`.
- Scientific fingerprint: `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`.

The exact metrics, ablations, repeatability checks, per-class analysis, test receipt, and machine-readable artifacts are documented in [Pipeline 3 Exact 5,000-Area Training Report](PIPELINE3_5000_TRAINING_REPORT.md).

## Completed architecture phases

- Canonical Pipeline 3 runtime consolidation and archived-pipeline isolation.
- Typed immutable architecture contracts for scenes, requests, results, references, evidence, and provenance.
- Executable deterministic scene-analysis capability and capability registry boundary.
- Representation/artifact references and logical `artifact://` addressing.
- Optional receipt-backed live representation persistence.
- Trusted local receipt catalog with checksum-verified materialization and lifecycle state.
- Exact 5,000-area S2 acquisition, multimodal completeness audit, frozen training, validation selection, sealed test evaluation, and scientific artifact generation.

These are architecture infrastructure components; they are not additional AI models.

## Current executable capabilities

- Deterministic Pipeline 3 scene analysis.
- Physical feature extraction.
- Official CROMA optical, SAR, joint, pooled, and spatial representations.
- Controlled hybrid scene-level prediction using the 830-D scientific representation.
- Spatial evidence and deterministic evidence claims.
- Constrained interpretation and provenance.
- Architecture contracts, representation/artifact references, optional local persistence, and trusted receipt lookup/materialization.
- Loopback API, report persistence/round-trip, and browser frontend.

## Future capabilities

The following are **not currently implemented**: VQA, captioning, learned grounding, temporal/change prediction, change VQA, calibrated confidence, and a full agentic planner. They must be introduced as separately validated capabilities without changing this frozen checkpoint.

## Dataset roles

### Current

The BigEarthNet v2 selected 5,000-area multimodal dataset is the core Pipeline 3 training/evaluation dataset. It contains 5,000 S1/SAR inputs, 5,000 S2/optical inputs, and 5,000 reference targets.

### Future

- `BigEarthNet.txt`: future remote-sensing language/adaptation resource.
- RSVQA: future single-image VQA evaluation.
- VRSBench: future captioning/grounding-related evaluation where applicable.
- CDVQA: future bi-temporal/change VQA evaluation.
- ISRO/SAC hidden dataset: future private/generalization evaluation.

The repository does not claim these future datasets are integrated.

## Known limitations

- Strong class imbalance and low support for rare classes.
- Coastal wetlands had zero positive held-out test examples.
- Current predictive validation is scene-level.
- Spatial evidence is deterministic; learned grounding is unavailable.
- VQA and production temporal/change prediction are unavailable.
- Confidence is descriptive and not calibrated probability.
- The receipt catalog is local.
- Representation persistence is synchronous/local.
- Hidden ISRO/SAC evaluation remains future work.

## Reproducibility and hygiene

Raw datasets, satellite imagery, downloaded archives, model weights, feature caches, local artifact stores, credentials, environment files, and transient outputs are excluded from version control. Small contracts, manifests, receipts, predictions, metrics, and reports required to identify and audit the checkpoint are intentionally retained.

Production configuration is environment-driven; logical persisted representations use `artifact://` references. Historical documents and frozen scientific artifacts may retain clearly contextualized machine-local acquisition paths as provenance. No production module resolves or depends on those recorded paths.

## Immediate next research phase

Select one future capability and define its dataset contract, model identity, evaluation protocol, evidence contract, and fail-closed runtime boundary before implementation. VQA, captioning, grounding, temporal/change, calibration, and agentic planning remain separate research tracks; none is implied by the current scene-level checkpoint.
