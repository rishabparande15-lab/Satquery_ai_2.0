# SatQuery AI Second-Pass Gap Closure Plan

Date: 2026-09-11

This plan is based on a fresh inspection of the four scientific audit documents, the current source/test tree, the prior 115-test result, and preserved uncommitted Phase 3.5/3.6 work. A component is marked closed only where execution evidence exists.

| Gap | Status | Current evidence | Closure action / boundary |
|---|---|---|---|
| Repository-wide regression collection | CLOSED | `pytest.ini` scopes collection to `tests`; 115 tests previously passed | Re-run after all changes |
| Canonical S1/S2 band constants and order | CLOSED | Shared constants and shape checks; real sample executed | Add exhaustive per-band missing tests at strict loader boundary |
| Strict BigEarthNet loader | OPEN | Existing loader discovers/resamples real three-sample data but does not load reference tensors or exhaustively validate source grids | Implement additive strict validation, reference loading, explicit duplicate/orphan checks, and failure-injection tests |
| Deterministic preprocessing | CLOSED | Versioned CROMA normalization and metadata; real execution | Preserve semantics; test finiteness and reproducibility through strict loader |
| Actual CROMA shape contract | CLOSED | Fresh sample produced all three `[1,225,768]` and scene `[1,768]` outputs | Re-run in multi-area smoke |
| 19-class target generation | PARTIALLY CLOSED | Existing deterministic `[N,225,19]` implementation and controlled 1,000-area evidence | Connect raw reference loading to smoke runner and independently verify invariants |
| Scientific target definition | OPEN | Token coverage exists; existing SatQuery scene-level hybrid repeats one scene vector across tokens | Document mathematically explicit token and scene aggregation contracts; do not conflate tasks |
| Token alignment / region mapping | PARTIALLY CLOSED | Deterministic box/token overlap module and 11 tests | Add synthetic target-to-token identity proof and optional debug visualization |
| Fresh 5-10 area raw-input training smoke | OPEN | Only fresh one-area representation run exists | Run 5-10 raw areas with train/validation/test, one or more actual epochs, checkpoint, predictions, metrics, and receipts |
| Physical-only coverage baseline | OPEN | Physical extraction exists; no apples-to-apples token coverage evaluation | Build same-split trainable baseline without unsupported features |
| CROMA-only coverage baseline | PARTIALLY CLOSED | Historical controlled joint-CROMA result exists | Freshly run same-split smoke and largest accessible evaluation |
| Aligned hybrid coverage baseline | OPEN | Historical scene-level hybrid result exists but loses token spatial detail | Define a deliberate scene-level target or token-aligned feature broadcast contract and evaluate honestly |
| Area-level split / leakage | PARTIALLY CLOSED | Existing 600/200/200 audit reports no overlap | Recompute from fresh manifests and write `docs/data_leakage_audit.md` |
| Fresh held-out evaluation | OPEN | Historical metrics exist; external prepared artifact directories were inaccessible in prior audit | Attempt raw-data/access recovery without destructive changes; run largest accessible same-split comparison |
| BigEarthNet.txt linkage | PARTIALLY CLOSED | Historical counts and conservative linker exist; Parquet is readable | Recompute counts and text-preservation checks from available annotation file if schema permits |
| Optical scientific prediction | PARTIALLY CLOSED | Representation path executes; no consolidated held-out optical head result | Evaluate if fresh smoke data supports it; otherwise retain representation-only status |
| SAR scientific prediction | PARTIALLY CLOSED | Representation path executes; no consolidated held-out SAR head result | Evaluate if fresh smoke data supports it; otherwise retain representation-only status |
| Joint scientific prediction | CLOSED for historical controlled task | Historical held-out joint-CROMA coverage metric exists | Distinguish historical result from fresh reproduction |
| Learned grounding | BLOCKED | Mapping foundation only | No suitable trained grounding dataset/model established; keep foundation status truthful |
| VQA | BLOCKED | No validated model/evaluation found | Do not implement without verified RSVQA-compatible resources; mark not validated |
| Captioning | BLOCKED | Annotation linkage only | No trained/evaluated VRSBench caption model found; mark not validated |
| Temporal/change prediction | BLOCKED | Spatial/date validation exists | No validated change model/data pair; retain rejection/unavailable behavior |
| Confidence calibration | BLOCKED | UI reports uncalibrated state | No eligible calibrated prediction deployment; retain `NOT VALIDATED` |
| Controller truthfulness | PARTIALLY CLOSED | Existing tool selection rejects unavailable specialists | Add/verify explicit four-state capability reporting where needed |
| Website integration | PARTIALLY CLOSED | Optical/SAR/joint representations and truthful errors work | Do not expose new heads without validated checkpoint; rerun API/frontend regression |
| Artifact provenance/tamper detection | CLOSED | Save/reload/hash tests and fresh bundle verification | Include smoke checkpoint/predictions/config in reproducible experiment tree |
| Clean-environment reproduction | OPEN | Dependencies/commands exist but external datasets/checkpoints are machine-local | Audit exact asset/dependency requirements and document blockers |
| Final independent audit | OPEN | First-pass audit exists | Perform only after second-pass implementation and real runs; rewrite final audit from evidence |

## Execution order

1. Strict loader tests (red), additive loader/reference implementation (green), real fixture verification.
2. Synthetic token/target alignment proof and scientific target contract documentation.
3. Build and run the 5-10 area smoke experiment from raw files, with immutable splits and actual optimization.
4. Evaluate constant, physical, CROMA, and hybrid representations on exactly the same smoke split; attempt the largest accessible controlled run.
5. Recompute leakage and BigEarthNet.txt linkage evidence.
6. Audit controller/UI truthfulness and failure paths without exposing unfinished models.
7. Run complete Python/frontend/compilation/static checks, artifact tamper tests, and independent receipt/metric recomputation.
8. Write fresh evaluation, leakage audit, reproducibility status, and the final audit.

## Scientific decisions fixed before implementation

- The primary reference task remains token-level class coverage: `X` is a frozen per-token representation and `Y[i,c]` is the fraction of the 8x8 pixel block for token `i` belonging to class `c`, with excluded pixels tracked separately.
- A scene-level SatQuery experiment must use a scene-level target obtained by summing class counts over the scene and dividing by the total eligible labelled pixels. Repeating one scene feature against 225 token targets is allowed only as a documented diagnostic, not as a spatially aligned hybrid claim.
- Physical scene summaries naturally support the scene-level target. CROMA scene GAP and physical+CROMA scene fusion can be compared on that same scene-level target. Token-level CROMA remains a separate spatial experiment.
- Test areas remain untouched until the configuration and selected epoch are fixed using train/validation data.

## Stop conditions

- If fewer than five valid raw areas with reference maps are readable, the mandatory smoke run is BLOCKED and no synthetic substitute will be called real data.
- If external artifacts remain permission-inaccessible, historical 1,000-area metrics remain historical evidence only.
- No VQA, captioning, grounding prediction, temporal prediction, or calibrated confidence will be claimed without an evaluated model and ground truth.
