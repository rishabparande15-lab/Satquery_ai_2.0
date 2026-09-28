# Phase 3M — SatQuery Research-Readiness Remediation Plan

**Scope:** planning and evidence audit only. This plan preserves the Phase 3L.5 classifications, Phase 2F fail-closed spatial rule, frozen scientific pipeline, adapters, contracts, datasets, checkpoints, splits, receipts, controller, API, and frontend. Existing dirty-worktree changes were present before this report; no production or scientific artifact was modified.

## 1. Executive summary

Phase 3L.5 establishes **NOT_RESEARCH_READY**. Of 47 audited requirements, 24 are IMPLEMENTED, 14 EXPERIMENTAL, 7 BLOCKED, 1 UNVERIFIED, and 1 DEFERRED. There are **22 active gaps** (14 experimental + 7 blocked + 1 unverified) and one intentional deferred design choice.

Research readiness is prevented by four data-first gates—SAR language, paired optical-SAR, learned temporal/change, and grounding—and by the three mandatory benchmark paths (RSVQA, VRSBench, CDVQA), hidden ISRO/SAC verification, and a conclusive full-suite result. No model or evaluator work may bypass its preceding data, license, provenance, split/leakage, and spatial/temporal-semantic gate.

## 2. Phase 3L.5 baseline and current state

The authoritative frozen scientific baseline remains **65.0% accuracy**, **4.1034286734 pp MAE**, and **9.5873312123 pp RMSE**. It is a 5,000-area Pipeline 3 coverage result, not a VLM metric.

Implemented boundaries needing no remediation are the frozen scientific predictor; 12-band S2 engineering contract; BigEarthNet.txt text foundation; spatial/temporal/grounding contracts; SAR projector; bounded Qwen RGB and RSRCC temporal RGB execution; controller, planner, typed evidence, summaries, report, API/web, scene resolver; checkpoint/receipt infrastructure; and the known local hardware measurements. Their maintenance condition is preservation of existing tests, provenance, and fail-closed behavior.

## 3. Experimental requirements

The table records the exact remediation disposition for every Phase 3L.5 experimental requirement. “Actionable” means planning/verification can proceed now; it does not authorize data acquisition, training, or implementation in this phase.

| Experimental requirement | Current evidence / exact gap | Categories | Earliest resolving phase | Disposition | Reopening / promotion condition | Actionable now? |
|---|---|---|---|---|---|---|
| Declared input modalities | Contracts cover S2, S1 and RGB pairs; only fixed scene references are resolver-supported. | APPLICATION, PROVENANCE, VALIDATION | Source-admission phase | REMAIN_EXPERIMENTAL | Each admitted external source has validated modality, identity, preprocessing and provenance. | Yes: define admission evidence. |
| Optical / multispectral understanding | Real-S2 learned projector POC exists; no semantic benchmark. | MODEL, ADAPTER, BENCHMARK, EVALUATOR | S2 evaluation phase | REMAIN_EXPERIMENTAL | Held-out, leakage-audited VQA/caption evaluation with official evaluator/result. | Yes: specify evaluation protocol. |
| SAR adaptation | VV/VH projector is structural only. | DATASET, SUPERVISION, MODEL, ADAPTER, LICENSE | SAR source-verification phase | BLOCKED | SAR reopening conditions in section 7 all pass. | No: blocked by source evidence. |
| Bi-temporal analysis | Contract plus real RGB-pair execution exists. | DATASET, MODEL, TEMPORAL_SEMANTICS, VALIDATION | Temporal source-verification phase | REMAIN_EXPERIMENTAL | Valid temporal source and held-out task evaluation. | Yes: verify candidate metadata. |
| Single-image VQA | RGB smoke and S2 POC; no RSVQA metric. | DATASET, BENCHMARK, EVALUATOR, VALIDATION | RSVQA readiness phase | BLOCKED | Verified RSVQA release, evaluator, held-out result and provenance. | No: benchmark path blocked. |
| Captioning / scene description | RGB smoke and S2 POC; no VRSBench metric. | DATASET, BENCHMARK, EVALUATOR, VALIDATION | VRSBench readiness phase | BLOCKED | Verified VRSBench geometry/release/evaluator and result. | No: benchmark path blocked. |
| RS-adapted VLM | Frozen Qwen + S2 16×2048 post-vision projector. | MODEL, ADAPTER, BENCHMARK, VALIDATION | S2 evaluation phase | REMAIN_EXPERIMENTAL | Held-out task results demonstrate bounded claimed modality/task; no native-12-band claim without a native encoder. | Yes: evaluation design only. |
| Provenance | Pipeline 3 is strong; blocked external branches lack complete records. | PROVENANCE, DATASET, LICENSE | Per-source verification | REMAIN_EXPERIMENTAL | Required source/dataset/revision/IDs/modality/representation/adapter/model/preprocessing/device/dtype/execution fields are complete for each executed branch. | Yes. |
| Reproducibility | Pipeline 3 fingerprints, manifests, receipts and seeds exist. | PROVENANCE, SPLIT, VALIDATION | Per-branch execution gate | REMAIN_EXPERIMENTAL | Reproducible source acquisition, pinned revisions/hashes, deterministic preprocessing, split fingerprints and independently rerunnable evaluation. | Yes. |
| Leakage prevention | Scientific split and S2 train/validation image separation verified. | SPLIT, LEAKAGE, DATASET | Per-dataset verification | REMAIN_EXPERIMENTAL | Image/scene/pair/annotation duplication and geographic/near-duplicate audits where source metadata permits. | Yes: define audit inputs. |
| Security | API facade filters path/token fields; no complete security review evidence. | VALIDATION, APPLICATION, PROVENANCE | Release validation | REMAIN_EXPERIMENTAL | Secret, model-cache, filesystem and error-exposure review with test evidence. | Yes. |
| Testing | Focused tests exist; full suite is inconclusive. | VALIDATION | Full-suite gate | REMAIN_EXPERIMENTAL | One recorded complete pytest terminal summary plus focused regressions for each added branch. | Yes. |
| Learned S2 projector POC/scaling | 100/50 reproduced; 250/100 and 500/100 language-loss scaling; no test metric. | MODEL, ADAPTER, BENCHMARK, SPLIT, LEAKAGE | S2 benchmark phase | REMAIN_EXPERIMENTAL | Predeclared held-out evaluation, official metric/evaluator, result, and complete resource/provenance receipt. | Yes: protocol and holdout audit. |
| S1 acquisition/linkage | Fixture and projector/linkage tests exist; no lawful real SAR-language record crossed the adapter. | DATASET, SUPERVISION, LICENSE, PROVENANCE | SAR source-verification phase | BLOCKED | Verified licensed, pinned record with IDs, VV/VH semantics and split, then bounded structural adapter check. | No: source gate blocks it. |

## 4. Blocked requirements and reopening conditions

| Blocked requirement | Reason and evidence already available | Missing evidence / work | Categories | Priority | Earliest resolving phase | Reopen only when |
|---|---|---|---|---|---|---|
| Optical-SAR paired analysis | OSVQA paper verifies intended joint supervision; architecture is documented; controller blocks joint claims. | Official release, licenses, immutable data pin/hashes, record pair IDs, modality/preprocessing, co-registration evidence, splits and leakage audit; then fusion/model/evaluation. | DATASET, SUPERVISION, LICENSE, PROVENANCE, SPLIT, LEAKAGE, SPATIAL_SEMANTICS, MODEL, ADAPTER, BENCHMARK | BLOCKING | Joint-source verification | Every source/release and pair gate in section 8 passes. |
| Change reasoning | T1/T2 contract and RSRCC execution exist; branch frozen at execution. | Training-ready temporal source, model, evaluator and held-out result. | DATASET, SUPERVISION, TEMPORAL_SEMANTICS, MODEL, EVALUATOR, BENCHMARK | BLOCKING | Temporal-source verification | All temporal source gates in section 9 pass. |
| Learned grounding | Phase 3K.2 contract validates/refuses unsafe output; Phase 2F retains geometry. | Licensed, pinned geometry source, coordinate semantics/dimensions/linkage, split/leakage audit, grounding model and evaluator. | DATASET, SUPERVISION, SPATIAL_SEMANTICS, LICENSE, PROVENANCE, SPLIT, LEAKAGE, MODEL, EVALUATOR | BLOCKING | Grounding-source verification | All geometry/source gates in section 10 pass. |
| RSVQA evaluation | Interface is an explicit future hook; RGB/S2 bounded paths exist. | Verified release/license/splits/provenance; model-task contract; evaluator; held-out result. | DATASET, BENCHMARK, EVALUATOR, LICENSE, SPLIT, PROVENANCE | BLOCKING | RSVQA readiness | Dataset, model, evaluator, execution, result all become AVAILABLE. |
| VRSBench evaluation | Grounding audit found unresolved geometry/release conditions; no grounder/evaluator/result. | Valid geometry source and mapping semantics before grounding model/evaluator/result. | DATASET, BENCHMARK, EVALUATOR, SPATIAL_SEMANTICS, LICENSE, SPLIT, LEAKAGE | BLOCKING | Grounding-source verification | Geometry and VRSBench release gates pass, then grounding is evaluated. |
| CDVQA evaluation | CDVQA has audit-level supervision evidence only; no usable imagery/evaluator/model. | Source release, temporal record metadata, model and official evaluator/result. | DATASET, BENCHMARK, EVALUATOR, TEMPORAL_SEMANTICS, LICENSE, SPLIT, PROVENANCE | BLOCKING | Temporal-source verification | Temporal data gate and evaluation implementation pass. |
| BigEarthNet spatial annotation mapping | Source geometry is preserved as `SOURCE_GEOMETRY_PRESERVED`, `MAPPING_UNVERIFIED`. | Publisher tuple/origin/axis/normalization/edge semantics plus exact reference grid, crop/resample and S1/S2/SatQuery transform; validate against records. | SPATIAL_SEMANTICS, PROVENANCE, VALIDATION | BLOCKING for BigEarthNet grounding only | Publisher-semantics verification | Authoritative definition and record/raster verification pass; Phase 2F must not be relaxed. |

## 5. Unverified and deferred requirements

### Hidden ISRO/SAC final evaluation — UNVERIFIED

No access, data, protocol, or result is claimed. To change this only to **VERIFIED**, obtain documented authorization and a controlled evaluation agreement specifying: dataset availability; permitted access; immutable version/manifest and checksums; files/format; optical/SAR/temporal modality and band/value/preprocessing requirements; scene/pair IDs; spatial correspondence/co-registration evidence when paired; task labels and evaluator; provenance fields; split/hidden-test handling; output submission/receipt rules; and whether model weights, predictions, logs, and intermediate artifacts may be retained. Hidden labels must not be used for model selection, training, calibration, source admission, or design decisions.

### Deferred optical-SAR fusion architecture — DEFERRED

No fusion mechanism is selected intentionally. It should remain deferred until a joint source passes all reopening gates. It becomes actionable only after those gates; selection then depends on verified data/interface evidence, not on existing separate modality outputs.

## 6. SAR remediation

**Current facts.** Real S1 fixture/acquisition-linkage evidence, exact canonical `[VV,VH]` `[2,120,120]` contract, deterministic normalization, and an untrained 16×2048 SAR projector exist. SARLANG-1M, SAR-TEXT, FSAR-Cap, SAREval, and OSVQA were audited. SARLANG-1M has SAR-language evidence and a pin-able visible commit, but its corpus/upstream-image license and record-level modality/linkage/split evidence are incomplete. Other candidates have one or more weak source pinning, license, split, linkage, polarization, or format gaps. No optical-label transfer is a proposed path.

**Minimum evidence to reopen SAR training:** one authoritative SAR-language source must provide (1) explicit image, annotation, and derivative-use license; (2) immutable release revision and archive/per-file hashes; (3) official train/validation/test split manifest; (4) exact image and annotation IDs plus linkage; (5) sensor, polarization, value scale, format, preprocessing and canonical VV/VH conversion evidence; (6) bounded licensed sample route; (7) image/scene/geographic/near-duplicate leakage audit; and (8) source, split, adapter, model, preprocessing, device/dtype and execution provenance. Only then may a bounded untrained adapter test precede separately authorized training and held-out SAR VQA/caption evaluation.

## 7. Optical-SAR remediation

**Current facts.** The architecture has a conceptual two-branch-to-future-fusion path only. OSVQA verifies a paper-level intended aligned optical-SAR VQA task, but there is no official dataset release, license-bearing manifest, immutable data pin, actual pair linkage, record-level modality specification, co-registration transform/residual evidence, split, or leakage audit.

**Reopening condition:** an author-published, license-bearing, version-pinned OSVQA (or equivalent) release must provide optical/SAR bytes, pair/optical/SAR/annotation IDs, modality/preprocessing metadata, source hashes, official train/validation/test partitions, co-registration evidence, and image/pair/scene/geographic leakage audit. Then validate a small pair set against source-specific adapters before choosing, implementing, training, or evaluating a fusion mechanism. No joint model is authorized beforehand.

## 8. Temporal remediation

**Execution is not learning.** `TemporalInput` and ordering/correspondence contracts are implemented. Phase 3J.5 demonstrated Qwen FP16 execution over one RSRCC RGB pair with retained provenance. This is not change detection, learned temporal reasoning, change-VQA accuracy, spatial registration, multispectral temporal reasoning, or SAR temporal reasoning.

**Source status.** CDVQA has design-level change supervision but lacks verified image release/linkage metadata. ChangeChat/LEVIR has supervised RGB records and some license/pin evidence but lacks complete timestamp/registration/leakage evidence and bounded fixture access. RSRCC supplies revision-pinned RGB before/after fixture pairs, semantic ordering and exact pair keys, but lacks source-issued annotation IDs, dates/intervals, sensor/source-scene metadata, reproducible registration and geographic/near-duplicate audit. Thus it remains an execution fixture, not training evidence.

**Reopening condition:** one source must supply authoritative T1/T2 bytes; ordered pair and source-issued annotation IDs; explicit change supervision; dates/intervals; correspondence and reproducible registration evidence; sensor/preprocessing provenance; image/annotation licenses; immutable pin/hashes; reproducible bounded access; official train/validation/test protocol; and exact-pair, geographic and near-duplicate leakage audit. Only then can a temporal model, CDVQA evaluator and held-out result be considered.

## 9. Grounding remediation

The grounding contract supports point, bbox, polygon and mask validation but emits no learned geometry. Phase 2F is authoritative: BigEarthNet source geometry is preserved but unmapped, and it must not be converted to pixel/token/learned-grounding evidence.

VRSBench, OPT-RSVG, GeoChat, GeoGround/refGeo and GeoPixelD have each failed at least one necessary gate: authoritative coordinate serialization/dimensions, complete image–text–geometry linkage, license chain, immutable release/archives, official split/leakage evidence, or bounded sample route. GeoPixelD additionally requires separately prepared imagery. None authorizes training or mapping.

**Reopening condition:** select one source only after it provides a license-bearing immutable image/annotation release, complete IDs and linkage, supported geometry type with authoritative coordinate space/origin/axis/bounds and image dimensions, source transforms, official partitions, leakage audit, per-file/annotation hashes, and bounded reproducible sample access. Validate records fail-closed; then implement and evaluate a learned grounder. BigEarthNet mapping remains independently blocked until publisher semantics are resolved.

## 10. Benchmark remediation

| Requirement | Dataset | Model | Evaluator | Split / leakage | Provenance / license | Implementation / result | State |
|---|---|---|---|---|---|---|---|
| RSVQA | BLOCKED: no verified local release | PARTIAL: bounded RGB/S2 paths | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| VRSBench | PARTIAL: audited but geometry/release gate unmet | BLOCKED: no grounder | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| CDVQA | PARTIAL: audit-level annotations, image release blocked | BLOCKED: no learned temporal model | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

For each benchmark, the only resolving sequence is verified source → task/model input contract → evaluator implementation → held-out execution → provenance-bearing result. Annotation existence alone is not dataset availability or benchmark completion.

## 11. Full-suite validation

The state remains **FULL_SUITE_INCONCLUSIVE**. Historical reports contain several focused or full-suite attempts, but Phase 3L.5 found no current final complete pytest terminal summary that may be converted to a pass. Release confidence requires an isolated, recorded full `python -m pytest -q` run (using a writable explicit base temp directory if needed), exit code zero, complete collected/passed/skipped/failed terminal summary, environment/dependency identity, command, timestamp, and preservation of logs/receipt. It must be rerun after each future model/data/evaluator admission, with focused regressions for the affected branch.

## 12. Dependency graph

```text
SAR VQA / captioning
  -> licensed pinned SAR-language source + linkage + VV/VH semantics + splits/leakage
  -> bounded source validation and provenance
  -> SAR adapter execution
  -> authorized training
  -> held-out SAR evaluation

Optical-SAR reasoning
  -> licensed pinned joint release + pair linkage + modality metadata + co-registration + splits/leakage
  -> bounded pair validation
  -> fusion selection and implementation
  -> authorized training
  -> held-out joint evaluation

Temporal / CDVQA
  -> licensed pinned T1/T2 source + IDs + temporal metadata + registration + splits/leakage
  -> bounded temporal validation
  -> temporal model
  -> CDVQA evaluator
  -> held-out change result

Grounding / VRSBench
  -> licensed pinned geometry source + coordinate semantics/dimensions + linkage + splits/leakage
  -> fail-closed source-record validation
  -> learned grounding model
  -> VRSBench evaluator
  -> held-out grounding result

RSVQA
  -> verified RSVQA release + task contract + splits/provenance
  -> evaluator
  -> held-out VQA result

Hidden ISRO/SAC
  -> authorization + documented hidden protocol/data contract
  -> controlled no-leakage evaluation
  -> receipt-bearing final result
```

## 13. Blocking dependencies and current actionability

**FOUNDATIONAL:** authoritative licenses, immutable source revisions/hashes, IDs/linkage, modality semantics, coordinate/temporal semantics, official splits, and leakage evidence.

**BLOCKING:** the four source gates (SAR, joint, temporal, grounding), benchmark datasets/evaluators, BigEarthNet publisher spatial semantics for its grounding use, hidden ISRO/SAC authorization/protocol, and full-suite evidence.

**DEPENDENT:** adapter validation, fusion/model implementation, training, calibration, evaluation, and application capability admission. These must remain unstarted until their prerequisite source gate passes.

**OPTIONAL:** security depth and exhaustive provenance completion improve release confidence but cannot substitute for mandatory benchmark/capability evidence.

**DEFERRED:** optical-SAR fusion architecture selection remains deferred by design until a valid joint source exists.

Currently actionable planning/verification items are: establish source-admission checklists; define benchmark evaluator acceptance criteria; enumerate required provenance schemas; predefine leakage-audit inputs; create a reproducible full-suite receipt procedure; and obtain hidden-evaluation authorization/protocol if the owner permits. Currently unresolvable without external authority or state change are source licenses/releases, publisher coordinate semantics, hidden ISRO/SAC access, and unavailable data artifacts.

## 14. Research-readiness exit criteria

**RESEARCH_READY_WITH_DOCUMENTED_LIMITATIONS** requires every original mandatory requirement that is in scope for the claimed release to have implementation and validation evidence, with any explicitly permitted non-mandatory limitation documented, bounded, provenance-bearing, and fail-closed. At minimum this requires cleared applicable data gates; model/evaluator/held-out results for RSVQA, VRSBench and CDVQA; validated SAR, joint, temporal and grounding claims or an original-specification-authorized exclusion; hidden ISRO/SAC handling as required by its protocol; reproducibility/leakage/provenance evidence; and a conclusive full suite.

**RESEARCH_READY** requires the preceding evidence for all original mandatory capabilities without relying on exclusions: validated benchmark results, all claimed modality/task models, source and split safety, spatial and temporal semantics, provenance/reproducibility, security validation, and hidden final evaluation when access/protocol makes it mandatory. Neither status can be reached by reclassifying a smoke test, contract, fixture, or dataset paper as a completed capability.

## 15. Final summary

- **Total active gaps:** 22; **blocking gaps:** 7; **experimental gaps:** 14; **unverified gaps:** 1; **deferred item:** 1.
- The exact reopening conditions are in sections 4 and 6–10; their dependency order is section 12.
- Minimum work is source verification first, then bounded adapter validation, then authorized model/evaluator work, then held-out benchmark results, followed by complete leakage/provenance/security/full-suite validation.
- Scientific baseline verified unchanged: **65.0% accuracy, 4.1034286734 pp MAE, 9.5873312123 pp RMSE**.
- Current determination remains **NOT_RESEARCH_READY**. No commit or push was made.

## 16. Validation

Documentation consistency was checked against the Phase 3L.5 matrix and its cited phase reports. This planning-only phase makes no claim of a new test run or full-suite pass. `git diff --check` is the required final whitespace check.
