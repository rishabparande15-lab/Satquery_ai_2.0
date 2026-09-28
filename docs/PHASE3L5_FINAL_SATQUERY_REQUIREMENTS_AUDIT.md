# Phase 3L.5 — Final SatQuery System Requirements Audit

**Audit date:** 2026-09-24  
**Scope:** requirements-only audit of the current checkout. No production code, scientific artifact, dataset, split, model checkpoint, contract, or controller behavior was changed. Existing worktree changes predate this report and are not attributed to this audit.

## Determination

**NOT_RESEARCH_READY.** The frozen scientific baseline and bounded controller/API workflow have implementation evidence, but mandatory VQA/caption benchmarks, learned SAR and optical-SAR reasoning, learned temporal/change reasoning, learned grounding, and hidden ISRO/SAC evaluation are blocked or unverified. A contract, projector, fixture, or smoke generation is not treated as benchmark completion.

## Audit method and authoritative inputs

The audit read the original repository baseline and current status in [README.md](../README.md) and [CURRENT_PROJECT_STATUS.md](CURRENT_PROJECT_STATUS.md), architecture/freeze documents, all Phase 2A–2H and Phase 3A–3L.4 reports named in the Phase 3L.5 request, and the live `src/`, `tests/`, and referenced artifacts. The latest phase reports are used to refine older README statements where they describe later implemented bounded work.

Status meanings are exactly: **IMPLEMENTED**, **EXPERIMENTAL**, **BLOCKED**, **UNVERIFIED**, and **DEFERRED**. “Implemented” is bounded to the stated evidence and is not a claim that adjacent research capabilities are complete.

## Requirement-by-requirement matrix

| # | Requirement | Status | Current evidence and implementation location | Validation evidence | Known limitation / remaining dependency |
|---:|---|---|---|---|---|
| 1 | Project objective: reproducible BigEarthNet scientific scene analysis | IMPLEMENTED | Frozen 19-class `physical_62d + joint_croma_gap_768d -> hybrid_830d` flow; `src/run_pipeline.py`, `src/hybrid_fusion.py`. | Saved 5,000-area inference artifact. | This is not a general VLM system. |
| 2 | Declared input modalities | EXPERIMENTAL | Optical/S2, S1 VV/VH, RGB pair contracts exist in `src/eo_vlm/`; scene resolver supports Pipeline 3 and RSRCC references. | Contract and resolver tests. | No generic validated external-scene ingestion; modality reasoning varies by branch. |
| 3 | Optical / multispectral understanding | EXPERIMENTAL | Canonical 12-band S2 projector in `src/eo_vlm/multispectral_projector.py`. | Phase 3G/3G.1 real-S2 POC and scaling receipts. | Native multispectral VLM understanding and benchmark validation are not established. |
| 4 | SAR adaptation | EXPERIMENTAL | Deterministic VV/VH normalization and `SARProjector` in `src/eo_vlm/sar_projector.py`. | Structural projector tests. | No licensed, pinned SAR-language training/evaluation data; no learned SAR-language claim. |
| 5 | Optical-SAR paired analysis | BLOCKED | Pair/fusion path is architecture-only; controller blocks `OPTICAL_SAR_REASONING`. | Phase 3I.1 OSVQA audit and controller tests. | No releasable OSVQA records, joint fusion, trained model, or evaluation. |
| 6 | Bi-temporal analysis | EXPERIMENTAL | `TemporalInput`/ordering/correspondence contracts and `TemporalRGBAdapter`. | Real RSRCC RGB-pair CUDA receipt. | RGB execution does not establish learned temporal analysis. |
| 7 | Change reasoning | BLOCKED | Controller routes change requests to `CHANGE_VQA` blocked result. | Phase 3J.6 freeze; controller tests. | No source clears timestamp, registration, annotation-ID, and leakage gates. |
| 8 | Single-image VQA | EXPERIMENTAL | Qwen RGB adapter in `src/eo_vlm_adapter.py`; S2 is separate POC. | One controlled RGB CUDA generation; S2 language-loss POC. | No RSVQA result or task-quality validation. |
| 9 | Captioning / scene description | EXPERIMENTAL | RGB Qwen generation and S2 caption capability registration. | RGB smoke generation and S2 POC. | No VRSBench evaluator/result or caption-quality study. |
| 10 | Learned grounding | BLOCKED | Contract-only adapter deliberately raises unavailable in `src/eo_vlm/grounding_contracts.py`. | Grounding-contract tests; Phase 3K.1–3 gate. | No cleared geometry/data/supervision/evaluation source. |
| 11 | RS-adapted VLM | EXPERIMENTAL | Frozen Qwen plus learned S2 post-vision projector. | Phase 3G/3G.1 loss POC, pinned Qwen revision. | It is not a native 12-band encoder or benchmark-proven RS VLM. |
| 12 | BigEarthNet.txt adaptation role | IMPLEMENTED | `BigEarthNetTextAdapter` and artifacts in `src/image_language_foundation.py`, `artifacts/image_language/bigearthnet_txt/`. | Phase 2E audit and tests. | Foundation data view only; not a completed VQA/caption benchmark. |
| 13 | RSVQA evaluation | BLOCKED | `RSVQAAdapter` is an explicit future hook. | Phase 2E reports it raises `NotImplementedError`. | Dataset/evaluator/result are absent. |
| 14 | VRSBench evaluation | BLOCKED | `VRSBenchAdapter` is an explicit future hook. | Phase 2E and 3K audits. | Dataset/evaluator/result are absent; geometry evidence is unresolved. |
| 15 | CDVQA evaluation | BLOCKED | CDVQA was audited; no evaluator/model execution exists. | Phase 3J.1 and 3J.6. | Image release/linkage, temporal metadata, and evaluation execution are blocked. |
| 16 | Hidden ISRO/SAC final evaluation | UNVERIFIED | No hidden-data manifest, access record, evaluator, or result was found. | README/current-status limitation statements. | Access/readiness cannot be claimed without the hidden data and protocol. |
| 17 | Agentic controller | IMPLEMENTED | `src/agent/controller.py`, capability registry, tools, trace, reconciler, answer assembler. | Phase 3L.1–3L.2 real bounded flows and controller tests. | It only executes registered, validated tool boundaries. |
| 18 | Query understanding | IMPLEMENTED | Typed task vocabulary and deterministic planner in `src/agent/query_types.py`, `planner.py`. | Agent workflow/controller tests. | Keyword/declaration routing, not open-ended semantic planning. |
| 19 | Task planning | IMPLEMENTED | `DeterministicPlanner` selects one declared task and preserves blocks. | Planner/controller tests. | No multi-step autonomous research planner. |
| 20 | Input validation | IMPLEMENTED | Request normalization, scene resolver, modality/shape contracts. | Resolver, temporal, S2/SAR and API tests. | External acquisitions remain outside the scene-reference workflow. |
| 21 | Representation selection | IMPLEMENTED | Capability registry and task tool requirements select scientific/RGB/S2 representations. | Phase 3L tests. | Selection does not create unavailable representations. |
| 22 | Tool execution | IMPLEMENTED | Real frozen-scientific cache, RGB Qwen, temporal RGB, and S2 projector tools in `src/agent/real_tools.py`. | Phase 3L.2 bounded real executions. | SAR/joint/change/grounding tools are intentionally absent. |
| 23 | Evidence reconciliation | IMPLEMENTED | Typed evidence/reconciler/answer assembler in `src/agent/evidence.py`, `reconciler.py`, `answer_assembler.py`. | Evidence and controller conflict tests. | Reconciliation cannot validate a blocked capability. |
| 24 | Confidence handling | IMPLEMENTED | Unknown/model-provided/calibrated source semantics; numeric unknown confidence rejected. | `test_grounding_contracts.py`, agent tests. | No calibrated task confidence is produced. |
| 25 | Provenance | EXPERIMENTAL | Scientific receipts and controller provenance preserve source/reference, representation, adapter/model, device/dtype and execution ID where applicable. | Receipt/catalog and Phase 3L.2 tests. | Not every external dataset has revision, IDs, hashes, preprocessing, or acquisition records; field completeness is input-dependent. |
| 26 | Audit/execution summaries | IMPLEMENTED | `src/agent/workflow.py` returns trace, evidence, audit and execution summaries. | Phase 3L.3–3L.4 API tests. | Summary truth is bounded by underlying evidence. |
| 27 | Report generation | IMPLEMENTED | Deterministic controller report in `workflow.py`. | Agent workflow tests. | It is a structured execution report, not a research report generator over unverified claims. |
| 28 | Web/API application | IMPLEMENTED | Controller facade `src/agent/api_integration.py`; existing API/frontend renderer integration. | Phase 3L.3 API and Phase 3L.4 workflow tests. | Real-model smoke remains opt-in; unsupported branches are blocked. |
| 29 | Scene/input reference workflow | IMPLEMENTED | `scene_resolver.py` resolves Pipeline 3 IDs and RSRCC fixture pair IDs before controller execution. | Scene-resolver/workflow tests. | It is the core workflow; legacy upload endpoints do not establish arbitrary external-scene ingestion as a controller capability. |
| 30 | Model swappability | IMPLEMENTED | Model-neutral adapter/tool interfaces and capability registry. | Architecture and contract tests. | A swap needs its own data, execution, provenance, and evaluation gate. |
| 31 | Reproducibility | EXPERIMENTAL | Pipeline 3 has fingerprints, hashes, manifests, receipts, deterministic preprocessing/seed and split records. | Historical 5,000-area artifact; receipt/catalog tests. | Full system reproducibility is not established for blocked external datasets or all VLM paths. |
| 32 | Leakage prevention | EXPERIMENTAL | Image/group split policy and frozen Pipeline 3 split checks in `src/leakage_policy.py`. | Phase 2B, Phase 3G.1 zero train/validation image intersection. | Geographic, pair, annotation and external-dataset overlap are often unknown. |
| 33 | Spatial semantics | IMPLEMENTED | Typed coordinate/raster/token contracts in `src/spatial_contract.py`. | Phase 2B/2F tests and fail-closed audit. | BigEarthNet.txt source coordinates are not mapped to pixels/tokens. |
| 34 | Temporal semantics | IMPLEMENTED | Ordered T1/T2, correspondence and registration-status contracts in `temporal_contracts.py`. | Temporal contract tests and RSRCC receipt. | Timestamps/registration can remain unknown; no temporal learning follows. |
| 35 | Optical-SAR fusion architecture | DEFERRED | Future fusion alternatives are documented; no joint model is selected. | Phase 3I.1/3L blocked routing. | Requires released paired supervision, modality contracts and validated fusion evaluation. |
| 36 | Grounding architecture | IMPLEMENTED | Point/bbox/polygon/mask request/result validation, coordinate and provenance contract. | Phase 3K.2 structural tests. | Architecture cannot emit learned grounding or map unverified source geometry. |
| 37 | Security | EXPERIMENTAL | HTTP-safe facade filters path/token-named fields; `.env.example` avoids committed credentials. | API integration tests and source inspection. | No comprehensive penetration/security test, secrets scan receipt, or cache-access hardening proof found. |
| 38 | Local hardware feasibility | IMPLEMENTED | RTX 5060 Laptop (8,151 MiB), FP16 Qwen RGB and temporal pair execution measured. | Phase 3D.3/3J.5 receipts: RGB load 7,520,955,904 B allocated; temporal peak 7,802,813,952 B. | Only bounded batch-1 paths measured; SAR/fusion/large-context feasibility is unverified. |
| 39 | Checkpoint/provenance management | IMPLEMENTED | Qwen revision and checksums; S2 projector receipts; Pipeline 3 checkpoint/receipt catalog. | Phase 3D.2, 3G.1 and scientific regression artifacts. | SAR/temporal learned checkpoints do not exist; external data releases lack receipts. |
| 40 | Testing | EXPERIMENTAL | Focused tests cover contracts, adapters, controller/API, spatial, receipts and scientific artifacts. | Individual phase reports record focused passes. | Current audit did not find a single current complete-suite terminal pass; status remains `FULL_SUITE_INCONCLUSIVE`. |
| 41 | Qwen2.5-VL-3B-Instruct RGB execution | IMPLEMENTED | Pinned `66285546d2b821cf421d4f5eb2576359d3770cd3` adapter path. | Phase 3D.3 FP16 CUDA load and controlled RGB generation. | This is RGB smoke execution, not EO benchmark performance. |
| 42 | S2 12-band engineering contract | IMPLEMENTED | Exact `[N,12,120,120]` canonical contract and deterministic adapter path. | Phase 3F/3F.1 real raw-S2 execution and tests. | Engineering compatibility is distinct from learned semantic validation. |
| 43 | Learned S2 projector POC/scaling | EXPERIMENTAL | 5,549,184-parameter projector emits 16×2048 Qwen-compatible tokens. | Phase 3G.1 reproduced 100/50 and ran 250/100 and 500/100 with zero test records. | Loss only; no held-out VQA/caption metric or benchmark. |
| 44 | S1 acquisition/linkage | EXPERIMENTAL | S1 fixture/linkage audit and VV/VH projector exist. | Phase 3H.1 and SAR linkage/projector tests. | No approved real SAR-language sample has crossed the projector. |
| 45 | Evidence-type separation | IMPLEMENTED | `SCIENTIFIC_PREDICTION`, `MODEL_LANGUAGE_OUTPUT`, `SOURCE_METADATA`, `SOURCE_GEOMETRY`, `MODEL_GROUNDING`, `PIXEL_EVIDENCE`, `REGION_EVIDENCE` are separated by contracts. | Controller/evidence/grounding tests. | No unavailable evidence type is fabricated; source geometry remains non-grounding evidence. |
| 46 | BigEarthNet spatial annotation mapping | BLOCKED | Source geometry is retained with `SOURCE_GEOMETRY_PRESERVED` / `MAPPING_UNVERIFIED`. | Phase 2F and Phase 3K fail-closed tests. | Publisher coordinate semantics and source-to-raster/token transform are unresolved. |
| 47 | Frozen scientific baseline | IMPLEMENTED | 5,000 areas, 4,600/200/200 split, 830-D predictor. | `historical_checkpoint_inference.json`: accuracy 65.0%, MAE 4.1034286465 pp, RMSE 9.5873311030 pp; authoritative rounded baseline remains 65.0%, 4.1034286734 pp, 9.5873312123 pp. | These values are scientific coverage metrics, not VLM metrics. |

**Counts (47 requirements):** IMPLEMENTED 24; EXPERIMENTAL 14; BLOCKED 7; UNVERIFIED 1; DEFERRED 1.

## Status tally (authoritative)

To remove any ambiguity from the inline count, the statuses in the 47 table rows tally to:

| Status | Count |
|---|---:|
| IMPLEMENTED | 24 |
| EXPERIMENTAL | 14 |
| BLOCKED | 7 |
| UNVERIFIED | 1 |
| DEFERRED | 1 |
| **Total** | **47** |

## Required capability and benchmark distinctions

### Model / VLM

- **Qwen RGB execution: VALIDATED.** It is a bounded FP16 CUDA controlled-image generation through the pinned checkpoint, with provenance; it is not RSVQA/VRSBench validation.
- **S2 learned projector: VALIDATED POC.** Real 12-band S2 reaches the Qwen post-vision token interface; 100/50 reproduction and 250/100, 500/100 language-loss runs exist. **Native multispectral VLM understanding is not established.**
- **SAR projector: IMPLEMENTED engineering path.** `SARProjector` accepts canonical `[B,2,120,120]` VV/VH and emits 16×2048 tokens. **SAR language reasoning/training is BLOCKED.**
- **Optical-SAR architecture: DESIGNED/DEFERRED; learned joint reasoning: BLOCKED.** No independent modality answers are represented as joint reasoning.
- **Temporal RGB execution: demonstrated.** Qwen accepted a real RSRCC pair. **Learned temporal/change reasoning is BLOCKED.**
- **Grounding: BLOCKED.** There is no learned grounding result.

### Benchmarks

| Benchmark | Dataset available | Model available | Evaluator available | Evaluation implemented | Result available | Audit status |
|---|---|---|---|---|---|---|
| RSVQA | No verified local release | RGB/S2 bounded paths only | No | No | No | BLOCKED |
| VRSBench | No verified local release | No grounder | No | No | No | BLOCKED |
| CDVQA | Annotation-level evidence only; images/release blocked | No learned temporal model | No | No | No | BLOCKED |
| Hidden ISRO/SAC | UNVERIFIED | N/A | UNVERIFIED | No | No | UNVERIFIED |

## Evidence, provenance, reproducibility, leakage, and security findings

The controller preserves the required boundary: model-language output is not promoted to scientific prediction; temporal RGB language is not change evidence; source geometry is not learned grounding; and a numerical confidence without a non-unknown source is rejected. The source locations are `src/agent/evidence.py`, `src/agent/reconciler.py`, `src/agent/answer_assembler.py`, and `src/eo_vlm/grounding_contracts.py`.

Pipeline 3 provenance is the most complete: dataset/split fingerprints, cached-representation and checkpoint hashes, model/revision context, preprocessing/source records, device/dtype where executed, and receipt artifacts exist. The missing system-wide fields are chiefly external dataset revision/archive hashes, authoritative scene/image/annotation IDs, sensor and preprocessing metadata, exact source geometry transforms, acquisition records, and test/evaluation receipts for blocked branches.

Leakage evidence is adequate for the frozen scientific split and S2 POC image train/validation separation. It is not evidence of complete annotation-, pair-, scene-, geographic-, or near-duplicate audits for external SAR, joint, temporal, or grounding datasets. Phase 2F remains fail-closed across annotation → coordinate space → image dimensions → pixels → analysis grid → CROMA tokens → grounding/evidence.

The API controller facade removes dictionary fields named like paths or tokens and avoids exposing tool exceptions. This is useful implementation evidence, but it does not establish a complete security review of API/HF secrets, model-cache access, filesystem exposure, or error behavior under adversarial inputs.

## Critical summary

**A. Implemented:** frozen 5,000-area scientific prediction; representation/receipt infrastructure; typed spatial, temporal and grounding contracts; BigEarthNet.txt foundation; bounded Qwen RGB and temporal RGB execution; deterministic S2 engineering path; SAR projector; controller/planner/tool/evidence/report/API/frontend scene-reference workflow; baseline hardware measurements.

**B. Experimental:** RGB VQA/captioning only at smoke level; learned S2 adaptation POC/scaling; SAR engineering path and S1 linkage; temporal pair execution; system-wide provenance/reproducibility/leakage/security/testing coverage.

**C. Blocked by data/supervision:** SAR VQA training; optical-SAR learned fusion/reasoning; learned temporal/change reasoning and CDVQA; learned grounding; RSVQA and VRSBench completion; BigEarthNet source-geometry mapping.

**D. Unverified:** hidden ISRO/SAC data, evaluator, protocol, and result; any claim of full external-scene ingestion or full-system reproducibility.

**E. Intentionally deferred:** learned optical-SAR fusion architecture selection and all training downstream of the temporal branch freeze; no such deferral is treated as implementation.

**F. Scientific baseline:** unchanged. The authoritative baseline is **65.0% accuracy**, **4.1034286734 pp MAE**, and **9.5873312123 pp RMSE**. The checked historical artifact differs only at documented floating-point precision (4.1034286465 and 9.5873311030).

**G. Current application capabilities:** a user supplies a resolvable Pipeline 3 scene ID or RSRCC fixture pair plus query; resolver → controller → registered tools → typed evidence → reconciliation → answer → summaries/report. Scientific and RGB are supported; temporal RGB is execution-only; S2 remains POC/inconclusive for full auditable generation; blocked tasks return `Fallback: NONE`. Arbitrary image upload is not the core controller architecture.

**H. Exact blockers:** license-bearing immutable SAR release with raw modality/linkage/splits; official OSVQA release with per-pair records and geometry; temporal source with IDs, timestamps, registration and leakage evidence; grounding source with geometry semantics, licenses, pinning/linkage/splits; benchmark datasets/evaluators; authoritative BigEarthNet text-geometry mapping; hidden ISRO/SAC access and protocol.

**I. Minimum remaining work, in dependency order:**

1. **Data/supervision:** clear one documented source per blocked SAR, joint, temporal and grounding gate, including license, immutable manifests/hashes, linkage, split and leakage evidence.
2. **Model adaptation:** only after those gates, validate source-specific S1/joint/temporal/grounding adapters and train/evaluate the appropriate learned models without relaxing Phase 2F.
3. **Benchmarks:** implement official RSVQA, VRSBench and CDVQA evaluators; execute held-out results; obtain and execute the hidden ISRO/SAC protocol when authorized.
4. **Application:** admit only validated source-reference workflows and tools; do not add fallback substitutions for blocked tasks.
5. **Validation:** complete model-specific calibration, provenance field coverage, security review, leakage audits and a conclusive complete-suite run.

## Audit validation

- Documentation consistency/read-only source and artifact checks: performed.
- Broad focused test attempt: the listed controller/API/evidence/adapter test selection emitted only partial progress and no terminal pytest summary, so it is **INCONCLUSIVE** and is not counted as a pass.
- Confirmed focused regression: `python -m pytest -q tests/test_agent_controller.py --basetemp=.pytest-phase3l5-single` — **7 passed in 0.09s**.
- No-write syntax compile check: Python `compile()` over every `src/**/*.py` file — **PASS**.
- Whitespace check: `git diff --check` — **PASS** (existing CRLF warnings only).

No full-suite pass is claimed without a terminal summary. No commit or push was made.
