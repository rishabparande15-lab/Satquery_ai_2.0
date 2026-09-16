# Phase 6R.3 — Production Architecture Repair and Connectivity Validation

## Verdict

GREEN. The real train, validation, and test API requests returned HTTP 200 through the Capability Registry. Exact identities, raster content, representation dimensions, checkpoint identity, evidence separation, TaskResult serialization, and frontend compatibility were validated without changing the dataset, labels, feature definitions, CROMA, checkpoint, training, or evaluation.

## 1. Problem diagnosis

The pre-edit trace identified five concrete defects:

1. `src/api.py::Handler.do_POST` called `analysis_engine.run_analysis` directly, bypassing `CapabilityRegistry`.
2. `src/data_orchestrator.py::DataOrchestrator.retrieve` and `/api/samples` used non-strict discovery. That parser reduced BigEarthNet identities to row/column suffixes such as `07_47`, so distinct acquisitions could collide and the real full identity was not selectable.
3. `src/deterministic_scene_analysis.py::Pipeline3AnalysisAdapter` wrapped only the legacy representation runner. It did not execute the validated 830D checkpoint.
4. `src/architecture_contracts.py::adapt_run_analysis_output` carried legacy features/evidence into `TaskResult`, but the legacy result contained no trained prediction.
5. CROMA tokens were correctly represented as `[1,225,768]`, but the trained head is scene-level. There is no trained `[225,19]` output and one must not be synthesized.

## 2. Root causes

The 422 was a connectivity failure rather than a scientific failure: the HTTP route entered the suffix-based local discovery path instead of the exact 5,000-area manifest path. The capability abstraction existed but was not the API entry point. The frozen scientific probe and its provenance existed only in experiment tooling and were not connected to the production adapter.

## 3. Identity-resolution design

`Pipeline3ManifestResolver` uses the existing frozen 5,000-area manifest. It accepts only an exact `area_id`, `s2_identity`, or `s1_identity`. It then verifies that all three authoritative identities on the resolved row are unique.

For the raster bundle it requires exactly one of each:

- twelve S2 bands from the row's declared `s2_source_roots`;
- S1 VV and VH members whose complete names contain the row's exact `s1_identity`;
- one reference member whose complete name contains the exact `area_id`.

Every selected raster is checked against the manifest content hash. CRS, extent, orientation, masks, finiteness, the canonical B02 `120x120` grid, S1 alignment, and integer reference codes are validated. Zero matches and multiple matches are explicit errors. There is no fuzzy match, suffix match, or first-result fallback.

The legacy small fixture can still use its suffix parser because it has no metadata table. `DataOrchestrator` now automatically enables metadata-backed strict discovery when `metadata.parquet` exists.

## 4. API execution path before and after

Before:

`POST /api/analyze -> validate_request -> analysis_engine.run_analysis -> HTTP`

After:

`POST /api/analyze -> TaskRequest -> CapabilityRegistry -> DeterministicSceneAnalysisCapability -> Pipeline3AnalysisAdapter -> Pipeline3 scene probe -> evidence adapter -> interpretation adapter -> TaskResult -> HTTP`

The `api.run_analysis` symbol is retained only as a fault-injection compatibility seam and points to the registry dispatcher. The API no longer imports or directly calls the legacy `analysis_engine.run_analysis` function.

## 5. Capability Registry integration

The API constructs `TaskRequest(task_type="deterministic_scene_analysis")` and invokes `run_deterministic_scene_analysis_with_context`. That function resolves the task through `CAPABILITY_REGISTRY.require_task_type`. The HTTP response includes `capability_route`, canonical `task_result`, and `scene_contract` alongside legacy-compatible top-level fields.

A regression test replaces the API dispatch seam, asserts that it receives a `TaskRequest`, and verifies the authoritative registry route marker. Existing capability tests also prove missing capabilities and unsupported task types fail explicitly.

## 6. Model integration

The production inference contract is:

`physical_62D + joint_croma_gap_768D = hybrid_830D -> validated linear probe -> scene_prediction_19D`

The implementation recomputes physical and CROMA representations from the exact rasters. It validates:

- physical: `[62]`;
- optical CROMA tokens: `[1,225,768]`;
- SAR CROMA tokens: `[1,225,768]`;
- joint CROMA tokens: `[1,225,768]`;
- joint CROMA GAP: `[1,768]`;
- hybrid: `[830]`;
- probe output: `[19]`.

The unchanged checkpoint is accepted only if its file SHA-256 is `0694dd82d4a3c03663ed7128adb4be6df729178e07b4ff8a1ec4295ed912130f`, its state SHA-256 is `85d9390c66a887276db24a7cadb58c398770cde2e17488a1a9c42e16819da634`, its input dimension is 830, and its dataset fingerprint matches the frozen experiment.

## 7. Scene-versus-spatial distinction

The current validated model is a scene-level 830D→19 classifier/probe.

The system does NOT yet contain a trained 225×19 spatial prediction head.

The `[225,768]` CROMA tensors remain token-level latent representations. The `[19]` probe output is stored with `level: "scene"` and `spatial_prediction: false`. It is never broadcast across tokens, inserted into token predictions, called segmentation, or presented as grounding.

## 8. Evidence integration

The model output and evidence are deliberately separate:

- `model_results.prediction` contains the scientific scene vector and verified checkpoint identity;
- `spatial_evidence` contains deterministic physical-feature evidence regions;
- `interpretation` contains only allowlisted claims derived from that deterministic evidence.

The model vector is marked `semantic_interpretation: "not_semantically_interpreted"`. No class index is converted into a prose claim. Its calibration status is unavailable and its values are not called calibrated confidence.

## 9. TaskResult structure

`TaskResult.output` contains task metadata, input identity, representation references, model output, features, evidence, controlled interpretation, warnings, and explicit unavailable confidence. Provenance independently records input identities, preprocessing, representations, execution trace, evidence provenance, dataset fingerprint, split fingerprint, and checkpoint hashes.

`RepresentationType.SCENE_PREDICTION` records the model result as a scene-level `[19]` representation. Token CROMA references remain token-level `[1,225,768]` representations.

## 10. Real-data API tests

The real `ThreadingHTTPServer` handler was called with exact manifest identities:

| Split | Exact area | HTTP | Resolved split | TaskResult | Dimensions |
|---|---|---:|---|---|---|
| train | `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_47_85` | 200 | train | success | `62 + 768 = 830 -> 19` |
| validation | `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_70_71` | 200 | validation | success | `62 + 768 = 830 -> 19` |
| test | `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_39_48` | 200 | test | success | `62 + 768 = 830 -> 19` |

All three returned exact requested/resolved identity equality, authoritative registry routing, available deterministic evidence, controlled interpretation, `spatial_prediction: false`, unavailable calibration, and the verified checkpoint hash.

## 11. End-to-end trace

The test-area trace recorded:

`manifest dataset + full area identity -> exact S2/S1/reference identities -> content/grid validation -> physical [62] -> optical/SAR/joint CROMA [1,225,768] -> joint GAP [1,768] -> hybrid [830] -> scene output [19] -> deterministic evidence -> controlled interpretation -> TaskResult -> HTTP 200`

## 12. Frontend compatibility

The HTTP response preserves the existing top-level report fields. The UI now displays a separate “Scene model output” metric, describes the output as scene-level, and explicitly says it is not a spatial prediction. System capability text distinguishes the available scene probe from the unavailable 225×19 spatial head. No redesign was performed.

## 13. Failure handling

Regression tests cover:

- ambiguous authoritative BigEarthNet identity;
- the known suffix-collision behavior (`07_47`) and rejection of suffix-only lookup;
- missing area;
- missing S2 band;
- missing S1 archive;
- malformed HTTP request (existing suite);
- unsupported task;
- registry missing capability;
- checkpoint mismatch;
- invalid representation dimensions.

Identity and model-contract failures use distinct 422 error codes. Unexpected errors remain sanitized 500 responses and timeouts remain controlled 408 responses.

## 14. Regression tests

Focused architecture/API tests after the repair: `107 passed`.

Frontend state tests: `7 passed`.

The complete repository run produced `311 passed, 5 skipped, 1 failed`. The sole failure is an unrelated pre-existing working-tree inconsistency: the user-modified `experiments/pipeline3_5000/baseline_metrics.json` currently contains `status: "not_run"`, while the unchanged historical audit test expects `complete_validation_and_final_test`. This repair did not modify that scientific artifact. Running the suite without that audit module produced `307 passed, 5 skipped`; all Phase 6R.3, API, architecture, evidence, provenance, and Pipeline 3 implementation tests passed. Because the complete suite is not entirely green, no commit was created.

Repeated real inference for the held-out test area produced maximum absolute prediction difference `0.0` across two runs.

The Phase 6R.2B scientific values remain unchanged: MAE `4.10342865 pp`, RMSE `9.58733110 pp`, and dominant-class accuracy `130/200 = 65.0%`.

## 15. Remaining limitations

- Scene probabilities are not calibrated per-scene confidence.
- The probe output is not semantically interpreted by the evidence adapter.
- There is no trained token-level/pixel-level class head, segmentation, grounding, VQA, captioning, or temporal-change model.
- Uploaded imagery and the legacy three-sample fixture remain representation-only because they are outside the frozen exact-5,000 checkpoint identity contract.
