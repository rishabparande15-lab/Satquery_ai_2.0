# Pipeline Consolidation

Date: 2026-09-13  
Decision: Pipeline 3 is the canonical SatQuery architecture.  
Scope: audit, dependency separation, runtime clarification, and preservation of scientific history. No model, dataset, split, preprocessing, prediction, metric, evidence, interpretation, API, or frontend behavior was intentionally changed.

The forensic source of numbered pipeline identity is
`experiments/reports/autonomous_execution_report.json`. The former
`experiments/pipelines/README.md` used a conflicting Pipeline 1 label; that
index has been corrected. Classification below uses exactly:

- **A — required by Pipeline 3**
- **B — useful scientific/validation asset**
- **C — historical/experimental**
- **D — duplicate of Pipeline 3**
- **E — dead/unused**
- **F — uncertain; preserved until verified**

## 1. Pipeline 1 summary

Pipeline 1 is the completed So2Sat-LCZ42 S1/S2 FusionCNN experiment preserved in
`experiments/github_sen12ms_training` and described by the master execution
report. It is not the local BigEarthNet+CROMA path; that older label was stale.

| Concern | Forensic mapping |
|---|---|
| Entry points / CLI | `python -m experiments.github_sen12ms_training.train`, `.evaluate`, and `.infer [index]` |
| Main modules | `train.py` owns `H5Dataset`, `FusionCNN`, training, and metrics; `evaluate.py` and `infer.py` import it |
| Data flow | external So2Sat HDF5 `sen1` (8 channels) + `sen2` (10 channels) → float32/channel-first → two CNN branches → concatenation → 17-class logits |
| Model | locally trained dual-branch `FusionCNN`; cross-entropy; Adam; saved external checkpoint |
| Features / preprocessing | learned CNN activations; only dtype conversion and HWC→CHW transpose are explicit |
| Dataset | fixed external `D:\Satquery_ai datasets\github_datasets\So2Sat-LCZ42`; HDF5 train/validation/test |
| GEE / CROMA | none / none |
| API / frontend | no imports or routes; no frontend call reaches it |
| Tests | no current test imports these modules |
| Experiments / artifacts | external checkpoints, reports, predictions, metrics; summary values retained in `autonomous_execution_report.json` |
| Configuration | hard-coded external root plus `SO2SAT_DEVICE`; no `.env.example` setting |
| External dependencies | `h5py`, NumPy, PyTorch, scikit-learn; `h5py` is intentionally not promoted into canonical requirements |

Classification:

| Component | Class | Decision |
|---|---|---|
| Training/evaluation/inference scripts | C | Preserved in place, marked archive/reproduction-only; disconnected from production |
| Persisted summary metrics and artifact references | B | Retained as scientific history; external assets are not copied into Git |
| Fixed path/device configuration | C | Retained to reproduce the original run, not accepted as production configuration |
| Old index statement “Pipeline 1 = local BigEarthNet+CROMA” | D | Corrected; those shared components belong to canonical Pipeline 3 |

## 2. Pipeline 2 summary

Pipeline 2 is the GEE-only experiment. The numbered scaffold is
`experiments/pipelines/gee`; `gee_temporal` and `gee_temporal_classical` are
later real-GEE research extensions.

| Concern | Forensic mapping |
|---|---|
| Entry points / CLI | `python -m experiments.pipelines.gee.runner --mode auto|local|gee`; historical temporal runners use `--mode gee` |
| Main modules | numbered `config`, `data_access`, `preprocessing`, `reporting`, `runner`; temporal variants add authentication, acquisition, differences, masks, and reports |
| Data flow | numbered runner: GEE auth check → either labelled local BigEarthNet summaries/plots or an authenticated-but-no-query report. Temporal runners: configured AOI/dates → GEE collections/download → GeoTIFF validation → exploratory differences |
| Models | none; classical temporal thresholds are not learned models |
| Features / preprocessing | local summary statistics/display normalization; temporal differences, ratios, NDVI, and exploratory masks |
| Dataset | local three-sample BigEarthNet fallback; temporal variants require provider data and saved exports |
| GEE | `earthengine-api`; temporal retrieval also uses HTTP downloads |
| CROMA | none in Pipeline 2 |
| API / frontend | not imported by `src/api.py`, `analysis_engine.py`, or frontend code |
| Tests | no production or frontend tests invoke these runners |
| Experiments / artifacts | historical output roots below `experiments/outputs/real_gee`; tracked master execution summary |
| Configuration | `GEE_OUTPUT_ROOT`, `GEE_PIPELINE_MODE`; temporal `GEE_PROJECT`, `GEE_AOI_GEOJSON`, before/after variables |
| External dependencies | Earth Engine API, requests, Rasterio, Matplotlib, NumPy; credentials and data remain external |

Classification:

| Component | Class | Decision |
|---|---|---|
| `gee_status` / `select_mode` behavior used by the Pipeline 3 harness | A | Rehomed unchanged to `src/gee_experiment_runtime.py`; Pipeline 2 re-exports it for compatibility |
| Master report and real-GEE acquisition metadata | B | Retained outside production execution |
| Numbered GEE-only runner and visualization/reporting code | C | Marked archived/reproduction-only; no production routing |
| GEE temporal/classical runners | C | Preserved as historical experiments; explicit non-production banners added |
| `src/qa_runtime.py` audit of stored temporal exports | F | Retained because it also performs a useful API startup check and audits prior real-provider data; it is not imported by the server |

## 3. Pipeline 3 summary

Pipeline 3 is the selected GEE+CROMA lineage plus the validated feature,
scientific-evaluation, spatial-evidence, and constrained-interpretation work
built on its shared modules.

| Concern | Forensic mapping |
|---|---|
| Production entry | `python -m src.api --host 127.0.0.1 --port 8000` |
| Programmatic entry | `src.analysis_engine.run_analysis(request)`; `save_report` persists immutable JSON |
| Research entry | `python -m experiments.pipelines.gee_croma.runner --mode auto|local|gee`; canonical scientific CLIs are documented in the root README |
| API / frontend | `src/api.py`; `src/static/index.html`, `app.js`, `style.css`; `/api/analyze`, `/api/upload`, `/api/availability/search`, `/api/sources`, `/api/samples`, `/api/report/{id}`, `/api/health` |
| Planning / data | `query_interpreter.py`, `tool_selector.py`, `data_orchestrator.py`, `imagery_availability.py` |
| Validation / loading | `input_validation.py`, `raster_inputs.py`, `dataset_loader.py`, `multimodal_cube.py` |
| Preprocessing | established raw-raster alignment plus CROMA per-channel mean ± 2 standard deviations clipped to `[0,1]`; 120×120 canonical input |
| Representations | `croma_adapter.py`: optical, SAR, joint `[1,225,768]` encodings and `[1,768]` GAP vectors |
| Physical features | `modality_features.py` and `gee_features.py`: 52 optical, 9 SAR, or legacy joint 62-value schemas |
| Fusion / prediction | `hybrid_fusion.py`: web runtime creates a deterministic 192-value untrained representation and no prediction; trained heads exist only in controlled scientific workflows |
| Spatial evidence | `pixel_features.py`, `region_grounding.py`, `evidence_schema.py`; deterministic 15×15 tokens and four-connected regions |
| Interpretation | `interpretation_adapter.py`; allowlisted deterministic claims, no LLM/VLM |
| Typed scientific foundation | `phase1_foundation.py`, `phase2_integration.py`, `phase3_orchestration.py`, `phase3_5_end_to_end.py`, `phase3_6_benchmark.py` |
| Training/evaluation | `feature_cache.py`, `prepare_training.py`, `training_data.py`, `train_landcover.py`, `evaluate_landcover.py`, `hybrid_pipeline.py`, Pass 3/5/5D modules and scripts |
| Datasets | external BigEarthNet S1/S2/reference data and metadata; tracked dataset contracts/manifests only |
| GEE | live API performs availability queries only; numbered harness performs auth selection; historical `gee_croma_temporal` contains real acquisition evidence |
| Configuration | `.env`/`.env.example` through `src/config.py`: dataset, features, official CROMA source/checkpoint; live GEE discovery additionally uses `GEE_PROJECT` |
| External dependencies | requirements: Click, Einops, Matplotlib, NumPy, pandas, PyArrow, python-dotenv, Rasterio, PyTorch, Earth Engine API; browser code uses no package manager |
| Tests | all Python tests, especially app/QA, strict loader, CROMA/hybrid, Phase 1-3, Pass 3/5D, pixel/evidence/interpretation; `frontend_state.test.cjs` |
| Artifacts / docs | `experiments/pass3`, `pass5`, `pass5d`, `experiments/smoke`, `artifacts/pixel_feature_probe`, `spatial_evidence`, `interpretation`, and their docs/provenance |

## 4. Why Pipeline 3 is canonical

Pipeline 3 is the only lineage that combines the canonical S1/S2 contracts,
official frozen CROMA, physical and hybrid representations, controlled held-out
evaluation, spatial evidence, constrained interpretation, loopback API, and
frontend. Pipeline 1 has incompatible channels/classes/data and Pipeline 2 has
no CROMA or production analysis integration. The live `analysis_engine` is not a
fourth pipeline: it is the current production controller for Pipeline 3's
validated modules.

The canonical claim is bounded. The web runtime has no trained task head, so
`prediction` is `null`; it exposes representation/evidence results only. GEE is
available for discovery, but live analysis does not yet acquire provider
rasters. The typed Phase 3 orchestration code is scientific foundation and is
not silently substituted for the compatible dictionary-based web controller.

## 5. Dependency analysis

The production call graph is:

`static app → src.api → analysis_engine → query/data/input/raster modules → physical + optional CROMA + optional representation-only fusion → spatial evidence → constrained interpretation → report/frontend`

Import tracing established:

- `src/api.py` imports `analysis_engine`; neither it nor the frontend imports an experiment package.
- No `src/*.py` file imports `experiments.pipelines` (enforced by a regression test).
- No tests invoke Pipeline 1 or Pipeline 2 runners.
- Pipeline 3 previously imported `gee_status` and `select_mode` from Pipeline 2's `data_access.py`; this was the only numbered P3→P2 import and is now removed.
- Pipeline 2 temporal variants still import each other. They are isolated historical dependencies and do not enter production.
- `gee_croma_temporal` consumes saved Pipeline 2 temporal exports. It remains a historical chained experiment, not the canonical live runtime.
- `src/run_pipeline.py` and `src/hybrid_pipeline.py` reuse canonical loaders/CROMA/features for extraction and validation; they are scientific CLIs, not alternate API controllers.
- No Docker files, workflow files, Python packaging entry points, JavaScript package manifests, or CI routing definitions were present to redirect.

## 6. Components migrated

The GEE package/authentication status check and `auto|local|gee` selection logic
were moved without semantic change into `src/gee_experiment_runtime.py`. Both
numbered experiments now import this shared adapter. The Pipeline 2 module
re-exports the same symbols, preserving historical import compatibility, while
Pipeline 3 no longer depends on Pipeline 2.

No scientific calculation was migrated. In particular, CROMA normalization,
feature formulas, fusion initialization, evidence thresholds, model code, and
metrics were untouched.

## 7. Components retained for validation

- All Pass 3, Pass 5, and Pass 5D manifests, splits, predictions, metrics,
  receipts, provenance, comparisons, and reports.
- Canonical pixel, evidence, and interpretation artifacts for sample `61_39`.
- Phase 1/2/3 typed scientific contracts and end-to-end verification modules.
- `hybrid_pipeline.py`, training/evaluation utilities, and scientific smoke code.
- `autonomous_execution_report.json`, including Pipeline 1/2 results and paths,
  as a historical audit record.
- Stored real-GEE audit support in `src/qa_runtime.py` (class F, pending a future
  dedicated separation that must retain its scientific evidence).

## 8. Components archived

- Pipeline 1 `experiments/github_sen12ms_training`: class C, now explicitly
  archive/reproduction-only.
- Pipeline 2 `experiments/pipelines/gee`, `gee_temporal`, and
  `gee_temporal_classical`: class C, marked non-production.
- Pipeline 3's `gee_croma_temporal`: retained as a historical Pipeline 3
  experiment, not as temporal production functionality.

Archive status is documentation and dependency isolation, not a cosmetic mass
move. Paths were preserved so prior commands and reports remain auditable.

## 9. Components deleted

None. The audit did not prove any source, scientific record, or historical
runner to be category E with enough confidence to delete it. No datasets,
weights, artifacts, metrics, tests, or user-owned files were removed.

## 10. Runtime architecture after consolidation

The sole supported application startup is `python -m src.api`. HTTP analysis
calls exactly `src.analysis_engine.run_analysis`. The detailed authoritative map
is `docs/CANONICAL_RUNTIME_MAP.md`.

```text
INPUT (upload/local; GEE availability search is separate)
  → PIPELINE 3 CORE: validation + aligned S1/S2
  → REPRESENTATION: physical + official CROMA + optional hybrid
  → PREDICTION: unavailable in production; controlled validation only
  → EVIDENCE: deterministic pixel/token/region sidecar
  → INTERPRETATION: constrained allowlisted text
  → API: src/api.py
  → FRONTEND: src/static
```

Production modules are under `src/`; numbered old runners stay under
`experiments/`. Scientific CLIs under `src/` are validation utilities and do not
receive HTTP traffic.

## 11. Scientific invariants preserved

- Canonical optical order remains B01, B02, B03, B04, B05, B06, B07, B08,
  B8A, B09, B11, B12; SAR remains VV, VH.
- Input size, B02-grid alignment, finite/grid validation, and the official CROMA
  normalization remain unchanged.
- CROMA source/checkpoint selection and optical/SAR/joint tensor shapes remain
  unchanged.
- Physical schemas remain 52 optical, 9 SAR, and legacy joint 62.
- Hybrid scientific predictors, seed behavior, labels, datasets, fixed splits,
  training selection, metrics, and stored outputs remain unchanged.
- Pixel/token mapping, evidence schemas/thresholds/claims, interpretation
  allowlist/templates, and provenance remain unchanged.
- Production still makes no task prediction, temporal result, accuracy claim,
  or calibrated confidence claim.

## 12. Risks

- “Pipeline 3” names both a numbered experiment and the broader selected
  production lineage; this document fixes their relationship, but old commit
  history retains earlier wording.
- The numbered `gee_croma` runner checks GEE authentication yet processes local
  samples rather than acquiring an AOI/date raster. Treating it as live GEE
  acquisition would overclaim capability.
- Live GEE discovery and historical GEE acquisition use separate code paths.
  Unifying them would change operational scope and was intentionally deferred.
- `analysis_engine` and typed `phase3_orchestration` exchange different contract
  styles. Routing one through the other without compatibility tests is risky.
- Historical Pipeline 1 has unpinned optional dependencies and fixed local
  paths; it is evidence, not a clean-machine supported workflow.

## 13. Remaining legacy references

- Historical CLI commands remain inside archive READMEs for deliberate
  reproduction; every such README now labels its status.
- `autonomous_execution_report.json` intentionally retains absolute machine
  paths as a historical report and contains a duplicate `existing_report` JSON
  key in the Pipeline 2 object. Editing that evidence was avoided.
- `gee_croma_temporal` still depends on saved Pipeline 2 temporal outputs. This
  is class C historical chaining, not production coupling.
- `src/qa_runtime.py` references stored Pipeline 2 temporal exports (class F) in
  addition to testing API startup. It is isolated from server imports.
- `GEE_OUTPUT_ROOT`/`GEE_PIPELINE_MODE` and temporal before/after variables remain
  scoped to archive runners; canonical runtime configuration uses `.env.example`
  plus `GEE_PROJECT` for availability discovery.
- The two pre-existing untracked documents `ARCHITECTURE_AUDIT.md` and
  `ARCHITECTURE_FREEZE.md` were inspected and preserved without modification.

## 14. Recommended next architecture step

After this consolidation passes all gates, the next task should be **Phase 6A.3
— Canonical Architecture Contracts**: introduce `SceneBundle`, `TaskRequest`,
`TaskResult`, a capability registry, evidence-contract integration, provenance
propagation, and explicit compatibility boundaries. That work must wrap the
current Pipeline 3 behavior and must not silently reroute or change scientific
outputs.

## Verification record

Before changes:

- branch/commit: `main` at `1e8a327`
- working tree: only user-owned untracked `ARCHITECTURE_AUDIT.md` and
  `ARCHITECTURE_FREEZE.md`
- Python: 190 passed, 360 warnings
- frontend: 7 passed
- representative output: completed; physical 62; pooled CROMA 2304; hybrid 192;
  17 regions; 3 claims; prediction `null`

After changes:

- Python: 194 passed, 360 warnings (the increase is four new consolidation
  guards; no prior test was removed or weakened)
- frontend: 7 passed
- Python compilation, JavaScript syntax, and `git diff --check`: passed
- API startup/health: passed on loopback
- numbered Pipeline 3 local-reference harness: passed for all 3 samples on CUDA,
  with zero errors and all optical/SAR/joint CROMA token/GAP shapes preserved
- real `61_39` HTTP analysis: completed; physical 62; CROMA shapes
  `[1,225,768]`/`[1,768]` for optical, SAR, and joint; pooled 2304; hybrid 192;
  17 regions; 3 claims; interpretation `ANSWERED`; prediction `null`
- persisted `/api/report/{id}` response matched the analysis response exactly
- normalized pre-change `HEAD` and post-change scientific/report fingerprints:
  `32753a3e316b8dbc6921f72d32aaa51bd9a5e4d8c87cb41de3836f352e66ccb9`
  for both trees

Fingerprint normalization excludes generated UUIDs, timings, adapter cache
state, and interpretation hashes derived from request-specific provenance. It
includes physical and CROMA values, hybrid values, spatial evidence, rendered
interpretation content/claims, model-result state, confidence state, and
temporal state. Scientific and API behavior are unchanged.
