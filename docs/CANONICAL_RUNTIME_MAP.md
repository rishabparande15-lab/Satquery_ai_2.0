# Canonical Pipeline 3 Runtime Map

Status: authoritative runtime map, reviewed 2026-09-15.

The typed capability/contracts layer, optional representation persistence, and trusted receipt catalog wrap or extend this runtime without creating another production controller. The completed 5,000-area predictor remains a controlled scientific workflow; the web runtime's 192-D seeded fusion output remains representation-only.

```text
Browser: src/static/index.html + app.js
                    |
                    v
HTTP: src/api.py  (the only supported application entry point)
                    |
                    v
Orchestrator: src/analysis_engine.run_analysis
  | request/query   -> query_interpreter.py + input_validation.py
  | source choice   -> data_orchestrator.py
  | GEE discovery   -> imagery_availability.py (availability only)
  | raster loading  -> raster_inputs.py + dataset_loader.py
  | preprocessing   -> CROMAAdapter's established normalization
  | physical        -> modality_features.py -> gee_features.py
  | CROMA            -> croma_adapter.py -> external official CROMA/checkpoint
  | fusion           -> hybrid_fusion.py (representation only in web runtime)
  | spatial sidecar  -> pixel_features.py -> evidence_schema.py
  | interpretation   -> interpretation_adapter.py
  | persistence      -> analysis_engine.save_report
  v
JSON response/report -> frontend rendering + experiments/outputs/web_reports
```

Scientific validation uses the same underlying loaders, feature/CROMA modules,
schemas, and frozen artifacts through `src/phase1_foundation.py`,
`src/phase2_integration.py`, `src/phase3_orchestration.py`, Pass 3/5/5D modules,
scripts, tests, and tracked experiment records. Those are validation and
reproduction paths, not alternate production controllers.

`experiments/pipelines/gee_croma` is the numbered Pipeline 3 research harness.
`experiments/pipelines/gee`, `experiments/pipelines/gee_temporal*`, and
`experiments/github_sen12ms_training` are archived. No production module imports
an experiment package, and Pipeline 3 no longer imports Pipeline 2.

Important boundary: the live API can query GEE availability through
`imagery_availability.py`, but `run_analysis` processes uploaded rasters or
configured local samples. It does not currently acquire GEE imagery. This map
does not imply temporal prediction, a deployed task head, or calibrated
confidence.
