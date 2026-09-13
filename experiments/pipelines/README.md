# Experimental Pipeline Archive

These directories preserve the three numbered research pipelines and later
temporal experiments. They are not production entry points. The only supported
application entry point is `python -m src.api`; it dispatches to
`src.analysis_engine.run_analysis`.

The numbered identities are fixed by
`experiments/reports/autonomous_execution_report.json`:

| Pipeline | Historical implementation | Current status |
|---|---|---|
| 1 | So2Sat-LCZ42 S1/S2 FusionCNN under `experiments/github_sen12ms_training` with external data/checkpoints | Archived, reproduction-only |
| 2 | GEE-only scaffold under `experiments/pipelines/gee`; real-GEE temporal extensions under `gee_temporal*` | Archived, reproduction-only |
| 3 | GEE + official CROMA experiment under `experiments/pipelines/gee_croma`; prior real-GEE+CROMA temporal evidence under `gee_croma_temporal` | Canonical scientific lineage, but these folders remain experiment/reproduction harnesses |

An earlier version of this index called the local BigEarthNet+CROMA path
“Pipeline 1.” That label conflicted with the persisted master execution report.
The local BigEarthNet loader, preprocessing, CROMA adapter, physical features,
hybrid representation, evidence, and interpretation are now components of the
canonical Pipeline 3 production architecture.

See `docs/PIPELINE_CONSOLIDATION.md` for the forensic classification and
`docs/CANONICAL_RUNTIME_MAP.md` for the supported runtime path.
