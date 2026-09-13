# Repository Cleanup Report

Checkpoint date: 2026-09-13. Branch: `main`. Previous checkpoint: `5490b2e` (`feat: validate pixel explainability sidecar`).

## Removed

| File or directory | Reason | Evidence |
|---|---|---|
| `.pytest_cache/` | Local pytest cache with stale inaccessible entries | Ignored cache; no imports, runtime use, API use, CLI use, tests, or reproducibility references |
| `.pytest-temp/` | Local pytest scratch directories, including a Pass 5D test copy | `pytest.ini` marks it non-collectable; no scientific artifact or documentation depends on it |
| `.ruff_cache/` | Local static-analysis cache | Ignored generated metadata with no runtime or reproducibility role |
| `experiments/outputs/final_checkpoint/` | Temporary independent Pass 5C rebuild used during this checkpoint | Rebuilt evidence matched the tracked canonical hash exactly; output was intentionally temporary and under the ignored output root |

No versioned source, test, scientific report, manifest, prediction, or validated result was deleted.

## Retained

| File or group | Reason |
|---|---|
| Pass 3 and Pass 5 manifests, splits, metrics, predictions, provenance, and repeat outputs | Required scientific scope, metric, leakage, and byte-reproducibility evidence |
| `artifacts/pixel_feature_probe/61_39` metadata and diagnostic | Pass 5B feasibility evidence; large NumPy arrays remain ignored |
| `artifacts/spatial_evidence/61_39` | Canonical Pass 5C evidence and deterministic diagnostic |
| `artifacts/interpretation/61_39` | Generated Pass 5E simple/technical answers and provenance |
| `data/contracts` | Small Pass 4 dataset contracts; these are metadata, not datasets |
| Completed Pass 3–5E modules, scripts, tests, and documentation | Referenced by runtime, tests, reproduction commands, or scientific reports |
| Historical experimental pipelines and reports | Prior user work and scientific history; deletion was not justified |

## Uncertain

| File or group | Why retained |
|---|---|
| `experiments/github_sen12ms_training/` | Historical training implementation; not part of the active API but potentially useful prior user work |
| `experiments/pipelines/gee*` | Historical/experimental GEE variants; some modules remain documented and dependency-linked |
| `experiments/reports/autonomous_execution_report.json` | Historical report with no evidence that it is safe or desirable to remove |
| `QA_REPAIR_REPORT.md` and `FRONTEND_DESIGN.md` | Existing tracked project records; outside the new Pass 5E implementation and not obsolete by proof |

## .gitignore Changes

Added explicit exclusions for `.tar`, `.tar.*`, `.zst`, JP2/HDF imagery or data containers, machine-local `data/raw`, `datasets`, and `checkpoints` roots, pytest scratch, coverage output, editor metadata, OS metadata, private keys, and conventional credential/service-account files. Existing exclusions for TIFF, ZIP, NumPy, PyTorch/checkpoint, Parquet, environment, logs, caches, and generated experiment outputs remain.

Validated JSON manifests, small reports, schemas, documentation, source, and tests are intentionally not ignored.

## Dependency Changes

None. Every direct dependency has repository use or an established runtime/transitive role. `pyarrow` supplies Parquet support, `click` is required in the Rasterio environment, and `einops` is required by the external official CROMA runtime; removing them would be unsafe.

No `pyproject.toml`, JavaScript package manifest, or lockfile exists. Frontend tests use the installed Node standard library only.

## Dataset Protection

- No archive, TIFF/JP2 imagery, NumPy tensor, Parquet dataset, model checkpoint, or external dataset is intended for staging.
- The 5,000-area and fragmented official archives remain outside the repository.
- Local S2, CROMA source, and checkpoint paths remain machine-local configuration values.
- The tracked `data/contracts` files contain only dataset identity, validation, and role metadata.
- Secret scanning found no credential value, API token, private key, or password in intended files.

## Test Impact

No test was removed or weakened. The pre-README checkpoint passed 190 Python tests and 7 frontend tests, Python compilation, JavaScript syntax checking, and `git diff --check`. The real `61_39` pipeline and loopback API also passed. Final post-README results are recorded in the README and final checkpoint handoff.
