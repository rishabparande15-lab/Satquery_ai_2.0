# SatQuery AI Scientific Gap Analysis

Date: 2026-09-11

## Scope and evidence

This audit covers the repository source tree, README, QA/repair report, application API and frontend, local raster ingestion, preprocessing, CROMA adapter and contracts, physical and hybrid feature paths, temporal validation, training/evaluation utilities, experiment pipelines, configuration, tests, generated documentation, and the externally stored controlled BigEarthNet artifacts configured by the project. It also compares the architecture with `raviasha/Sat_Query` as a scientific reference and uses the Superpowers principles of specification first, incremental test-driven changes, and evidence before completion.

The worktree already contains user-owned modifications and untracked Phase 3.5/3.6 work. These changes must be preserved and reconciled; they are not a clean baseline to overwrite.

## 1. What the current system already implements

### Application/runtime

- A loopback HTTP API and static frontend with optical-only, SAR-only, and joint local analysis paths.
- Typed query interpretation, source/tool selection, raster validation, upload lifecycle controls, evidence, report persistence, controlled errors, and explicit unsupported-capability messages.
- Real physical feature extraction: a stable joint 62-feature schema plus modality-specific schemas.
- CROMA model caching, single-modality inference, joint inference, and truthful partial results when the checkpoint/model path is unavailable.
- Temporal input validation that rejects incompatible or unrelated pairs and does not claim an implemented change model.
- QA scripts and regression tests for API, frontend state, uploads, reports, CROMA failures, temporal safety, and application recovery.

### Scientific foundation

- Canonical band constants: optical `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12` and SAR `VV,VH`.
- BigEarthNet sample discovery and loading against a B02-defined 120x120 grid.
- Deterministic per-patch CROMA-compatible normalization and recorded normalization metadata.
- Explicit tensor contracts for optical `[N,12,120,120]`, SAR `[N,2,120,120]`, CROMA spatial outputs `[N,225,768]`, and pooled outputs `[N,768]`.
- Frozen official CROMA integration with exact six-output validation.
- A 19-class CORINE mapping and row-major 15x15 target construction from 8x8 pixel blocks, including unknown/excluded counts and masks.
- A reference `Linear(768,19)` coverage head, softmax fractions, soft-target cross-entropy in the Phase 3.6 benchmark, validation-selected epochs, early stopping, and reproducible seeds.
- Area-level split integrity checks, constant-training-mean baseline support, MAE/RMSE/bias/dominant-class/per-class evaluation, and area-bootstrap uncertainty.
- Physical, pooled-CROMA, and hybrid representation infrastructure; independent optical/SAR/joint CROMA representations are retained.
- Provenance-aware artifact bundles with shapes, dtypes, hashes, and reload verification.
- Conservative BigEarthNet.txt linkage by Sentinel-1 and Sentinel-2 identity, preserving annotation records and explicitly not treating linkage as VLM training.

### Executed evidence already present

- A real three-sample local BigEarthNet/CROMA path and Phase 3.5 one-sample end-to-end verification.
- A controlled external 1,000-area dataset with 600/200/200 area splits, prepared tensors, CROMA features, targets, and a pinned reference comparison.
- The documented reference reproduction reports 53.8743% dominant-class accuracy and 6.1553 percentage-point MAE versus a 9.2137 pp constant baseline.
- The documented current SatQuery scene-level hybrid probe reports 41.9170% dominant accuracy and 7.1989 pp MAE. It is scientifically weaker than the token-level CROMA reference and is correctly described as an untrained frozen representation plus trained linear probe.
- BigEarthNet.txt linkage statistics and leakage checks have already been generated for the controlled subset.

## 2. What is missing or incomplete

### Dataset loader contract

- Discovery presently returns incomplete samples and defers missing-file errors until load time; it does not expose a structured discovery/validation report.
- Reference-map duplication is not rejected, and orphan/incomplete S1, S2, or reference areas are not comprehensively classified.
- The loader reprojects optical and SAR inputs but does not independently reject missing source CRS, non-north-up orientation, invalid source bounds, malformed reference dimensions/content, or incompatible source footprints before reprojection.
- The reference map is not loaded into `PreparedSample`, so the core loader does not itself produce the required `[N,120,120]` reference tensor.
- The distinction between valid resampling of differing native optical resolutions and invalid shifted/mismatched sensor grids needs an explicit policy and tests.

### Target/debug foundation

- Target generation exists, but no repository utility visualizes the reference map, 15x15 grid, and class-fraction representation together.
- Region/bounding-box to token-grid mapping and its required boundary/overlap tests are absent.

### Training/evaluation unification

- The older training-ready workflow is multilabel BCE/sigmoid at scene level, while the scientific coverage task is softmax fraction prediction at token level. These paths are both valid for their stated tasks but are easy to confuse and need an explicit scientific CLI boundary.
- The controlled Phase 3.6 benchmark implements the scientific probe, but the checked-in code/documentation does not yet provide one canonical, small smoke command covering discovery through target creation, one epoch, evaluation, artifact saving, and report generation.
- A single consolidated `docs/scientific_evaluation_report.md` in the requested structure is absent.
- The requested three-way token-aligned physical-only/CROMA-only/hybrid experiment is not complete. The existing SatQuery hybrid benchmark repeats one scene vector over 225 blocks, which loses spatial information and is not evidence that token-level hybrid fusion helps.
- Complete optical-only, SAR-only, and joint token-level coverage comparisons on the same held-out protocol are not consolidated.

### Annotation, grounding, temporal, and controller scope

- Annotation linkage exists, but a reusable parser/statistics interface for the exact BigEarthNet.txt artifact is not yet a clear public module.
- No grounding model exists; only the requested deterministic box/token data foundation should be added.
- No validated temporal prediction model exists. Only pair validation and representation-difference foundations are supportable.
- The application controller recognizes several tasks but should be audited against the scientific model registry so classification/VQA/captioning/change requests cannot select an untrained model.
- The website does not yet expose a trained token-coverage model because no production-approved checkpoint exists. It must continue reporting uncalibrated confidence truthfully.

### Verification and reporting

- The exact requested final documents `docs/scientific_evaluation_report.md` and `docs/FINAL_PIPELINE_AUDIT.md` are absent.
- Failure injection is broad in the application tests but incomplete at the BigEarthNet scientific loader boundary.
- Performance evidence exists in multiple reports but is not consolidated into the requested startup/CROMA cold/warm/extraction/training/memory/artifact audit.
- Browser visual verification remains unavailable in the existing QA record.

## 3. What can be reused

- `src.phase1_foundation`: canonical constants, normalization, frozen CROMA validation, target generation, coverage head, split checks, metrics, and conservative annotation linkage.
- `src.dataset_loader` and `src.preprocessing`: discovery conventions, B02 target grid, raster reprojection, and deterministic normalization, after strengthening validation and reference loading.
- `src.croma_adapter`: official model loading and actual output paths.
- `src.phase2_integration` and `src.phase3_orchestration`: identity/provenance contracts and safe bridge into the existing runtime.
- `src.gee_features`, `src.modality_features`, and `src.hybrid_fusion`: established physical features and runtime-compatible fusion interfaces.
- `src.phase3_6_benchmark`: scientific soft-target training, validation selection, leakage checks, shared held-out manifests, and comparison/report helpers.
- `src.scientific_artifacts`: deterministic local artifact persistence and integrity verification.
- Existing API, frontend, upload/report infrastructure, temporal validation, QA scripts, and test fixtures.
- The external controlled datasets and generated manifests under `D:\Satquery_ai datasets`; they remain external to Git.

## 4. What must be newly implemented

1. A strict scientific dataset manifest/loader that validates duplicates, missing files, all source/reference rasters, CRS, north-up orientation, dimensions, bounds, finiteness, and cross-sensor compatibility, and returns reference tensors.
2. Focused loader failure-injection tests covering every requested malformed-area case.
3. A target visualization/debug utility and deterministic image-box-to-token mapping foundation.
4. A canonical scientific smoke runner that connects loader, preprocessing, frozen CROMA, targets, physical features, representation modes, one-epoch coverage training, held-out evaluation, artifacts, and a machine-readable receipt.
5. Clear coverage-task CLIs/configuration that cannot be confused with the existing multilabel scene-label training workflow.
6. Where the existing controlled artifacts permit it without test-set reuse for architecture selection, optical, SAR, joint, physical, and spatially aligned hybrid comparisons on one immutable area split.
7. Consolidated scientific evaluation and final audit documents generated strictly from executed artifacts.
8. README reproduction commands and an explicit capability/status matrix.

## 5. What must not be replaced

- The working API/frontend, upload security boundaries, report persistence, request routing, and CROMA cache.
- The official CROMA source/checkpoint or its real output semantics.
- Canonical band order, B02 grid policy, 19-class ordering, unknown/excluded-pixel policy, or row-major token order without explicit scientific evidence and migration documentation.
- Existing physical feature names/dimensions used by cached artifacts and runtime compatibility.
- Existing truthful unsupported states for VQA, captioning, grounding prediction, calibrated confidence, and temporal prediction.
- User-owned worktree changes and external datasets/artifacts.

## 6. Risks to the current runtime

- Tightening the shared loader may break the three development fixtures if scientific checks are applied indiscriminately to legacy/runtime uploads. Strict scientific validation should be additive or mode-specific.
- Changing preprocessing would invalidate existing CROMA caches and benchmark hashes. Normalization profiles must be versioned rather than silently changed.
- Conflating scene-level multilabel classification with token-level coverage prediction could produce plausible but scientifically wrong outputs.
- Reusing the existing test set for repeated architecture selection would invalidate final claims. New representation work must use validation or a fresh holdout.
- CROMA and 1,000-area runs depend on external data/checkpoints, GPU availability, and writable external artifact locations; tests need controlled skips and honest failure reports.
- Adding a trained head to the website without a reviewed checkpoint and calibration would create fabricated confidence semantics.
- Broad refactors could destabilize an application already covered by extensive regression tests.

## 7. Proposed implementation order

1. Preserve the current worktree and establish a fresh full-test baseline.
2. Strengthen the loader and reference tensor contract with red/green failure-injection tests.
3. Add region/token mapping and target visualization with deterministic unit tests.
4. Add a canonical smoke experiment runner and artifact receipt, initially using 5-10 controlled areas if the external assets support direct loading.
5. Unify scientific coverage configuration, training, checkpointing, evaluation, and constant-baseline reporting around the existing Phase 3.6 primitives.
6. Add sensor/representation experiments without touching the immutable test split during design; use a fresh final holdout if the existing test results have already informed changes.
7. Audit controller/application integration and retain explicit unavailable responses for unvalidated models.
8. Run the full regression suite, real smoke pipeline, failure injection, and practical performance measurements.
9. Perform an independent evidence audit and generate the two requested final reports and README instructions.

## Initial verdict

The repository is already beyond an engineering-only prototype in several narrow, executed paths, but it does not yet satisfy the complete requested deliverable as one reproducible platform. Current status is **YELLOW / PARTIALLY VALIDATED**: the reference scientific coverage pipeline and a controlled comparison have real evidence, while the strict reusable loader, grounding foundation, unified smoke runner, complete representation suite, and consolidated final audit remain incomplete.
