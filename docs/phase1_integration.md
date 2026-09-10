# Phase 1 integration: BigEarthNet/CROMA foundation

## Baseline freeze

- Starting commit: `3227014bd111038ed57d1df2822ebae52395bb17`.
- Starting worktree: already dirty (`README.md` modified and the application, tests, experiments, requirements, and reports mostly untracked). Those user files were preserved.
- Initial Python test result: collection failed in four modules because the active interpreter could not import the declared `click` dependency through Rasterio.
- Initial frontend test result: 6 passed.
- Existing CROMA behavior: `src.croma_adapter.CROMAAdapter` dynamically loads the configured official CROMA source/checkpoint, accepts raw canonical `[12,120,120]` optical and `[2,120,120]` SAR arrays, applies its own normalization, and returns six representations. Existing recorded artifacts have `[1,225,768]` encodings and `[1,768]` pooled outputs.
- Existing API behavior: the local HTTP API supports local or uploaded optical/SAR analysis, validates uploads, computes physical and optional CROMA/hybrid features, and explicitly declines task predictions and temporal claims when unavailable.
- Recovery: no original function or file was deleted or replaced. Phase 1 is added through `src.phase1_foundation`.

## Repository audit

The current project has a standard-library HTTP backend (`src/api.py`), static HTML/CSS/JavaScript frontend (`src/static`), CROMA adapter (`src/croma_adapter.py`), BigEarthNet loader and preprocessing (`src/dataset_loader.py`, `src/preprocessing.py`), raster/API validation (`src/input_validation.py`, `src/raster_inputs.py`), feature/cache/training modules, tests, and extensive experimental GEE/temporal pipelines and saved reports. The authoritative downstream design is the existing physical + CROMA + hybrid pipeline. Dataset defaults point to an external three-sample BigEarthNet v2 fixture and external official CROMA source/checkpoint under `D:\Satquery_ai datasets`.

## Reference mapping

Reference revision inspected: `raviasha/Sat_Query` as downloaded on 10 September 2026. The repository's pinned CROMA source revision is `59505a6bcadbf36ba20767270154bf9f3067c5e7`; its pinned checkpoint revision/hash remain reference provenance, not copied weights.

| Reference component | Reference implementation | Our component | Action | Reason |
|---|---|---|---|---|
| Strict raster loading | `preprocessing.py` | `dataset_loader.py`, `input_validation.py`, Phase 1 contract | ADAPT | Keep current application loader while adding strict tensor/identity boundary; reference requires native BigEarthNet geometry. |
| Normalization | `normalize_patch` | `phase1_foundation.normalize_patch` | REUSE | Preserve mean ±2 sample std, uint8 quantization, `/255`, and constant-channel behavior as a named profile. |
| CROMA wrapper | `features.py` | `CROMAAdapter`, `FrozenCromaFoundation` | ADAPT | Keep configured official implementation; add frozen six-output validation and provenance requirement. |
| CLI extraction | `extract_croma.py` | existing pipeline CLIs | DO NOT IMPORT | Reference CLI orchestration would duplicate and redirect our pipeline. |
| 19-class targets | `targets.py` | `patch_label_targets` | REUSE | Exact official class order/codes and unknown-area accounting are scientifically important. |
| Coverage head | `prediction.py` | `ReferenceCoverageHead` | ADAPT | Imported only as a clearly named reference baseline; softmax outputs are fractions, not confidence. |
| Artifact pairing | `prediction_data.py` | typed identities, validation and receipts | ADAPT | Preserve identity-first joins without importing reference directory layout. |
| Split discipline | `evaluation.py`, `validation_training.py` | `validate_split_integrity` | ADAPT | Prevent identity overlap and keep test held out; existing advanced trainer remains authoritative. |
| Metrics | `evaluation.py` | `coverage_metrics` | ADAPT | Add MAE/RMSE/bias, per-class support, dominant accuracy/confusion, and whole-area bootstrap interval. |
| Annotation matching | `match_annotations.py` | `link_annotations` | ADAPT | Match both Sentinel IDs and retain records/splits while marking semantics and token labels unvalidated. |
| Selective downloader | `download_bigearthnet.py`, `remote_lmdb.py` | external dataset provisioning | DO NOT IMPORT | Useful but not required inside the current runtime; source-specific networking would broaden Phase 1. |
| Manifests/receipts | exporters and stage scripts | `write_experiment_report` | ADAPT | Keep compact machine-readable provenance and SHA-256 completion receipt without adopting Colab layout. |
| Vendored CROMA | `_vendor/croma.py` and license | configured external official CROMA | DO NOT IMPORT | Current project already uses the official source/checkpoint; copying creates two architectures and licensing/provenance risk. |
| GEE/UI/temporal/language pipeline | proposed or separate reference work | existing advanced pipeline | DO NOT IMPORT | Explicitly outside Phase 1 and must not replace our architecture. |

## Contracts

Optical order is exactly `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12`; SAR order is `VV,VH`. Prepared batches are finite float32 `[N,12,120,120]` and `[N,2,120,120]`. B02 is the reference grid, with float32 bilinear resampling documented at the loading boundary. Required identity includes patch ID, both Sentinel IDs, official split, acquisition metadata, CRS, bounds, positive resolution, and source provenance. Missing identity or incompatible shapes fail rather than being fabricated or silently repaired.

CROMA is frozen and must return exactly optical, SAR, and joint spatial encodings `[N,225,768]` plus scene representations `[N,768]`. The 225 tokens use row-major `row * 15 + column` ordering on a 15×15 grid.

Reference targets are `[N,225,19]`. Each token corresponds to an 8×8 raster block. Fractions use denominator 64 even when pixels are unknown/excluded; unknown share is separately preserved and `fully_labeled_mask` controls baseline eligibility.

The reference baseline is `Linear(768,19)` followed by softmax. Its values are estimated class fractions, not confidence probabilities. It is separate from, and does not replace, the advanced fusion model.

Annotations are linked by both optical and SAR identities. Q&A, caption, bounding-box and region strings are retained. Missing records are not negative labels; semantics and token-level validity remain false until later specialist validation.

## Evaluation and reproducibility

Training may use only train identities; validation selects epochs/models; test is final held-out evaluation. Image-derived annotations inherit image splits. Reports include configuration, ordered sample identities, model/checkpoint information, feature contract, metrics, and a SHA-256 completion receipt. Metrics are coverage errors in percentage points and whole-area bootstrap intervals; they are not calibration claims.

## Tests and known limitations

`tests/test_phase1_foundation.py` covers deterministic normalization, strict tensor/metadata contracts, frozen six-output CROMA shapes, official taxonomy and token ordering, unknown-label preservation, baseline fractions, split leakage, metrics, provenance receipts, and conservative annotation linkage. Existing real-fixture tests cover the three local samples when the external dataset is present.

Phase 1 does not copy the reference downloader/export directory machinery, distribute the 778 MB checkpoint, parse annotation box geometry, train a language model, add GEE/live retrieval, implement temporal change analysis, or alter the UI. A full real-data reference comparison still requires the reference's 1,000-area artifacts, which are not contained in either Git clone.

## Boundary to Phase 2

`BigEarthNet S1/S2 → strict validated tensors → named reference normalization → frozen CROMA optical/SAR/joint features → reference targets and conservatively linked annotations → OUR ADVANCED PIPELINE`.

Recommended Phase 2 entry point: add an adapter that consumes this typed foundation contract in the existing advanced fusion pipeline, then design and validate specialist models without weakening identity, split, spatial, or provenance checks.
