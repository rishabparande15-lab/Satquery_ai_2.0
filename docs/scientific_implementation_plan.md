# SatQuery AI Scientific Platform Implementation Plan

Date: 2026-09-11

This plan follows SPECIFY -> INSPECT -> PLAN -> IMPLEMENT -> TEST -> VERIFY -> DOCUMENT. Each milestone is additive, preserves the current runtime, and ends at an executable evidence gate.

## Milestone 1: Establish a reliable regression boundary

- Add repository-local pytest configuration restricting collection to `tests/`; generated/external experiment artifacts are data, not test packages.
- Run the complete Python suite, compilation, and frontend JavaScript tests.
- Record exact baseline failures rather than treating collection errors as test failures.

Verification: `python -m pytest -q`, `python -m compileall -q src tests`, and `node --test tests/frontend_state.test.cjs`.

## Milestone 2: Strict BigEarthNet area contract

- Extend `Sample`/`PreparedSample` without breaking existing callers: preserve raw and normalized S1/S2 fields, add the categorical reference tensor and source provenance.
- Validate dataset roots and discover S2, S1, and reference identities independently.
- Reject duplicate bands/maps and malformed or orphaned areas with area-specific errors.
- Validate every source raster before reprojection: readability, one band, CRS, finite/nodata state, north-up orientation, nondegenerate bounds, and expected native dimensions.
- Permit the established B02-grid optical resampling policy while requiring S1/reference spatial compatibility with the B02 target grid.
- Load reference maps with nearest-neighbor categorical semantics and require integer `[120,120]` output.
- Add synthetic tests for valid data, missing/duplicate bands, invalid reference, dimensions, CRS, bounds/orientation, nonfinite data, shifted grids, and incomplete areas.

Verification: focused loader tests first, then full suite and a real three-sample load.

## Milestone 3: Token targets and grounding foundation

- Keep the existing row-major 15x15/8x8 target construction as the canonical target implementation.
- Add normalized/clipped image-space bounding-box validation and deterministic mapping to intersecting row-major token indices.
- Represent token overlap fractions so partial boxes do not become fabricated hard assignments.
- Add center, corner, full-image, small, boundary, and partially overlapping box tests.
- Add an optional debug plot utility for reference map, grid, and dominant/fraction summaries.

Verification: exact token-index assertions, target fraction invariants, and rendered debug artifact when plotting dependencies are available.

## Milestone 4: Canonical scientific smoke experiment

- Build a runner around existing Phase 1/3.6 primitives rather than another model implementation.
- Accept a controlled manifest/root, immutable area split, batch size, seed, device, output root, and maximum area count.
- Execute discovery/validation, raw loading, preprocessing metadata capture, frozen CROMA extraction, six shape checks, 19-class targets, physical features, representation assembly, a one-epoch linear coverage head, validation, test metrics, constant-mean baseline, and artifact receipts.
- Keep test data out of model/epoch selection and stop with a documented unavailable status if required external data/model assets cannot be read.

Verification: execute 5-10 real controlled areas where a valid train/validation/test allocation is available; otherwise execute the already materialized controlled pipeline and document the boundary precisely.

## Milestone 5: Representation and sensor comparisons

- Reuse the immutable 600/200/200 controlled subset and existing artifacts.
- Compare optical-token CROMA, SAR-token CROMA, joint-token CROMA, supported physical features, and a spatially preserving hybrid only when the latter is specified without consulting held-out test labels.
- Use the same target mask, area IDs, soft-target loss, seed policy, validation selection, and evaluation functions.
- Clearly separate the existing scene-pooled untrained HybridFusion comparison from token-aligned scientific experiments.

Verification: shared-manifest identity hashes, explicit leakage audit, per-experiment held-out prediction artifacts, and actual metric tables. If a fresh holdout is required because prior test results informed architecture changes, do not reuse the old test set for a new final claim.

## Milestone 6: Annotation linkage and application truthfulness

- Provide a reusable BigEarthNet.txt parser/linkage statistics boundary over the existing conservative identity linker.
- Preserve original strings and distinguish Q&A, MCQ, captions, and region/object records.
- Audit controller routes for optical, SAR, joint, classification, feature analysis, caption/VQA, and temporal requests.
- Expose a scientific prediction only when a validated checkpoint is explicitly configured; otherwise retain truthful unavailable and uncalibrated-confidence messages.

Verification: linkage count/identity tests and API regression tests for every unsupported capability.

## Milestone 7: Independent audit, failure injection, and documentation

- Run the full suite, real smoke path, controlled larger experiment available in the environment, and deliberate malformed inputs.
- Measure practical startup, cold/warm CROMA, feature extraction, training, memory/device behavior, and artifact size without extrapolating beyond the tested machine.
- Independently reload manifests/predictions/artifacts and recompute shape, target, leakage, and metrics checks.
- Generate `docs/scientific_evaluation_report.md` and `docs/FINAL_PIPELINE_AUDIT.md` from actual evidence.
- Update README with exact reproduction commands, dataset boundaries, capability status, and limitations.

Verification: every positive claim cites a command and an artifact; unavailable stages remain YELLOW/RED and no metric or confidence is invented.

## Stop conditions

- Do not download or process additional large data before the controlled smoke path is valid.
- Do not alter official CROMA code/checkpoints or existing cached feature semantics.
- Do not integrate an unvalidated trained head into the website.
- Do not call annotation linkage VLM training or token mapping a grounding model.
- Do not claim temporal prediction, calibrated confidence, captioning, or VQA without an executed validated model.
