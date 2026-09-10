# Phase 3 — Core satellite data orchestration

## Audit and architectural assessment

Baseline commit: `de432c539312b1501ba1ae6e62bafddf9dc3f92f`; the worktree was clean before Phase 3.

The existing request path (`query_interpreter.py` → `data_orchestrator.py` → `input_validation.py`/`raster_inputs.py` → `analysis_engine.py`) is operational for the local web proof of concept, but primarily exchanges dictionaries. `imagery_availability.py` contains a useful source registry, AOI normalization, truthful GEE configuration states, and limited GEE search. It is a prototype rather than the canonical Phase 3 contract. `dataset_loader.py` and `preprocessing.py` remain the real local BigEarthNet raster path. `phase1_foundation.py` and `phase2_integration.py` are the canonical ML boundary. `hybrid_pipeline.py` remains the authoritative advanced feature path. `temporal.py` compares already-created representations and is not an acquisition or change-model implementation.

Phase 3 adds `src.phase3_orchestration` alongside these components. It does not replace the web API or duplicate preprocessing/CROMA/fusion. Existing GEE code remains an optional backend candidate to place behind `DataSourceAdapter`; it was not expanded.

## Operational flow

```text
USER REQUEST
  → AOI CONTRACT
  → TEMPORAL CONTRACT
  → SOURCE / MODALITY REQUIREMENTS
  → RESOLUTION CONTRACT
  → AVAILABILITY ADAPTER
  → DETERMINISTIC SCENE SELECTION
  → ACQUISITION / LOCAL STAGING
  → RASTER + SPATIAL VALIDATION
  → PHASE 1 FOUNDATION
  → PHASE 2 ADAPTER
  → EXISTING ADVANCED PIPELINE
  → PROCESSING RESULT
```

## Contracts

`SatQueryRequest` contains the user query, normalized AOI, temporal request, modality, satellite preferences, requested GSD, cloud ceiling, before/after requirement, and request UUID. `ExecutionPlan` records selected source keys and explicit validation steps. `SceneCandidate` records scene/satellite/sensor identity, modality, timezone-aware acquisition time, native GSD, CRS, bounds, AOI coverage, exact band order, cloud metadata, product type, access metadata, and suitability notes. `ValidatedSatelliteDataset` contains validated optical/SAR arrays and all identities, spatial metadata, resolutions, hashes, acquisition times, and provenance. `ProcessingResult` can carry Phase 2 input, existing advanced features, warnings, errors, and execution provenance.

### AOI

`AOI.from_input` accepts latitude/longitude points, bounding boxes, GeoJSON Points, Polygons, and GeoJSON Features. It preserves EPSG:4326 geometry, bounds, input/source type, and a deterministic approximate area where available. Coordinates must be finite and within geographic ranges; polygons must be closed and non-degenerate. Named locations are representable as user input but require an explicit geocoder adapter—no coordinates are invented.

### Temporal

`TemporalRequest.create` accepts a single acquisition date, an inclusive date range, or an unambiguous before/after pair with nonnegative tolerance. Requested dates remain distinct from `SceneCandidate.acquisition_time`; selected acquisition times are preserved in the validated dataset. No change detection is performed.

### Sources and modalities

The verified registry defines Sentinel-2 MSI with canonical Phase 1 bands and band-level 10/20/60 m native metadata, and Sentinel-1 C-SAR with VV/VH at the operational 10 m product contract. It reports local support and GEE only when configured. Unsupported future sources such as Landsat, Cartosat-2S, and RISAT are not falsely declared operational.

Modalities are explicit: `OPTICAL`, `SAR`, `OPTICAL_SAR`, and `MULTISPECTRAL`. A pair is not considered co-registered merely because one optical and one SAR scene exist.

### Resolution semantics

`ResolutionContract` separately records `requested_gsd`, `native_gsd`, `source_product_gsd`, `processing_gsd`, `resampled_gsd`, `resolution_status`, and `information_status`. A 5 m request against native 10 m data remains `unavailable_at_requested_resolution`. If intentionally processed at 5 m, it is labelled `resampled_from_10m`, never native 5 m.

## Availability and selection

`AvailabilityRequest → DataSourceAdapter → AvailabilityResponse` is provider-neutral. `LocalSceneAdapter` is operational without GEE credentials. Providers must return unavailable/configuration-required states instead of fabricated scenes.

`filter_candidates` applies source, modality, inclusive date range, full AOI coverage, and cloud constraints. `select_scene` deterministically orders eligible scenes by native-resolution suitability, distance from the center of the requested date window, cloud percentage, and scene ID. The selected reason is explicit and stable.

## Acquisition and validation

`AcquisitionAdapter` separates metadata selection from materialization. `LocalAcquisitionAdapter` validates configured local paths, does not copy them into Git, and records SHA-256. Future download and GEE exporters can implement the same interface and stage into external/cache storage protected by `.gitignore`.

`validate_raster` requires an existing readable raster, CRS, positive dimensions, north-up transform, exact band count/order, full valid mask, finite values, scene/raster identity agreement, and full AOI coverage. Sentinel-2 accepts exactly `B01…B12` in the Phase 1 order; B10 is rejected. Sentinel-1 accepts exactly `VV,VH`. Reordering is never silent. `validate_spatial_pair` requires matching CRS, bounds, transform, resolution, and grid shape before joint processing.

## Phase 1/2 bridge and provenance

`bridge_to_phase2` converts a `ValidatedSatelliteDataset` into the canonical Phase 1 `SampleIdentity` and `PreparedBatch`, runs the supplied frozen CROMA-compatible model, computes the existing physical feature schema through the supplied provider, and calls `adapt_phase1_output`. It preserves request/scene IDs, acquisition times, AOI, CRS, bounds, native/processing GSD, band order, normalization profile, file hashes, CROMA checkpoint identity, Phase 3 version, and Phase 2 adapter provenance. It does not change `HybridFusion` or add/train a model head.

Successful callers should also record retrieval time (`datetime.now(timezone.utc)`), provider response, selection reason, execution configuration, and the existing model/checkpoint identity in `ProcessingResult.provenance`. Secrets must never enter these records.

## Failure handling

`OrchestrationError` provides stable codes: `INVALID_AOI`, `INVALID_TEMPORAL_REQUEST`, `UNSUPPORTED_SOURCE`, `NO_SCENE_AVAILABLE`, `REQUESTED_RESOLUTION_UNAVAILABLE`, `INSUFFICIENT_AOI_COVERAGE`, `MODALITY_PAIR_INCOMPATIBLE`, `DATA_SOURCE_UNAVAILABLE`, `DATA_SOURCE_CONFIGURATION_REQUIRED`, and `INVALID_RASTER`. There is no unrelated-scene fallback.

## Tests

`tests/test_phase3_orchestration.py` covers valid/invalid AOIs, temporal variants, the Sentinel registry, unsupported sources, modality filtering, truthful GSD semantics, deterministic scene selection, no-scene behavior, exact bands, B10/order rejection, non-finite raster rejection, AOI coverage, optical/SAR grid mismatch, and Phase 3 → Phase 1 → Phase 2 identity/provenance preservation. The full suite retains Phase 1, Phase 2, API, frontend, data-loader, training, and scientific feature regressions.

## Limitations and deferred work

Phase 3 does not geocode named locations, add a production remote downloader, make GEE mandatory, implement cloud-mask science, resample acquired products, crop AOIs, select paired acquisitions as a temporal model, calculate change, train/retrain models, add VQA/captioning/grounding, create an agent/controller, calibrate confidence, redesign the UI, or deploy the service. Those require later explicit adapters and validation. The current bridge requires paired optical/SAR inputs because the established Phase 1/2 joint foundation does.

Phase 4 should add concrete remote source adapters and a durable execution service around these contracts, beginning with integration of the existing GEE search/export code behind `DataSourceAdapter`/`AcquisitionAdapter`, while preserving truthful provider states and the canonical Phase 1/2 bridge.
