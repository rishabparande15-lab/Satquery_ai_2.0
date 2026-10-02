# Phase 3U — Single-Image VQA and Grounding Foundation

**Status: `PHASE3U_VQA_COMPLETE_GROUNDING_BLOCKED`**

Phase 3U adds an explicit S2-only VQA path to the existing application. It reuses the frozen Phase 3O.5 S2 projector and pinned Qwen without training or changing weights. Text-guided grounding remains blocked: the repository has preserved source geometries, but no approved specialist or verified image-coordinate mapping.

## Reused Components

- S2 identity/split metadata and raster validation: `discover_s2_samples` and `load_optical_sample` in `src/dataset_loader.py`. This route reads only optical bands and does not require S1 or reference maps.
- Canonical 12-band order, 120x120 grid, CRS/resolution/bounds, and Phase 3O `robust_channel_scale_v1` preprocessing.
- Frozen `S2MultispectralProjector` from Phase 3O.5: checkpoint SHA-256 `e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db`; state fingerprint `db9ae1c4ed85e7c5044150fa32797a49403a4a3e3cfbe65d875b15803892df46`.
- `Qwen25VLRGBAdapter` with Qwen2.5-VL-3B-Instruct revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, frozen, plus the Phase 3O.10 generation-prefix interface.
- Existing binary and MCQ parsing/evaluation semantics from Phase 3O.6/3O.7.
- Existing fail-closed grounding geometry contracts; these remain model-neutral and do not emit predictions.

## Single-Image Contract and API

`POST /api/v1/single-image` selects `SINGLE_IMAGE_VQA` for one optical image and a binary or multiple-choice question. Local patches must be in train or validation according to `metadata.parquet`; TEST patches are excluded. A single external 12-band GeoTIFF is also accepted through the existing upload endpoint and is marked `external_inference`, not assigned a benchmark split. TIFF band descriptions must identify canonical B01..B12 order or the user must confirm that order.

The route validates one image, finite `[12,120,120]` data, CRS, resolution, bounds, and image identity. It does not read S1, invoke the joint CROMA projector, or alter the existing `/api/v1/satquery` paired route. Binary and MCQ are enabled; unsupported tasks return a 422 error. Temporal intent returns `TEMPORAL_ROUTE_NOT_IMPLEMENTED`. Explicit paired optical-SAR intent does not fall back to S2.

The response includes selected route/specialist, generated and parsed answer, image split/identity, spatial metadata, projector/Qwen provenance, visual-token shape, and execution trace. Qwen output remains experimental and is not treated as scientific evidence or a benchmark prediction.

## VQA Evaluation

The deterministic Phase 3O.7 task-balanced validation panel was reused without new selection or inference. The Phase 3U VQA subset consists of 60 validation records: 30 binary and 30 MCQ. The source panel has 90 validation records total, with 30 captions; caption results remain separate because all 30 were empty.

| Task | Records | Correct | Parsed | Result |
| --- | ---: | ---: | ---: | ---: |
| Binary | 30 | 17 | 28 | 56.7% exact accuracy |
| Multiple choice | 30 | 8 | 26 | 26.7% exact accuracy |
| Caption, limited observation | 30 | 0 nonempty | N/A | 30 empty outputs |

These are internal controlled validation results from Phase 3O.7, not RSVQA, VRSBench, generalization, or superiority claims. RSVQA is not locally acquired: its release-specific data license, linkage, evaluator, and leakage gates remain blocked. VRSBench assets are also not locally materialized; Phase 3K.3 records incomplete upstream imagery licensing, pinning, linkage, leakage, and geometry semantics.

Live smoke tests used one confirmed train-split patch. Playwright Binary returned `no`; MCQ returned `A`; the upload path returned `yes` for a test GeoTIFF materialized from that same verified train patch. The upload is recorded as external inference and was not added to any evaluation metric.

## Grounding Audit and Decision

The image-language audit records 12,178 text-box records and 13,122 point records (25,300 total). Their source geometry is preserved, but coordinate space, origin, axis order, normalization denominator, endpoint rule, source raster mapping, and conversion to the 120x120 analysis grid are unresolved. All 25,300 remain unmapped; none was used as a pixel/token label.

Phase 3K.1/3K.3 inspected VRSBench, OPT-RSVG/DIOR-RSVG, GeoChat, refGeo/GeoGround, and GeoPixelD. None cleared the complete licensing, immutable data pin, per-record image/annotation linkage, coordinate semantics, dimensions, and leakage gates. No grounding specialist was therefore selected. No predicted box, mask, score, confidence, overlay, or IoU/Dice metric is emitted. The UI returns `GROUNDING_MODEL_UNAVAILABLE` and explicitly says no geometry was produced.

## Routing and Frontend

The agent registry now distinguishes `SINGLE_IMAGE_VQA`, `SINGLE_IMAGE_GROUNDING`, `OPTICAL_SAR_ANALYSIS`, and `TEMPORAL_ROUTE_NOT_IMPLEMENTED`. The planner only uses these new task types when `SINGLE_IMAGE` mode is explicit, preserving prior routing for legacy requests. Caption remains experimental/limited.

The interface has a Single image / Optical + SAR mode toggle. Single image supports a local non-test S2 selector or one 12-band GeoTIFF upload; task selection exposes VQA and a visibly blocked grounding option. The paired route is still available separately, and bi-temporal mode is disabled. Playwright verified Binary, MCQ, upload, blocked grounding, missing/wrong image, unsupported task, temporal rejection, paired-intent rejection, mode toggles, and loading recovery. Successful VQA requests produced no browser page errors; expected HTTP 422/404 negative checks generated three browser resource-error log entries but no uncaught page errors.

## Integrity and Test Access

- Training performed: no.
- Qwen, CROMA, projector weights, prompts, and scientific metrics changed: no.
- Existing S2 projector/Qwen checksums and revisions are reported in the machine-readable artifacts.
- TEST records used for training, model selection, or metrics: zero; the reused Phase 3O.7 receipt reports zero TEST access. A broad initial repository search incidentally surfaced excerpts from the stored holdout annotation artifact; those excerpts were not loaded by code, used, scored, or copied into Phase 3U artifacts.

Machine-readable manifests and results are under `artifacts/final/single_image/phase3u/`.

## Limitations and Next Phase

VQA remains experimental: the validation sample is small and drawn from BigEarthNet.txt, not RSVQA; the learned projector has not been promoted to a benchmark model. Grounding is not implemented and no overlay is available. The existing scientific, paired S1+S2, and temporal branches remain separate; no scientific score or capability is inferred from the VQA smoke test.

The next phase is **BI-TEMPORAL CHANGE UNDERSTANDING + CHANGE-VQA**, beginning with source/license/split readiness gates. No temporal inference is enabled by Phase 3U.
