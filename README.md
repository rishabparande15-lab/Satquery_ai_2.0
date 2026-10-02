# SatQuery AI

SatQuery AI is a local, agentic vision-language system for remote-sensing imagery in SIH 2026 PS-26167. It validates the declared inputs and query, routes to a bounded specialist, and returns normalized evidence, provenance, warnings, execution trace, and JSON export.

## Final core status — 2 October 2026

- `PHASE3AC_COMPLETE`
- `PHASE3AD_SINGLE_SAR_COMPLETE`
- `MISSING_BUILDABLE_MANDATORY = 0`
- `SIH_CORE_IMPLEMENTED_WITH_DOCUMENTED_LIMITATIONS`
- `SATQUERY_CORE_FREEZE_RECOMMENDED`

This is a complete functional core, not a benchmark, SOTA, broad-generalization, calibrated-confidence, or semantic-performance claim.

## Current AI routes

| Route | Input and execution path | Status |
| --- | --- | --- |
| `SINGLE_IMAGE_VQA` | Sentinel-2 → S2 projector → frozen Qwen | `AVAILABLE` |
| `SINGLE_IMAGE_SAR_VQA` | Sentinel-1 VV/VH `[2,120,120]` → official CROMA S1 `[1,225,768]` → Phase 3P `S1SARProjector` `[1,16,2048]` → frozen Qwen | `AVAILABLE_WITH_LIMITATIONS` |
| `SINGLE_IMAGE_SCENE_DESCRIPTION` | Remote-sensing adapted Qwen2-VL specialist | `AVAILABLE` |
| `OPTICAL_SAR_ANALYSIS` | Sentinel-1 + Sentinel-2 → official CROMA joint encoder → `CromaJointProjector` → frozen Qwen | `AVAILABLE_WITH_LIMITATIONS` |
| `TEMPORAL_CHANGE_DESCRIPTION` | Explicit PRE + POST → Chg2Cap → natural-language change description | `AVAILABLE` |
| `SINGLE_IMAGE_GROUNDING` | Spatial grounding | `BLOCKED` |

The SAR projector checkpoint SHA-256 is `a7e57bff0cba141db5c34f549d0bc3f25638960fe39ab25246bbab49e979accd`.

SAR VQA preserves `SAR_ONLY_SEMANTIC_VALIDATION_LIMITED`. The optical-SAR route preserves `OPTICAL_SAR_TECHNICALLY_VALID_S2_DOMINANT`. No optical fallback is used for the single-SAR route, and grounding never fabricates boxes or masks.

## Unified system

- `SatQueryAgent` input/query classification, compatibility validation, and specialist routing
- fail-closed behavior for incompatible, unsupported, or ambiguous inputs
- response normalization, execution trace, unified `POST /api/v1/query`, local web UI, and JSON exports

## Input foundation

- bounded GeoTIFF/TIFF inspection and PNG/JPEG RGB support where applicable
- explicit sensor/modality/role/band-order declarations; no sensor inference from band count
- Sentinel-2 12-band and Sentinel-1 VV/VH 2-band contracts; PRE/POST temporal roles
- finite-value and CRS checks, corrupt-raster rejection, bounded upload/read limits
- Cartosat-2S and RISAT inspection is supported, but their model inference remains blocked pending domain validation

## Evidence and provenance

Public responses may contain raster metadata, CRS, affine transform, resolution, bounds, valid offline footprint, per-band bounded statistics, nodata/finite fractions, pair bounds overlap, and model/checkpoint provenance. Safe public serialization removes local cache, checkpoint, upload-staging, project, and Hugging Face paths.

`SPATIAL_BOUNDS_OVERLAP != COREGISTRATION_VERIFIED`

`COREGISTRATION_NOT_VERIFIED`

Confidence is deliberately uncalibrated:

```json
{
  "value": null,
  "type": "NOT_AVAILABLE"
}
```

## Runtime and safety

- Python 3.13.14; Torch 2.11.0+cu128; RTX 5060
- one active heavy-specialist family at a time; lazy loading/caching and serialized GPU access
- health/readiness endpoints, startup preflight, graceful shutdown
- safe staging cleanup/recovery, root symlink/reparse containment, and public provenance sanitization

Latest verified single-SAR GPU receipt:

```text
allocated: 8,387,342,336 bytes
reserved:  8,493,465,600 bytes
```

## Phase 3AD browser verification

The approved validation fixture completed the real browser flow: upload, explicit Sentinel-1/SAR/SINGLE/VV,VH declaration, live model request, rendered receipt, and JSON download.

- HTTP `200`; route `SINGLE_IMAGE_SAR_VQA`; actual answer `no`; validation label `yes`
- CROMA/projector route confirmed; no optical fallback
- projected-token correct-vs-shuffled SAR conditioning L2: `35.2681`
- console errors: `0`; page errors: `0`; failed requests: `0`
- API receipt and browser JSON export passed the public path-leak audit
- focused Phase 3AD: Python `44 passed, 2 skipped`; frontend `8 passed`

The validation answer is semantically incorrect for this sample. It demonstrates limited standalone-SAR semantic performance, but does not invalidate the verified functional SAR route, real computational conditioning, or UI/API completion.

Held-out TEST during current VLM/SIH verification: images accessed `0`; labels accessed `0`; inference `0`; metrics `0`.

## Historical sealed Pipeline-3 scientific baseline

This is a separate historical scene-level coverage experiment, not a VQA, captioning, grounding, temporal, or current API metric. Its sealed BigEarthNet v2 5,000-area multimodal baseline used 4,600 train, 200 validation, and 200 held-out test areas.

| Metric | Frozen result |
| --- | ---: |
| Mean absolute error | **4.1034286734 percentage points** |
| Root mean squared error | **9.5873312123 percentage points** |
| Dominant-class accuracy | **65.0%** |

- Scientific fingerprint: `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`
- Dataset fingerprint: `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`
- Split fingerprint: `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`

## Scientific limitations

- No SOTA, benchmark-superiority, or broad generalization claim.
- No demonstrated SAR improvement over optical-only; standalone SAR semantics are limited.
- Single-image VQA has internal validation only.
- Grounding is blocked; no fake boxes or masks.
- Temporal output is natural-language description, not a semantic change mask or area estimate.
- No calibrated confidence and no verified external coregistration.
- Cartosat-2S/RISAT inference-domain validation is unavailable.
- Official VRSBench/RSVQA/CDVQA evaluation is not claimed.

## Externally blocked validation

- VRSBench evaluation
- RSVQA evaluation
- CDVQA evaluation
- Cartosat-2S domain validation
- RISAT domain validation
- final ISRO/SAC evaluation data

## Installation and local run

SatQuery is a local Windows-oriented research runtime and does not automatically download imagery or model weights.

1. Install the pinned runtime and development dependencies.
2. Copy `.env.example` to `.env`, set machine-local paths, and do not commit `.env`.
3. Provide separately acquired approved data and checkpoints for the selected route.
4. Start the service:

   ```powershell
   python -m src.api --host 127.0.0.1 --port 8000
   ```

5. Open `http://127.0.0.1:8000`.

See [the final architecture](docs/SATQUERY_FINAL_ARCHITECTURE.md), [SIH compliance matrix](docs/SIH_26167_COMPLIANCE_MATRIX.md), [scientific limitations](docs/SATQUERY_SCIENTIFIC_LIMITATIONS.md), and [setup instructions](docs/SATQUERY_SETUP_AND_RUN.md).

## Repository layout

```text
src/          application, agent, specialists, API, and browser UI
tests/        API, routing, sensor, runtime, staging, and frontend tests
docs/         architecture, compliance, setup, and scientific records
scripts/      reproducible phase and verification scripts
artifacts/    local receipts and generated evidence (ignored by Git)
datasets/     machine-local datasets (ignored by Git)
checkpoints/  machine-local checkpoints (ignored by Git)
```
