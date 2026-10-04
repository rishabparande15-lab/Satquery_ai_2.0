# Phase 3W.1R — BigEarthNet local-root recovery

Status: `PHASE3W1_COMPLETE`

Overall Phase 3W status: `PHASE3W_COMPLETE`

The approved local BigEarthNet root was recovered without downloading data or modifying persistent configuration:

`D:\Satquery_ai datasets\comparison\raw-1000`

The runtime-only `DATASET_ROOT` setting selected that root for verification. The historical repository default was absent, so it was not used.

## Verified non-test inputs

| Capability | Identity | Split | Verification |
| --- | --- | --- | --- |
| Single image | `S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11` | train | Readable finite 12 × 120 × 120 Sentinel-2 tensor; metadata and loader agree. |
| Optical + SAR | `S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22` with `S1B_IW_GRDH_1SDV_20170717T064605_29UPV_35_22` | validation | Readable finite S2 12 × 120 × 120 and S1 2 × 120 × 120 tensors; exact metadata pairing verified. |

`metadata.parquet` SHA-256: `67533b5ca46459758567f19d5d3a427b0416a0348db551152a17e641ac8cd861`.

No TEST split records, imagery, labels, inference, or metrics were accessed (`TEST_ACCESS = 0`).

## Real browser completion

Using headless Google Chrome against the real local application, the browser made three non-mocked `POST /api/v1/query` requests:

- `SINGLE_IMAGE_VQA` completed with a real result and JSON export.
- `OPTICAL_SAR_ANALYSIS` completed with a real result and JSON export.
- `SINGLE_IMAGE_GROUNDING` returned the expected structured `BLOCKED` response, with no fabricated spatial evidence.

The happy-path browser run had zero console errors, page errors, and failed network requests. The screenshots are retained in `artifacts/final/agent/phase3w1_browser_completion/`.

Grounding is intentionally still unavailable until a validated grounding specialist and trustworthy coordinate mapping are admitted. Existing single-image and optical/SAR scientific limitations remain visible in their API responses.

The exact next phase is `PHASE3X`.
