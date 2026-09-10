# SatQuery AI — QA and repair report

Date: 2026-09-09. Final live audit: 17:18:18 UTC.

## 1. Overall status

**PASS WITH LIMITATIONS.** The repaired backend, API, real-data workflows, report downloads, and frontend state logic pass the checks described below. Browser visual/layout/console verification is **not confirmed**: browser control was stopped by the computer-use policy guard because it could not determine Brave's current URL. No alternate browser automation was attempted after that stop. This is not an unconditional visual release sign-off.

The final website was restarted and is running at http://127.0.0.1:8000.

## 2. Files inspected

Inspected the application modules: src/api.py, analysis_engine.py, query_interpreter.py, data_orchestrator.py, input_validation.py, tool_selector.py, multimodal_cube.py; the existing hybrid_pipeline.py, gee_features.py, hybrid_fusion.py, croma_adapter.py, dataset_loader.py, preprocessing.py, temporal.py, train_landcover.py and training_data.py; config.py, .env.example, requirements.txt and README.md; all original tests; original hybrid and application reports; native stored GEE temporal export reports and raster metadata. Read the external official CROMA use_croma.py to verify single-modality encoder interfaces. The official source and checkpoint were not modified.

Frontend was originally embedded in api.py. It is now served from src/static/index.html, app.js and style.css, with all three HTTP resources checked.

## 3. Files modified / created

Modified:

- src/api.py
- src/analysis_engine.py
- src/croma_adapter.py
- src/data_orchestrator.py
- src/input_validation.py
- src/multimodal_cube.py
- src/query_interpreter.py
- src/tool_selector.py
- src/temporal.py
- src/train_landcover.py
- README.md

Created:

- src/raster_inputs.py
- src/modality_features.py
- src/static/index.html
- src/static/style.css
- src/static/app.js
- src/qa_release.py
- src/qa_runtime.py
- tests/test_qa_regressions.py
- tests/frontend_state.test.cjs
- requirements-dev.txt
- this report and timestamped QA execution artifacts under experiments/outputs/qa_release

Existing tests and the established joint feature/fusion/training infrastructure were retained. The repository already had extensive user-owned untracked content and README changes; no reset, checkout, commit, or broad cleanup was performed.

## 4–6. Bugs, root causes and verified repairs

| Issue | Root cause | Repair and verification |
| --- | --- | --- |
| Optical/SAR requests ran a joint analysis | Engine ignored interpreted modalities and always loaded both inputs | Added modality-specific assembly and official encoder calls. Real optical and SAR runs contain only their requested modality and 768 pooled values; no placeholder other modality |
| Uploaded TIFFs could never reach extraction | Upload roles were discarded; engine returned pending band mapping unconditionally | Explicit role fields and canonical descriptions/confirmation now feed actual extraction. Real uploaded joint input produced a 192-value fusion result |
| Integer TIFFs failed inspection | Filling integer masked arrays with NaN raised a conversion error | Read masked float32 values before filling; regression verifies uint16 TIFFs |
| Incorrect bands, CRS, resolution and grids accepted | Validation mostly checked readability | Check bands, CRS, finite/nodata values, transform, extent, dimensions and resolution; missing joint inputs fail explicitly |
| Invalid AOIs accepted and AOIs not applied | Unknown types and coordinate ranges were not rejected; raster intersection never checked | Point/bbox validation and CRS-transformed intersection check. Cropping is explicitly unavailable; full-patch statistics are labelled |
| Invalid date order or month could be silently changed | Parser sorted dates, treated named months as January, and could fall back to a year | Preserve input order, extract named months and raw ISO date components, validate dates centrally, enforce local acquisition ranges |
| Explicit temporal mode could miss temporal validation | Temporal requirement depended on query keywords alone | Mode also forces temporal validation. Invalid pairs reject; validated pairs still report the unimplemented web algorithm as unavailable |
| Standalone temporal helper accepted dates without grids | Helper only checked identifiers and lexically ordered dates | ISO date and geospatial correspondence requirements; empty/non-finite/overflowing representation differences reject |
| CROMA loaded on every request | Adapter was constructed inside each analysis | One shared model keyed by source/checkpoint/device, locked execution, reuse recorded in reports, failed models evicted |
| Failed deep extraction lost useful context | Partial explanation relied on validation errors even when those were empty | Physical statistics are retained, partial status and cause are visible, raw exception details stay out of API reports |
| Reports leaked absolute filesystem paths | Retrieval assets and evidence used internal paths | Public source references and opaque upload IDs; raw server paths rejected at API boundary; finite JSON enforced |
| Reports disappeared after restart | Download route required an in-memory map | Persistent report lookup by bounded UUID filename; immutable report IDs. Download from a previous server process returned HTTP 200 |
| Uploads were unbounded and left behind | No request/raster cap, expiry or ownership lifecycle | 32 MiB request cap, four-million-band-pixel raster cap, 32 pending-upload capacity, validated staging, one-shot consumption, 15-minute expiry and cleanup |
| Success returned before cleanup/lock release | HTTP response was emitted before finally block ran | Complete resource cleanup before sending analysis response; regression checks immediate cleanup and readiness |
| Malformed requests and failures confused clients | Missing schema checks, arbitrary exception strings and inconsistent errors/status codes | Controlled JSON errors with 400/408/409/413/415/422/500 statuses; unknown routes 404; same-origin loopback boundary |
| UI could freeze or retain old results | Implicit window-ID globals, unchecked failed responses, no try/finally, reset or submission guard | Explicit DOM references, scoped request deadline, error recovery, clearing prior results/download, busy controls, reset, and source-aware notices |
| Raw JSON was the only result interface | No dedicated presentation of answer/evidence/confidence | Separate answer, dimensions/runtime/device, evidence, limitations, execution trace and expandable report sections |
| Legacy SAR values could be read as valid physical ratios | The legacy joint provider clips all indices to [-1,1], including SAR relationships | Preserve 62-feature compatibility but explicitly label this limitation. SAR-only uses raw channel statistics and unclipped mean VV-minus-VH in input units |
| Undefined spectral ratios looked like measured zeros | Legacy zero-denominator handling emits a zero placeholder | Report zero-valid-denominator indices as unavailable and label the placeholders |
| Three-sample training could bypass minimum size | CLI --allow-small-dataset bypassed the general sample threshold | Explicit inference-only population guard rejects three or fewer samples before feature loading/training |

## 7. Regression tests added

tests/test_qa_regressions.py adds parameterized coverage for request schema, empty query, unknown mode, AOI shape/ranges/intersection, dates, integer TIFFs, corrupt/unsupported uploads, band counts, CRS/resolution/shape mismatch, missing joint modality, temporal mode and valid-pair unavailable state, absent checkpoint, CROMA failure, CPU selection, opaque file IDs, upload cleanup/expiry, report persistence/immutability, same-origin behavior, request size, concurrency, server timeout, safe internal-error format, standalone temporal safety and the training guard.

tests/frontend_state.test.cjs contains six pure-JavaScript tests for duplicate submits, backend failure with old-result clearing, network failure and timeout recovery, reset, empty-query/missing-upload rejection and server-busy handling. Pytest invokes this harness. These use a small DOM test double; they are not browser tests.

## 8. Full test result

Final command: E:\Python313\python.exe -m pytest -q

**69 passed, 21 warnings in 15.72 seconds.** All original 20 tests remain. Warnings are Rasterio/Affine pending deprecation warnings while constructing test fixtures, not failed assertions.

Python compilation of src and tests passed. node --check src/static/app.js passed. The standalone JavaScript harness passed 6/6 tests.

## 9. Browser test result

**Blocked / not verified.** No browser provider was available in the browser-control interface. The native computer-use fallback located Brave, then stopped because its policy guard could not confidently determine the current browser URL. No browser actions followed that stop.

Therefore no claim is made that the final page was visually inspected, that responsiveness was observed at actual viewport sizes, or that the browser console had zero errors. Static asset delivery and JavaScript syntax/state behavior were tested independently.

## 10. API test result

**79/79 live release checks passed** after the final restart. The audit exercised homepage, CSS/JS resources, health, sample discovery, real analyses, uploads, report downloads, invalid inputs and recovery.

POST /api/analyze returns finite, JSON-serializable reports or controlled errors. HTTP 422 includes a rejected analysis report; malformed requests return HTTP 400. Unsupported upload types use 415, oversized bodies 413, corrupt rasters 422, overlapping inference 409, and forced timeout 408. Internal faults use 500 without returning stack traces or secret exception text.

Consumed uploads were gone when requests completed. Downloaded JSON equalled the API report. A report made by the previous server process also downloaded successfully after restart.

## 11–13. Real optical, SAR and joint demos

Sample 61_39, real raster data and official CROMA checkpoint, final restarted server:

| Mode | Input tensor(s) | Physical dimension | CROMA dimension | Hybrid dimension | First occurrence | Warm repeat | Device |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Optical | [12,120,120] | 52 | 768 | Not applicable | 2.634379 s, includes model load | 0.071978 s | CUDA |
| SAR | [2,120,120] | 9 | 768 | Not applicable | 0.026166 s, model already cached | 0.026957 s | CUDA |
| Joint | Both tensors | 62 | 2304 | 192 | 0.104238 s, model already cached | 0.102485 s | CUDA |

No dimensions were injected into reports to pass QA: dimensions come from actual arrays. Single optical/SAR CROMA vectors match the corresponding slices in the joint vector at atol=1e-5. The joint fusion vector matches the original hybrid pipeline at atol=1e-5.

The original hybrid pipeline independently processed all three development samples in 3.133130 seconds on CUDA, writing to the separate QA legacy_hybrid folder.

## 14. Temporal safety result

The requested 2020–2025 comparison without valid before/after inputs returns HTTP 422, the required spatially corresponding pair message, no prediction, and no fabricated change values or maps.

Stored native GEE exports were audited without new retrieval: all 12 Sentinel-2 before/after band grids match, but the before B1, B2 and B4 rasters contain missing/non-finite pixels. Both SAR band pairs have mismatched extents/transforms. Acquisition dates in the stored metadata are optical 2024-01-18/2024-02-02 and SAR 2024-01-10/2024-02-03. These exports were not silently promoted to a clean multimodal temporal pair.

## 15. Invalid-input result

All specified categories were exercised through API tests, engine tests or JavaScript state tests: no input, unsupported format, corrupt TIFF, wrong optical/SAR band count, missing SAR, incompatible CRS/resolution, non-finite values, empty query, invalid AOI/date range, absent checkpoint, CUDA unavailable, CROMA failure, timeout and duplicate submissions.

The absent-checkpoint and forced-inference-fault tests retain physical statistics with partial status. An actual CPU CROMA run, with CUDA availability disabled only inside its QA process, produced 768 optical values in 2.970227 seconds. The test did not change machine CUDA settings.

## 16. Runtime and resource observations

Clean separate-process startup to successful health response: **1.729376 seconds**, no startup traceback.

Final cold-model analysis: **2.634379 seconds**. A previous cold run took 9.493883 seconds; runtime depends on filesystem/OS caches and concurrent load. Report warm timings separately from cold model load.

Final process RSS after repeated live runs: **2,467,799,040 bytes** (about 2.30 GiB). Warm runs remained around 2.46 GB; no continued large growth was observed over this short audit. This is not a long-duration leak certification.

GPU: NVIDIA GeForce RTX 5060 Laptop GPU. Post-run snapshot: 0% utilization and 2379 MiB used of 8151 MiB total; that is a whole-device idle snapshot, not an inference peak measurement.

## 17. Feature dimensions and scientific status

Optical: 48 band statistics plus four spectral indices = 52 physical values, one 768-value pretrained representation.

SAR: eight band statistics plus mean VV-minus-VH = 9 physical values, one 768-value pretrained representation.

Joint: the preserved 62-value physical schema, three 768-value CROMA pooled vectors, and a 192-value seeded untrained fusion vector.

Task prediction, supervised accuracy and calibrated confidence remain unavailable. The current dataset is never trained. No masks or change maps are fabricated.

## 18. Evidence/report paths

- Final live audit: experiments/outputs/qa_release/20260909T171818Z/audit.json
- Final optical report: experiments/outputs/qa_release/20260909T171818Z/optical-1.json
- Final SAR report: experiments/outputs/qa_release/20260909T171818Z/sar-1.json
- Final joint report: experiments/outputs/qa_release/20260909T171818Z/joint-1.json
- Uploaded joint report: experiments/outputs/qa_release/20260909T171818Z/uploaded-joint.json
- Temporal rejection: experiments/outputs/qa_release/20260909T171818Z/temporal.json
- CPU report: experiments/outputs/qa_release/cpu/80310e02-5dd6-4e62-a595-984bc250c9d6.json
- Startup/native temporal audit: experiments/outputs/qa_release/runtime-20260909T171400Z.json
- Original pipeline rerun: experiments/outputs/qa_release/legacy_hybrid/run_report.json
- Final web-downloadable joint report: experiments/outputs/web_reports/37a46c70-0321-4c7d-a8ff-c1452b713ddc.json

Original ZIP SHA-256 before and after the audit:

fc2aa9371e64fa80ba231061797c9e1070594551dcb4a5cfac6c7b4874ef03e3

No ZIP modification, dataset download or supervised training was performed.

## 19. Remaining limitations

- Final browser visual/console checks are blocked as documented above.
- Remote LLM adapters and fresh GEE retrieval are not active. Query parsing and explanations are deterministic and labelled.
- Trained land-cover/water/VQA/segmentation/change heads and calibrated confidence remain unavailable.
- Web temporal analysis remains unimplemented even when a pair passes metadata validation.
- AOI input supports points and bounding boxes with intersection checking, not polygon/shapefile processing, cropping or interactive maps.
- Cloud masks, cloud filtering and validated SAR denoising are unavailable.
- Inputs outside 120x120 produce physical-only partial results. No undisclosed resizing is performed.
- Evidence includes actual source references and statistics; image/map visualization is not yet implemented.
- Native inference is serialized; the client deadline does not forcibly interrupt a hung native GPU operation.
- This is a loopback development server, not a public authenticated deployment. Reports remain on disk until deliberately managed; old pre-QA reports were preserved and can retain their original schema.
- Legacy SAR clipping remains for joint cache/model compatibility and is explicitly disclosed.

## 20. Exact startup and repeatable QA commands

From E:\SatQuery_ai_2.0\Satquery_ai_2.0:

```powershell
& 'E:\Python313\python.exe' -m src.api --host 127.0.0.1 --port 8000
```

The final server is already running. To rerun the live audit against it:

```powershell
& 'E:\Python313\python.exe' -m src.qa_release
& 'E:\Python313\python.exe' -m src.qa_runtime
```

Required test and compilation checks:

```powershell
& 'E:\Python313\python.exe' -m pytest -q
& 'E:\Python313\python.exe' -m compileall -q src tests
node --test tests/frontend_state.test.cjs
```
