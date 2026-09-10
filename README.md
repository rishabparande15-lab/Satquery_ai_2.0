# Satquery_ai_2.0

Phase 1's strict BigEarthNet/CROMA foundation and integration boundary are documented in [docs/phase1_integration.md](docs/phase1_integration.md).

Phase 3's typed, source-independent operational satellite-data backbone is documented in [docs/phase3_core_satellite_orchestration.md](docs/phase3_core_satellite_orchestration.md).
Phase 3.5's real end-to-end verification is documented in [docs/phase3_5_end_to_end_verification.md](docs/phase3_5_end_to_end_verification.md).
Phase 2's typed bridge into the existing physical+CROMA fusion pipeline is documented in [docs/phase2_integration.md](docs/phase2_integration.md).

## SatQuery AI web application

The local application wraps the validated GEE-style/CROMA/hybrid pipeline. It supports real optical-only, SAR-only, and joint analysis from local samples or canonically described multiband GeoTIFF uploads, with AOI intersection/date validation, a 15-step trace, evidence references, honest confidence status, and persistent JSON downloads.

Start it from the project root:

```powershell
& 'E:\Python313\python.exe' -m src.api --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, choose sample `61_39`, and run the default joint optical/SAR query. This performs actual local raster validation, 62 GEE-style physical features, official CROMA representations pooled to 2,304 values, and deterministic untrained fusion to 192 values. It deliberately does not emit land-cover, water, segmentation, or change predictions.

For a no-server demo report:

```powershell
& 'E:\Python313\python.exe' -c "from src.analysis_engine import run_analysis, save_report; print(save_report(run_analysis({'query':'Analyze the optical and SAR characteristics of this local region.','sample_id':'61_39'})))"
```

Temporal requests without spatially corresponding, dated before/after images are rejected. A pair that passes validation still returns an explicit unavailable status: the web temporal algorithm is not implemented. The three development samples are not a temporal pair.

### QA release behavior

| Mode | Physical features | CROMA pooled values | Hybrid values |
| --- | ---: | ---: | ---: |
| Optical only | 52 | 768 | Not applicable |
| SAR only | 9 | 768 | Not applicable |
| Joint optical + SAR | 62 | 2304 | 192 |

The joint 62-feature schema is preserved for cache/checkpoint compatibility. Its two legacy SAR relationship features are clipped and are explicitly labelled as unsuitable for physical dB-ratio interpretation. SAR-only mode computes VV/VH channel statistics and the unclipped mean VV-minus-VH in the input units. No dummy modality is supplied to CROMA. The official model is cached once per source/checkpoint/device, with serialized inference and an HTTP 409 busy response for overlapping analyses.

Uploads are limited to 32 MiB combined (including form overhead), 4 million band-pixels per TIFF, and explicit optical/SAR/before/after roles. Optical uploads require 12 bands; SAR requires VV/VH. Canonical band descriptions are honored, otherwise the user must confirm the documented band order. GeoTIFFs with invalid CRS, missing values or incompatible grids are rejected. CROMA requires 120x120 pixels; other validated sizes yield physical features with an explicit partial result. AOI coordinates check intersection only; statistics cover the entire raster, without implicit cropping. Dates filter actual acquisition metadata rather than relabelling demo data.

The API accepts opaque upload IDs, not server paths. Consumed uploads are removed before the response; abandoned uploads expire after 15 minutes (30-second cleanup sweep). Reports persist across restarts and never overwrite an existing ID. Keep the application on loopback: it is a development server, not an authenticated public deployment. The UI times out after 120 seconds; it does not interrupt an already running native model operation.

Install runtime dependencies using requirements.txt and QA dependencies using requirements-dev.txt. Node.js is needed for the pure JavaScript state tests. Run the complete checks:

```powershell
& 'E:\Python313\python.exe' -m pytest -q
& 'E:\Python313\python.exe' -m compileall -q src tests
node --test tests/frontend_state.test.cjs
& 'E:\Python313\python.exe' -m src.qa_release
& 'E:\Python313\python.exe' -m src.qa_runtime
```

The release audit requires the website to be running on port 8000 and uses real sample 61_39. It writes timestamped reports under experiments/outputs/qa_release. qa_runtime measures separate-process startup and audits the previously stored native temporal exports without retrieving or changing them. Browser visual verification remains unconfirmed because computer-use automation stopped on its current-URL policy check; HTTP and pure-JavaScript tests do not substitute for that verification.

## CROMA Feature Inspection

The verified BigEarthNet samples and saved CROMA features remain outside the project on D:. Run the inspection from the project root with:

```powershell
cmd /c python -m src.inspect_features
```

Visualizations are written to `D:\Satquery_ai datasets\croma_features\inspection\` and the machine-readable summary is written to `D:\Satquery_ai datasets\croma_features\inspection_report.json`. The module uses the existing loader and preprocessing path, preserves `[12, 120, 120]` optical and `[2, 120, 120]` SAR tensors, and treats CROMA encodings as 225-token (`15 x 15`) maps.

## Hybrid BigEarthNet Proof Of Concept

The supplied `bigearthnet-v2-three-samples.zip` is preserved at `D:\Satquery_ai datasets\bigearthnet-v2-three-samples.zip`. The local hybrid proof of concept uses its extracted copy, combines physical/geospatial features with official CROMA pooled optical, SAR, and joint representations, and writes compact reports under `experiments/outputs/hybrid_poc/`.

Run inspection:

```powershell
& 'E:\Python313\python.exe' -m src.inspect_dataset --dataset-root 'D:\Satquery_ai datasets\extracted\small-sample'
```

Run the end-to-end hybrid pipeline:

```powershell
& 'E:\Python313\python.exe' -m src.hybrid_pipeline --dataset-root 'D:\Satquery_ai datasets\extracted\small-sample' --output-root 'E:\SatQuery_ai_2.0\Satquery_ai_2.0\experiments\outputs\hybrid_poc'
```

The three samples are single-date patches, not temporal pairs. The prototype performs feature extraction and fusion only; supervised training and scientific accuracy claims require a larger labeled dataset and valid before/after pairs.

## Training-Ready Workflow

When a larger labeled BigEarthNet-compatible root is supplied, prepare reproducible splits and compact cached features:

```powershell
& 'E:\Python313\python.exe' -m src.prepare_training --dataset-root '<DATASET_ROOT>' --cache-root 'experiments/outputs/training_ready/feature_cache' --split-manifest 'experiments/outputs/training_ready/splits.json' --device cuda
```

Train the configurable 19-class multi-label head only after enough samples are available:

```powershell
& 'E:\Python313\python.exe' -m src.train_landcover --cache-root 'experiments/outputs/training_ready/feature_cache' --split-manifest 'experiments/outputs/training_ready/splits.json' --output-root 'experiments/outputs/training_ready/model' --device cuda
```

Evaluate only the untouched test split:

```powershell
& 'E:\Python313\python.exe' -m src.evaluate_landcover --cache-root 'experiments/outputs/training_ready/feature_cache' --split-manifest 'experiments/outputs/training_ready/splits.json' --checkpoint 'experiments/outputs/training_ready/model/best.pt' --output 'experiments/outputs/training_ready/evaluation/test_report.json' --device cuda
```

Training uses BCE-with-logits, sigmoid probabilities, micro/macro F1, per-class precision/recall/F1, Hamming loss, exact-match accuracy, and mean average precision. Confidence is explicitly marked uncalibrated until calibration data is available.
