# SatQuery AI

SatQuery AI is a local, evidence-first remote-sensing research system for validated Sentinel-1/Sentinel-2 ingestion, CROMA and physical representations, controlled scientific evaluation, spatial evidence, and constrained human-readable answers.

## Current project status

**YELLOW — PARTIALLY VALIDATED.** The repository has a validated S1/S2 scientific foundation, strict raster and split controls, official CROMA execution, physical features, controlled hybrid-predictor experiments, deterministic pixel/token evidence, constrained interpretation, provenance, a loopback API, and a tested frontend. Validation is limited to the documented BigEarthNet populations and targets. Arbitrary VQA, captioning, learned grounding, temporal/change prediction, calibrated confidence, and a complete agentic controller remain open.

The browser/API runtime performs feature and evidence analysis. It does not load a trained task head or emit land-cover, object, segmentation, or change predictions. Supervised predictors described below belong to controlled scientific evaluation workflows.

## Project objective

The long-term objective is a multimodal and multitemporal platform that accepts satellite observations and grounded questions, chooses a scientifically appropriate workflow, and returns evidence-backed results. The current system covers single-date optical, SAR, and joint representation analysis plus spatial evidence and controlled interpretation. Multitemporal prediction and open-ended vision-language behavior are future work.

## Current capabilities

- Sentinel-1 VV/VH and canonical 12-band Sentinel-2 ingestion.
- Strict identity, band, CRS, transform, extent, orientation, finite-value, AOI, and date validation.
- B02-grid alignment and the established CROMA normalization profile.
- Official CROMA optical, SAR, and joint token/scene representations.
- Optical/SAR physical statistics and spectral/radar relationships.
- Current physical+CROMA hybrid representation and controlled trained-probe evaluation.
- Pixel, local-window, edge, and 8×8 block aggregation as a spatial sidecar.
- A 15×15 / 225-token evidence model, deterministic regions, traceable claims, and provenance.
- Fail-closed, template-based human-readable interpretation without an LLM/VLM.
- Persisted scientific artifacts, hashes, manifests, predictions, metrics, and reports.
- Loopback HTTP API, upload validation, immutable JSON reports, and a browser frontend.

## Scientific architecture

```text
Sentinel-1 / Sentinel-2
          ↓
Strict validation and identity checks
          ↓
Alignment and preprocessing
          ↓
Official CROMA representations
          +
Physical feature vector
          ↓
Physical + CROMA hybrid representation
          ↓
Validation-selected scientific prediction head
          ↓
Held-out evaluation, metrics, leakage audit, provenance
```

Validation rejects malformed or mismatched inputs. Preprocessing creates CROMA-compatible tensors without relabelling or silently inventing data. CROMA and physical features are separate inputs to the hybrid representation. In controlled experiments, train-only scaling and training fit the predictor, validation selects the epoch, and the test split is used for final evaluation. The web runtime stops at representations/evidence because it has no deployed trained task head.

## Pixel and spatial-evidence architecture

```text
Raw pixels
    ↓
Pixel features and local statistics
    ↓
Non-overlapping 8×8 blocks
    ↓
15×15 row-major grid / 225 spatial tokens
    ↓
Four-connected deterministic evidence regions
    ↓
Validated evidence claims
```

The sidecar retains fine-grained B04/B08, optical indices, VV/VH, polarization difference, 3×3/5×5/7×7 local variability, and NDVI edge information. Block summaries include mean, standard deviation, minimum, maximum, median, p25, p75, and valid ratio. It is an explainability/evidence path, not the default predictor.

## Interpretation architecture

```text
Validated spatial evidence
           ↓
Exact allowlisted claim filter
           ↓
Question route + controlled sentence templates
           ↓
Human-readable answer + structured traceability
```

The adapter preserves epistemic status, evidence strength, sensor, features, token/region IDs, projected extents, source artifacts, and provenance. Unsupported semantics fail closed. It cannot turn vegetation evidence into a forest identity, water evidence into a named river, or pixels into object counts or calibrated confidence.

## Combined system architecture

```text
User
 ↓
Frontend
 ↓
Loopback API
 ↓
Query and input validation
 ↓
Analysis pipeline
 ├── Optical physical + CROMA
 ├── SAR physical + CROMA
 ├── Joint CROMA and physical features
 ├── Current hybrid representation / scientific predictor workflow
 ├── Pixel and local-feature sidecar
 ├── Structured spatial evidence
 └── Constrained interpretation
             ↓
Evidence-grounded response
             ↓
Immutable report, artifacts, hashes, provenance

Future, not completed:
VQA · captioning · learned grounding · temporal/change prediction
confidence calibration · complete agentic controller
```

## Datasets

### Selected BigEarthNet v2 5,000-area archive

`bigearthnet-v2-5000-20260911T162804Z-1-001.zip` contains 5,000 areas, 10,000 Sentinel-1 TIFFs (VV/VH), 5,000 reference maps, and a 5,000-row metadata table with 19 CORINE-style labels. It has no Sentinel-2 rasters and no VQA, caption, box, mask, or change annotations. It supports the leakage-controlled Pass 5 SAR evaluation after excluding all 1,000 Pass 3 areas. The archive stays outside Git.

### Matching local Sentinel-2

`D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2` supplied the matching S2 imagery used by the controlled Pass 3 and Pass 5D reproduction. That 1,000-area population overlaps the selected 5,000-area archive and is not an independent new holdout.

### BigEarthNet.txt

The pinned BigEarthNet.txt linkage audit matched 955 of the 1,000 selected images and 20,453 annotation records. It established identity/linkage and split preservation. Those annotations were inspected for future task planning; they are not an integrated or validated VQA model.

### Fragmented official archive

`bigearthnet-v2-full-official-20260911T162841Z-1-031.zip` contains seven non-contiguous `.tar.zst` fragments without the remaining parts, metadata, labels, or an authoritative reassembly manifest. It is not usable as a standalone training/evaluation source. Dataset contracts live in [`data/contracts`](data/contracts); raw data stays external.

## Model and evidence components

| Component | Input | Output | Purpose |
|---|---|---|---|
| CROMA | Normalized optical `[N,12,120,120]`, SAR `[N,2,120,120]`, or both | Spatial tokens and GAP scene vectors | Frozen pretrained representation extraction |
| Physical pipeline | Canonical optical/SAR rasters | 52 optical, 9 SAR, or legacy joint 62 values | Interpretable scene statistics and indices |
| Current hybrid | 62 physical + 2,304 pooled CROMA values | 192-value scene representation | Default feature representation for the current hybrid scientific predictor |
| Pixel/local sidecar | Raw aligned optical/SAR pixels | Feature maps and 225-token summaries | Spatial detail and explainability |
| Spatial evidence engine | Pixel/token features, geometry, source metadata | `spatial_evidence_v1` scene/tokens/regions/claims | Deterministic traceable evidence |
| Interpretation adapter | Validated evidence + controlled question | Answer, claims, evidence links, provenance | Readable evidence presentation without free-form generation |

The current hybrid remains the default predictor architecture. Pass 5D Version B is not promoted into production.

## Verified CROMA and hybrid outputs

The final real `61_39` API run confirmed:

| Output | Shape |
|---|---:|
| Optical CROMA tokens | `[1,225,768]` |
| SAR CROMA tokens | `[1,225,768]` |
| Joint CROMA tokens | `[1,225,768]` |
| Optical/SAR/joint scene vector, each | `[1,768]` |
| Concatenated pooled CROMA | `[1,2304]` |
| Physical features | `[1,62]` |
| Hybrid representation | `[1,192]` |

## Physical features

The joint schema contains mean, standard deviation, minimum, and maximum for all 12 optical bands and VV/VH, plus mean NDVI, NDWI, MNDWI, NDBI, VV-minus-VH, and VH-over-VV: 62 values total. The two legacy joint SAR relationship values are clipped to `[-1,1]` for checkpoint compatibility and must not be presented as physical dB ratios. Optical-only mode produces 52 values; SAR-only mode produces VV/VH statistics plus mean VV-minus-VH for 9 values.

## Pixel feature layer and Pass 5D decision

Pass 5B investigated whether pixel/local/token summaries add useful spatial or predictive information. Pixel-to-token orientation and aggregation passed exactly. Pass 5D then compared the unchanged current pipeline with a pixel-enhanced candidate on the same 600/200/200 reproduction split:

| Version | MAE pp | RMSE pp | Dominant accuracy |
|---|---:|---:|---:|
| A — current hybrid | 4.884571 | 11.025107 | 59.5% |
| B — pixel/local/token enhanced | 4.846286 | 10.936235 | 58.5% |

The paired MAE change was −0.0383 pp with a 95% interval of `[−0.1211,+0.0435]`, while accuracy decreased one point. This did not establish statistically defensible predictive superiority. Version A remains the default predictor; the pixel/token layer remains the spatial-explainability sidecar.

## Spatial evidence

`spatial_evidence_v1` maps each 120×120 scene into 225 row-major 8×8 blocks, groups thresholded tokens into deterministic regions, and links every claim back to feature values, token IDs, region geometry, source artifacts, and provenance. The canonical `61_39` artifact contains 225 tokens, 17 regions, and three supported claims: vegetation-related optical evidence, water-related optical evidence, and SAR polarization-contrast evidence. Two independent builds produced canonical SHA-256 `5906f4de2dbcb3d541590b57c802eba14e8a5d8585f591d88bc438eb3e6e3d8c`.

No learned segmentation, object detection, or joint optical-SAR semantic claim is inferred merely because both sensors are present.

## Human-readable interpretation

`constrained_interpretation_v1` converts exact validated claims into simple or technical text. A verified generated answer includes:

> The derived optical evidence shows moderate vegetation-related spectral evidence.

> The derived SAR evidence shows moderate polarization-related surface variation.

These are inferred statements over derived evidence, not model predictions or confidence claims. The adapter is deterministic and rejects altered claim wording, unsupported sensors/features, fabricated predictions, fake joint semantics, and unsupported questions.

## Scientific results

The controlled Pass 3 scene-level coverage experiment used a fixed 600/200/200 geographic-area split:

| Model | MAE pp |
|---|---:|
| Constant | 7.6984 |
| Physical | 7.7384 |
| Optical CROMA | 5.5708 |
| SAR CROMA | 5.6489 |
| Joint CROMA | 4.9815 |
| Hybrid | 4.8846 |

Joint CROMA outperformed the individual CROMA branches on this experiment, and the existing hybrid achieved the lowest listed MAE. Scope is limited to the selected population, scene-level labelled-pixel coverage target, split, and evaluation protocol. It is not evidence of general VQA, token grounding, or deployment performance.

An earlier distinct Phase 3.6 token-coverage comparison is retained as historical scientific evidence and must not be conflated with this scene-level table; its target granularity and probe design differ.

## Testing and final checkpoint

Final checkpoint verification on 2026-09-13:

| Gate | Result |
|---|---|
| Python tests | 190 passed |
| Frontend state tests | 7 passed |
| Python compilation | PASS |
| JavaScript syntax/static diagnostics | PASS |
| Real sample `61_39` | PASS |
| Application startup and health | PASS |
| Real analysis/report API round trip | PASS |
| CROMA/physical/hybrid shapes | PASS |
| Pixel/token/region evidence | PASS |
| Supported/adversarial interpretation questions | PASS |

Browser automation was attempted, but the provided Windows computer-use Node runtime could not start. Live HTML/API checks and frontend state tests passed; this does not claim a visual cross-browser review.

## Provenance and reproducibility

- Area identities, source hashes, grids, transforms, normalization, model versions, splits, and environment details are persisted.
- Train/validation/test units are whole geographic areas; joins use canonical identities rather than directory order.
- Pass 3 repeated predictions and metrics are byte-identical in the tested environment.
- Metrics have independent recomputation tests; checkpoint and artifact receipts verify hashes.
- Pass 5C evidence and Pass 5E semantic outputs are deterministic.
- Cross-device bitwise equality is not claimed.

## Repository and data hygiene

Raw imagery, archives, NumPy tensors, checkpoints, environments, credentials, caches, and generated scratch outputs are excluded by `.gitignore`. External dataset/checkpoint paths are machine-local. Small contracts, schemas, manifests, metrics, predictions, scientific reports, and provenance are intentionally preserved for reproducibility. The cleanup and classification record is [`docs/repository_cleanup_report.md`](docs/repository_cleanup_report.md).

## Supported user questions

The constrained adapter currently routes:

- “What is present?”
- “Is there vegetation?”
- “Where is vegetation-related evidence?”
- “Is there water?”
- “Where is water-related evidence?”
- “What does SAR show?”
- “What evidence supports the answer?”
- “Show technical details.”

Scene heterogeneity returns unavailable unless the exact validated heterogeneity claim exists.

## Unsupported capabilities

- Arbitrary VQA and free-form image interpretation.
- Object counting, arbitrary object detection, and exact identity/location claims.
- Caption generation and learned/referring-expression grounding.
- Learned segmentation in the evidence sidecar.
- Temporal/change prediction; only pair validation exists in the web path.
- Calibrated model or query confidence.
- A complete autonomous/agentic task controller.

## Roadmap

1. Scientific foundation — **COMPLETE**
2. Spatial/pixel evidence — **COMPLETE**
3. Constrained interpretation — **COMPLETE**
4. VQA — **NEXT**
5. Captioning and learned grounding
6. Temporal/change modelling
7. Confidence calibration
8. Agentic controller
9. Domain-specific and final evaluation

Roadmap entries after Phase 3 are plans, not implemented capabilities.

## Reproduction

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
```

Edit `.env` to point to external dataset, official CROMA source, and checkpoint locations. Do not copy those assets into Git.

Run the complete automated gate:

```powershell
python -m pytest -q
node --test tests/frontend_state.test.cjs
python -m compileall -q src tests scripts
node --check src/static/app.js
git diff --check
```

Start the local application:

```powershell
python -m src.api --host 127.0.0.1 --port 8000
```

Run sample `61_39` without the server:

```powershell
python -c "from src.analysis_engine import run_analysis; import json; print(json.dumps(run_analysis({'query':'What is present?','analysis_type':'joint_optical_sar_analysis','sample_id':'61_39','files':{}}), indent=2))"
```

Regenerate Pass 5C evidence into an ignored output directory before comparing it with the canonical artifact:

```powershell
python scripts/build_spatial_evidence.py --out experiments/outputs/check/spatial_61_39
```

Run the six-area scientific smoke evaluation:

```powershell
python -m src.scientific_smoke --dataset-root 'D:\Satquery_ai datasets\comparison\raw-1000' --output-root experiments\outputs\smoke-check --per-split 2 --epochs 8 --device cuda
```

The full 600/200/200 Pass 3 and Pass 5D experiments are persisted and should not be rerun merely for a repository check. Their reproduction commands and external prerequisites are documented in [`docs/pass3_scientific_validation.md`](docs/pass3_scientific_validation.md) and [`docs/pass5d_reproducibility.md`](docs/pass5d_reproducibility.md).

## Directory structure

```text
src/
  api.py                         loopback HTTP API
  analysis_engine.py             validated runtime orchestration
  croma_adapter.py               official CROMA boundary
  dataset_loader.py              strict BigEarthNet ingestion
  gee_features.py                physical features
  pixel_features.py              pixel/local/token sidecar
  evidence_schema.py             spatial_evidence_v1
  interpretation_adapter.py      constrained evidence-to-text
tests/                           Python and frontend regression tests
docs/                            scientific, audit, design, and cleanup reports
experiments/
  pass3/                         validated 600/200/200 artifacts
  pass5/                         leakage-controlled SAR generalization
  pass5d/                        A/B predictor comparison
  outputs/                       ignored generated runtime output
artifacts/
  pixel_feature_probe/61_39/     Pass 5B metadata/diagnostic
  spatial_evidence/61_39/        canonical Pass 5C evidence
  interpretation/61_39/          generated Pass 5E answers/provenance
data/contracts/                  small dataset contracts, no imagery
```

## Scientific reports

- [Scientific gap analysis](docs/scientific_gap_analysis.md)
- [Scientific implementation plan](docs/scientific_implementation_plan.md)
- [Pass 3 scientific validation](docs/pass3_scientific_validation.md)
- [Pass 3 leakage audit](docs/pass3_leakage_audit.md)
- [Pass 4 dataset audit](docs/pass4_dataset_quality_audit.md)
- [Pass 5B pixel feasibility](docs/pixel_feature_feasibility.md)
- [Pass 5C spatial evidence audit](docs/spatial_evidence_audit.md)
- [Pass 5D A/B report](docs/pass5d_pixel_vs_baseline_report.md)
- [Pass 5E interpretation audit](docs/interpretation_adapter_audit.md)
- [Final pipeline audit](docs/FINAL_PIPELINE_AUDIT.md)
- [Repository cleanup report](docs/repository_cleanup_report.md)

## Final honest summary

**CURRENTLY VALIDATED**

- Scientific S1/S2/CROMA foundation.
- Physical feature pipeline and controlled hybrid prediction.
- Pixel/local spatial-evidence sidecar.
- Structured deterministic evidence and constrained human-readable interpretation.
- Provenance, artifact integrity, reproducibility, API, and frontend integration.

**NOT YET VALIDATED**

- Arbitrary VQA.
- Captioning.
- Learned grounding.
- Temporal/change prediction.
- Calibrated confidence.
- Complete agentic controller.
