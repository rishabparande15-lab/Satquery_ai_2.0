# SatQuery setup and run guide

## Scope

This guide starts the frozen local release candidate. It does not download datasets or model weights, run training, or enable unsupported sensors.

## Validated runtime

- Windows x64
- CPython 3.13.14
- PyTorch 2.11.0 with CUDA 12.8
- NVIDIA GPU used for final rehearsal: GeForce RTX 5060 Laptop GPU with 8123 MiB reported VRAM
- Google Chrome 154.0.8037.58 used for browser rehearsal

Other environments may work, but the final browser evidence applies only to this validated runtime.

## Create an environment

From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-satquery-runtime.txt
pip install -r requirements-dev.txt
```

Use the official CUDA-enabled PyTorch wheel specified by `requirements-satquery-runtime.txt`. Do not substitute a CPU-only wheel for a final GPU demonstration.

## Configure local paths

Copy the example configuration. The values below are placeholders and must point to machine-local, separately acquired assets.

```powershell
Copy-Item .env.example .env
```

Set these values in `.env` or the current PowerShell session:

```powershell
$env:SATQUERY_LOCAL_DATA_ROOT = 'C:\path\to\satquery-data'
$env:DATASET_ROOT = 'C:\path\to\approved-bigearthnet-development-root'
$env:FEATURES_ROOT = 'C:\path\to\croma-features'
$env:CROMA_SOURCE = 'C:\path\to\croma-official-source'
$env:CROMA_CHECKPOINT = 'C:\path\to\CROMA_base.pt'
$env:SATQUERY_SCENE_DESCRIPTION_MODEL = 'C:\path\to\remote-sensing-Qwen2-VL-2B-Instruct'
```

For the temporal specialist, the release currently expects these repository-local, ignored assets:

```text
checkpoints/chg2cap/LEVIR_CC_batchsize_32_resnet101.pth
artifacts/final/temporal/phase3v2_change_description/_source/Chg2Cap-main/
artifacts/final/temporal/phase3v2a_inference_unblock/rscama_vocab.json
datasets/levir_cc/development_smoke/images/val/A/val_000001.png
datasets/levir_cc/development_smoke/images/val/B/val_000001.png
```

The release does not fetch these files automatically. Keep datasets, checkpoint files, model caches, `.env`, and credentials outside Git.

## Start the application

```powershell
.\scripts\start_satquery.ps1 -Port 8000
```

This runs the non-loading preflight first and stops with actionable missing-asset errors. To inspect readiness without starting the service, run `.\scripts\preflight_satquery.ps1`.

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The frontend is served by the same loopback process. Stop the process with `Ctrl+C` when finished.

## Run the approved demo

Follow [SATQUERY_FINAL_DEMO_GUIDE.md](SATQUERY_FINAL_DEMO_GUIDE.md). Use only the approved non-test inputs. Do not substitute test examples, ingest new data during a presentation, or claim that the generated wording is benchmark-scored.

For automated final rehearsal, a local Chrome executable can be supplied without changing production code:

```powershell
$env:SATQUERY_BROWSER_EXECUTABLE = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
$env:SATQUERY_BASE_URL = 'http://127.0.0.1:8000'
```

## Focused regression tests

```powershell
python -m pytest tests/test_satquery_agent.py tests/test_unified_api.py tests/test_temporal_change.py tests/test_temporal_api.py tests/test_single_image_api.py tests/test_satquery_v1.py tests/test_agent_controller.py -q
node --test tests/frontend_state.test.cjs
```

The final rehearsal recorded 37 passing Python tests and 7 passing frontend tests.

## Inspect an external GeoTIFF safely

Start the local server, then call `POST /api/v1/raster/inspect` with an explicit declaration. This endpoint inspects and gates an external file but does not run a model. See [PHASE3Z_GENERIC_INPUT_AND_SENSOR_ADAPTERS.md](PHASE3Z_GENERIC_INPUT_AND_SENSOR_ADAPTERS.md) for the declaration schema and a PowerShell example.

## Common Windows issues

- **CUDA unavailable:** verify that a compatible NVIDIA driver and CUDA-enabled PyTorch wheel are active. Do not represent CPU fallback as validated GPU demonstration evidence.
- **Dataset selector empty:** verify `DATASET_ROOT` points to the approved non-test development root and that its layout matches the existing loader.
- **CROMA route unavailable:** verify both `CROMA_SOURCE` and `CROMA_CHECKPOINT` are present and readable.
- **Scene route unavailable:** set `SATQUERY_SCENE_DESCRIPTION_MODEL` to the local model directory.
- **Temporal route unavailable:** restore the pinned Chg2Cap checkpoint, source directory, and compatible 501-token vocabulary listed above. Do not reconstruct vocabulary from forbidden test captions.
- **Browser automation unavailable:** use an existing Chrome or Edge executable through `SATQUERY_BROWSER_EXECUTABLE`; avoid changing the validated model environment solely for browser automation.

## Security and data policy

The loopback server binds only to `127.0.0.1` or `localhost`. Do not expose it publicly without a separate security review. Do not commit datasets, checkpoints, Hugging Face caches, local runtime logs, `.env`, API keys, credentials, or generated local artifacts.
