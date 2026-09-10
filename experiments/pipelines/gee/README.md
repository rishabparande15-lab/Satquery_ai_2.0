# GEE-Only Experimental Pipeline

This folder contains an isolated, minimal Google Earth Engine-only pipeline scaffold.

## Run

From the project root:

```powershell
cmd /c python -m experiments.pipelines.gee.runner --mode auto
```

`auto` uses authenticated GEE only when `earthengine-api` is installed and `ee.Initialize()` succeeds. Otherwise it runs a clearly labelled local-reference smoke path using samples `61_39`, `61_40`, and `61_41`. Local files are not reported as GEE imagery.

Use `--mode gee` to require real GEE authentication or `--mode local` to force the controlled local-reference path. Outputs are written to `experiments/pipelines/gee/outputs/` and `execution_report.json` records sources, shapes, preprocessing, runtime, errors, and limitations.

No GEE credentials, API keys, before/after dates, coordinates, or temporal results are stored in this repository. The current samples lack sufficient temporal metadata and are suitable for loading/feature smoke tests, not validating before/after change detection.
