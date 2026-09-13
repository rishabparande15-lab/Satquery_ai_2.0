# Pipeline 2 Archive: GEE-Only Experiment

**Status: archived and reproduction-only. This is not a production entry point.**

This folder preserves the isolated, minimal Google Earth Engine-only scaffold.
The supported application runs with `python -m src.api`. Pipeline 3 does not
import this package; the small shared GEE mode helper now lives in
`src/gee_experiment_runtime.py`.

## Historical reproduction

From the project root:

```powershell
cmd /c python -m experiments.pipelines.gee.runner --mode auto
```

`auto` uses authenticated GEE only when `earthengine-api` is installed and `ee.Initialize()` succeeds. Otherwise it runs a clearly labelled local-reference smoke path using samples `61_39`, `61_40`, and `61_41`. Local files are not reported as GEE imagery.

Use `--mode gee` to require real GEE authentication or `--mode local` to force the controlled local-reference path. Outputs are written to `experiments/pipelines/gee/outputs/` and `execution_report.json` records sources, shapes, preprocessing, runtime, errors, and limitations.

No GEE credentials, API keys, before/after dates, coordinates, or temporal results are stored in this repository. The current samples lack sufficient temporal metadata and are suitable for loading/feature smoke tests, not validating before/after change detection.
