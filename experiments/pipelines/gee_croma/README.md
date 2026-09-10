# GEE + CROMA Experimental Pipeline

This folder contains an isolated Google Earth Engine + CROMA experimental pipeline scaffold.

## Run

From the project root:

```powershell
cmd /c python -m experiments.pipelines.gee_croma.runner --mode auto
```

`auto` checks for authenticated GEE access. When it is unavailable, the runner uses the three local samples as controlled references, clearly marks them as non-GEE inputs, and still exercises the verified CROMA adapter and checkpoint. Use `--mode gee` to require real GEE authentication or `--mode local` to force the reference path.

Outputs are written to `experiments/pipelines/gee_croma/outputs/`. The pipeline reuses the existing adapter and checkpoint through an import boundary; it does not copy weights or modify the local pipeline. CROMA produces feature embeddings only. No change map or changed-area percentage is inferred.

No credentials, API keys, invented dates, coordinates, temporal results, or full datasets are stored in this repository. The current samples are suitable for loading and feature extraction tests, but not for validating before/after change detection.
