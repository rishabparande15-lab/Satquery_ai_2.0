# Real GEE Classical Temporal Pipeline

This third isolated pipeline performs transparent classical before/after change analysis on real GEE Sentinel-1/Sentinel-2 imagery. It supports absolute differences, normalized differences, SAR log-ratio/ratio extensions, NDVI differences, and documented threshold masks once authenticated retrieval is configured.

Set `GEE_PROJECT`, `GEE_AOI_GEOJSON`, and before/after date variables, run `earthengine authenticate`, then execute:

`GEE_PROJECT` must be a registered Google Cloud project ID or number with the Earth Engine API enabled. Configuration is validated before authentication or retrieval.

```powershell
cmd /c python -m experiments.pipelines.gee_temporal_classical.runner --mode gee
```

No accuracy claim is made without ground truth. No local fallback is used and no fake GEE output is written when authentication or retrieval fails.