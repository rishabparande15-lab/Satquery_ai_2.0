# Real GEE + CROMA Temporal Pipeline

This isolated pipeline is reserved for real Sentinel-1/Sentinel-2 retrieval from GEE followed by CROMA representation extraction. It never uses local fallback data in `--mode gee` and does not treat CROMA as a change detector.

Set `GEE_PROJECT`, `GEE_AOI_GEOJSON`, and before/after date variables, run `earthengine authenticate`, then execute:

`GEE_PROJECT` must be a registered Google Cloud project ID or number with the Earth Engine API enabled. Configuration is validated before authentication or retrieval.

```powershell
cmd /c python -m experiments.pipelines.gee_croma_temporal.runner --mode gee
```

The verified checkpoint at `D:\Satquery_ai datasets\checkpoints\CROMA_base.pt` is referenced, not copied. Reports distinguish authentication, retrieval, and feature-extraction failures.