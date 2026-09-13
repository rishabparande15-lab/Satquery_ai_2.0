# Real GEE Temporal Pipeline

**Status: archived Pipeline 2 research extension; reproduction-only and not
reachable from the production API. Temporal prediction remains unsupported.**

This is a new isolated real-GEE temporal pipeline. It never falls back to local files in `--mode gee`.

Configure `GEE_PROJECT`, `GEE_AOI_GEOJSON`, `GEE_BEFORE_START`, `GEE_BEFORE_END`, `GEE_AFTER_START`, and `GEE_AFTER_END` as local environment variables. Authenticate with `earthengine authenticate`, then run:

`GEE_PROJECT` must be a Google Cloud project ID or number registered for Earth Engine with the Earth Engine API enabled. The runner validates all required variables before authentication or retrieval.

```powershell
cmd /c python -m experiments.pipelines.gee_temporal.runner --mode gee
```

The runner reports authentication and retrieval failures without creating fake imagery. The current samples are not used as temporal pairs. The configured NDVI change mask is a documented exploratory threshold, not scientifically validated accuracy.
