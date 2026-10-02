# Phase 3Z — generic remote-sensing input and sensor adapter foundations

Status: `PHASE3Z_FOUNDATION_COMPLETE`

Phase 3Z adds a bounded, provenance-preserving external GeoTIFF inspection layer. It does not train a model, change a checkpoint, alter a metric, acquire a dataset, or establish Cartosat-2S/RISAT model generalization.

## Implemented flow

```text
User GeoTIFF
  -> Generic Raster Inspector
  -> Explicit sensor, modality, role, and band-order declaration
  -> Adapter foundation
  -> Compatibility gate
  -> eligible existing SatQuery route, or structured block
```

The inspector records actual source facts: driver, filename, byte size, SHA-256, width, height, band count, dtypes, CRS, transform, resolution, bounds, nodata values, band descriptions, color interpretation, and bounded full-read finite-value checks. It accepts only bounded GeoTIFF inputs in this phase. A larger raster fails inspection instead of being resampled, sampled, or silently transformed.

## Explicit declaration contract

Every inspected external file requires:

```json
{
  "sensor": "cartosat-2s",
  "modality": "optical",
  "role": "SINGLE",
  "band_order": ["red", "green", "blue"]
}
```

Allowed sensor values are `sentinel-2`, `sentinel-1`, `cartosat-2s`, `risat`, `generic-rgb`, `generic-multispectral`, and `generic-sar`. Allowed roles are `SINGLE`, `S1`, `S2`, `T1`, and `T2`.

The system normalizes only declared aliases. It never identifies a sensor from pixels, invents band names, reorders bands, or resamples a grid.

## Adapter and compatibility policy

| Declared source | Adapter status | Model compatibility |
| --- | --- | --- |
| Sentinel-2 | Declared compatibility check | Existing VQA route can proceed only with the exact 12-band, 120×120, CRS-bearing B01–B12 contract and an explicit canonical-order confirmation. |
| Sentinel-1 | Declared compatibility check | External joint S1/S2 execution is not integrated by this foundation. |
| Cartosat-2S | Foundation only | `SENSOR_DOMAIN_NOT_VALIDATED`; inspection and provenance only. |
| RISAT | Foundation only | `SENSOR_DOMAIN_NOT_VALIDATED`; inspection and provenance only. |
| Generic RGB | Declared compatibility check | Can be assessed for the existing explicit-RGB scene-description contract, with no sensor or benchmark-generalization claim. |
| Generic multispectral/SAR | Foundation only | No specialist contract is inferred. |

Cartosat-2S is never treated as Sentinel-2. RISAT is never treated as Sentinel-1.

## API

`POST /api/v1/raster/inspect` accepts multipart form fields:

- `raster`: one GeoTIFF file
- `declaration`: JSON sensor declaration
- `requested_route`: optional compatibility target

The response includes `inspection`, `sensor_declaration`, `adapter`, `compatibility_gate`, and `agent_handoff`. Inspection alone never triggers a model. A blocked result reports why and confirms that no band reordering, resampling, or model execution occurred.

Example PowerShell request:

```powershell
$declaration = '{"sensor":"cartosat-2s","modality":"optical","role":"SINGLE","band_order":["red","green","blue"]}'
curl.exe -X POST http://127.0.0.1:8000/api/v1/raster/inspect `
  -F "raster=@C:\path\to\image.tif" `
  -F "declaration=$declaration" `
  -F "requested_route=SINGLE_IMAGE_SCENE_DESCRIPTION"
```

The expected Cartosat-2S or RISAT compatibility result is `BLOCKED` with code `SENSOR_DOMAIN_NOT_VALIDATED`.

## Existing upload hardening

The existing external 12-band VQA upload path now requires an explicit Sentinel-2 declaration in addition to the existing canonical band-order confirmation. The browser UI makes this declaration visible. An undeclared external file cannot be treated as Sentinel-2 by default.

## Verification

- Generic metadata inspection and finite-value checks: passed.
- Missing declaration rejection: passed.
- Cartosat-2S foundation and blocked compatibility gate: passed.
- RISAT foundation and blocked compatibility gate: passed.
- Explicit Sentinel-2 VQA band contract: passed.
- Live local API call with a synthetic declared Cartosat-2S GeoTIFF: `INSPECTED`, adapter `cartosat-2s-foundation`, gate `BLOCKED`, code `SENSOR_DOMAIN_NOT_VALIDATED`, agent handoff `false`.
- Regression: 44 Python tests passed; 7 frontend tests passed.

## Scientific boundary

The evidence-based SIH counts remain unchanged: 17 criteria, 4 complete, 8 complete-with-limitations, 4 partial, and 1 blocked. Phase 3Z makes external input handling safer; it does not complete a sensor-domain model-validation criterion.
