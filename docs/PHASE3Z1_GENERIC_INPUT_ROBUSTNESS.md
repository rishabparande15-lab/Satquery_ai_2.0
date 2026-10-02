# Phase 3Z.1 — Generic input integration and robustness

Status: `PHASE3Z1_COMPLETE`

This phase extends the Phase 3Z metadata foundation without training, changing a checkpoint, adapting a sensor model, downloading a benchmark, or accessing test data.

## Input contracts

`src/generic_raster.py` accepts bounded TIFF/GeoTIFF, PNG, and JPEG inputs. It reports safe metadata, including a content SHA-256, dimensions, band count, dtype, CRS, transform, resolution, bounds, nodata declarations, and full bounded finite-value checks. It does not infer a satellite sensor from pixels or band count.

Explicit GenericRGB is compatible only with `SINGLE_IMAGE_SCENE_DESCRIPTION`. Generic multiband files remain `GENERIC_MULTISPECTRAL` unless a user makes a compatible explicit declaration. A 12-band file does not become Sentinel-2, and a 2-band file does not become Sentinel-1, by inspection alone.

The bounded limits are 32 MiB per request, 4,000,000 band-pixels, and 32 bands. Inputs exceeding a limit are rejected rather than resampled, partially read, or sent to a model. Corrupt and unsupported files return structured input errors. Non-finite values block model compatibility; zero/constant values remain visible as warnings and are never repaired.

## Pair policy

`POST /api/v1/raster/pair/inspect` validates external pair declarations only. Optical/SAR requires declared `S1` and `S2`; temporal requires distinct declared `T1` and `T2`, representing PRE and POST. Geospatial pairs require matching CRS and overlapping finite bounds. No alignment is performed. Even when basic checks pass, the response remains `FOUNDATION_ONLY` and includes `COREGISTRATION_NOT_VERIFIED`.

The existing approved BigEarthNet and LEVIR-CC route loaders are preserved. The new pair endpoint is an inspection and compatibility boundary; it does not reroute or adapt existing models.

## Live checks

Headless Google Chrome used the real local frontend and backend. A validation-safe LEVIR-CC RGB PNG was uploaded as explicit GenericRGB and produced the real scene-specialist response: “There are some green trees on the wasteland.” The browser showed the model route and result with zero console errors, page errors, or failed requests.

The inspection panel also verified Cartosat-2S and RISAT uploads as readable but model-blocked with `SENSOR_DOMAIN_NOT_VALIDATED`. An undeclared 8-band TIFF returned `MISSING_SENSOR_DECLARATION`; no sensor was guessed and no specialist executed.

Fresh real-browser sessions also reconfirmed external explicitly declared Sentinel-2 VQA, local approved S1+S2 analysis, and validation-safe PRE/POST Chg2Cap inference. Their individual receipts are in `artifacts/final/input/phase3z1/`.

`TEST_ACCESS = 0` throughout this phase.

## Remaining gap

Cartosat-2S and RISAT are safely supported for inspection, declaration, and provenance only. Their model compatibility has not been claimed. Any sensor-specific evaluation or adaptation requires a new, separately authorized phase.
