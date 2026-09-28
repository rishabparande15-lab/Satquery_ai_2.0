# Phase 2B Spatial and Raster Contract

## Contract identity

The authoritative additive contract is `spatial_contract_v1` in [src/spatial_contract.py](../src/spatial_contract.py). Existing scientific calculations remain authoritative and unchanged. The existing evidence mapping version is `north_up_row_major_120_to_15_v1`.

## Coordinate spaces

| Space | Meaning | Coordinates |
|---|---|---|
| `PIXEL` | Image/raster boundary coordinates | `x = column`, `y = row`; continuous; origin at upper-left; half-open image extent `[0,width] x [0,height]` |
| `ANALYSIS_GRID` | The validated 120×120 common scientific grid | Same row/column orientation and pixel-boundary semantics as `PIXEL` |
| `CROMA_TOKEN` | A discrete 15×15 spatial token grid | `token_index = row * 15 + col`, row-major, 225 tokens |
| `NORMALIZED_IMAGE` | Unit image boundary coordinates | `x = column / width`, `y = row / height`; `[0,1] x [0,1]` |
| `GEO` | CRS coordinates from a validated affine transform | `x/y` are map coordinates; longitude/latitude is not assumed |

Rows and columns are never silently exchanged with x/y. In structured geometry, x always means column/easting and y always means row/northing; row/column pairs are exposed only by token index helpers.

## Raster convention

Pipeline 3 validates a 120×120 B02 common grid at 10 m resolution. Optical bands are ordered `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12`; SAR is `VV,VH`. Optical inputs are reprojected to the B02 grid with bilinear resampling. SAR bands and categorical reference maps require matching CRS, bounds, dimensions, and transform; reference maps use nearest semantics and integer class values. The common transform is north-up: positive x scale, negative y scale, zero rotation.

Native source bands can have 10 m, 20 m, or 60 m shapes, but the scientific representation after loading is the common 120×120 analysis grid. This contract describes that validated result and does not change resampling.

## Pixels and geometry

Pixel coordinates are continuous boundary coordinates. Integer pixel `(col,row)` addresses the cell `[col,col+1) x [row,row+1)`. Bounding boxes use positive extent and half-open interpretation `[x_min,x_max) x [y_min,y_max)`. A box or point outside the declared extent is invalid; it is never silently clipped. Validation distinguishes `valid`, `out_of_bounds`, `malformed`, and `degenerate`.

Points on the outer boundary are valid boundary coordinates for deterministic transforms. A point or box is not implicitly converted between spaces. Source annotation values remain separately preserved by `annotation_foundation`; the new contract does not overwrite them.

## Analysis and normalized transforms

`pixel_to_normalized`, `normalized_to_pixel`, `analysis_to_normalized`, and `normalized_to_analysis` are linear boundary-coordinate transforms. They require valid source-space geometry and explicit dimensions. Round trips preserve deterministic floating-point values within ordinary arithmetic precision.

## CROMA token geometry

CROMA returns 225 spatial vectors with 768 dimensions per token. The repository’s established ordering is row-major. The 120×120 analysis grid maps to 15×15 blocks of 8×8 pixels:

- token 0: pixel bounds `[0,0,8,8]`
- token 112: pixel bounds `[56,56,64,64]`
- token 224: pixel bounds `[112,112,120,120]`

`token_row_col_to_index`, `token_index_to_row_col`, `analysis_to_token`, `pixel_to_token`, `token_to_analysis`, and `token_to_pixel` implement this correspondence. This is geometric correspondence only, not learned grounding.

## GEO and CRS

`RasterMetadata` records width, height, CRS, affine transform, bounds, resolution, and role. `image_to_geo` and `geo_to_image` are permitted only when both CRS and a validated north-up affine transform are present. The affine follows rasterio/GDAL boundary semantics:

`map_x = a * column + b * row + c`

`map_y = d * column + e * row + f`

No longitude/latitude interpretation is made from a projected CRS. BigEarthNet.txt source boxes and points remain `source_unit_square_unmapped` because their axis origin, image relation, and raster mapping are not established by the source contract. They must not be called geographic grounding.

## Evidence compatibility

Existing evidence remains `pixel -> token -> region -> claim -> provenance`. `evidence_schema.py` continues to own its current `spatial_evidence_v1` serialization and `token_extent` map-bound calculation. The new contract supplies reusable validation and transform semantics; it does not alter claims, thresholds, CROMA values, or interpretation behavior.

## Provenance and determinism

Every transformed object carries an explicit coordinate space. Raster metadata must be supplied by a validated artifact or receipt. `spatial_fingerprint` uses sorted UTF-8 JSON and SHA-256. Missing CRS, transform, dimensions, or incompatible spaces fail closed.
