# Spatial evidence mapping

The hierarchy is pixel → 8×8 block → one 15×15 token → deterministic region → 225-token scene. Tokens are north-up and row-major: `index = 15*row + column`. Token `i` covers columns `[8c,8(c+1))` and rows `[8r,8(r+1))`.

For sample `61_39`, CRS is EPSG:32633 and affine transform is `[10,0,373200,0,-10,5353200,0,0,1]`. Token 0 has pixel bounds `[0,0,8,8]` and map bounds `[373200,5353120,373280,5353200]`. Token 224 has pixel bounds `[112,112,120,120]` and map bounds `[374320,5352000,374400,5352080]`. Coordinates are derived only from the stored affine; none are invented.

Selected token features are mean NDVI, NDWI, MNDWI, NDBI, BSI, VV, VH, VV−VH, 5×5 NDVI/VV local standard deviation, and NDVI edge strength. Each token also carries the exact valid pixel count and ratio. Full Pass 5B statistics remain available in the source probe but are not copied indiscriminately.

Regions use deterministic four-neighbor connectivity over threshold-positive tokens. They are labelled `DETERMINISTIC_EVIDENCE_REGION`, never segmentation. Region ordering begins with the lowest token index and member indices are sorted. Thresholding is recorded per region, and every member retains exact pixel/projected extents.

The visual diagnostic has three left-to-right panels: north-up RGB with exact token grid, NDVI evidence, and RGB overlaid with deterministic NDVI regions. Shared boundaries pass visual orientation checks. Learned CROMA receptive fields can extend beyond a nominal 8×8 anchor, so token co-location does not imply a strictly block-limited receptive field.
