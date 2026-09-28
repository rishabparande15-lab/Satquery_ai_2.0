# Phase 2D Foundation: BigEarthNet Spatial Source Mapping

## Source conclusion

The approved mapping status is **KEEP_UNMAPPED**.

The official BigEarthNet.txt source is pinned to revision `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316` with SHA-256 `d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554`. Its schema exposes `input`, `output`, `patch_id`, `s1_name`, split, and patch-center latitude/longitude. The publisher documentation describes box output strings and point tags, but does not specify their raster coordinate frame.

## Verified facts

- Selected spatial annotations: `25,303` boxes/point-box records.
- Parsed boxes: `25,303`.
- Parsed points: `13,124`.
- Values are finite and within `[0,1]`.
- Box extents are positive.
- Points are inside their paired boxes.
- Raw source strings are preserved.
- Point instructions may repeat the same point tag; observed repeated tags agree.
- `latitude` and `longitude` are patch-center metadata, not box coordinates.

## Unknown semantics

The source and loader do not establish:

- axis origin
- x/y versus row/column meaning
- whether y increases downward
- whether intervals are closed or half-open
- whether values refer to full S2 imagery, S1 imagery, a crop, or another annotation frame
- source raster width/height for these coordinates
- S1/S2 grid correspondence
- CRS or affine transform for the annotation geometry

Ravi's matcher preserves the original strings and does not convert them to raster pixels or CROMA tokens. Its parsing behavior is therefore useful provenance precedent, not evidence for a coordinate transform.

## Mapping decision

No transform from source unit-square geometry to the 120×120 analysis grid or 15×15 CROMA grid is approved. Applying `x * 120`, `y * 120`, or any token conversion would be an unsupported assumption. The source values remain `source_unit_square_unmapped` and Phase 2C/2D links retain `SPATIAL_LINK_UNAVAILABLE`.

The only approved geometric mapping remains the Phase 2B contract for geometry that is independently supplied in `PIXEL`, `ANALYSIS_GRID`, or `NORMALIZED_IMAGE` space. Such mapping is labeled `GEOMETRIC_CORRESPONDENCE`, never learned grounding.

## Validation accounting

- Proven source semantics: `0`
- Safely mapped source spatial annotations: `0`
- Deliberately unmapped spatial annotations: `25,303`
- Invalid source geometry: `0`
- Out-of-bounds source geometry: `0`
- Degenerate source geometry: `0`
- CROMA token links from BigEarthNet source geometry: `0`

This is a limitation report, not a failed ingestion. Numeric validation and provenance are complete; raster semantics are not established.
