# Phase 3AB — Geospatial Evidence and Map / Statistics Layer

SatQuery now reports bounded source measurements as **IMAGE STATISTICS**, not model confidence or semantic evidence.  The generic inspector records per-band minimum, maximum, mean, standard deviation, p02/p50/p98, nodata fraction, finite fraction, dtype, names, and nodata values only after a bounded full read.

When a valid CRS and non-empty bounds are supplied, the response contains an offline bounding-rectangle footprint, resolution, transform, dimensions, and an attempted WGS84 bounds transform.  Missing or invalid geospatial metadata returns `GEOSPATIAL_FOOTPRINT_NOT_AVAILABLE`; RGB files remain valid scene-description inputs.

Pairs expose `SPATIAL BOUNDS OVERLAP` only for valid same-CRS footprints.  Every pair explicitly remains `COREGISTRATION_NOT_VERIFIED`.  Pixel difference is deliberately unavailable (`PIXEL_DIFFERENCE_NOT_AVAILABLE`) because none of the current routes establishes alignment adequate for a quantitative change product.  It is never presented as a change map.

The compact UI evidence panels and JSON export show source metadata, footprint, statistics, declarations, adapters, warnings, route provenance, and the operational execution trace.  They do not expose server paths, chain-of-thought, grounding boxes, semantic maps, or invented confidence.

Cartosat-2S and RISAT inspection remains metadata-only and shows `MODEL INFERENCE: NOT VALIDATED`; this phase does not validate a model on either sensor.  The temporal route preserves PRE/POST order and supplies pair evidence without a change mask.

Development policy is unchanged: no TEST data, benchmark admission, model weight change, training, or sensor-data acquisition was performed.
