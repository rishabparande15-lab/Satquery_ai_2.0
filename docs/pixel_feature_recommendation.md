# Pixel feature recommendation

## Decision: B — recommended for specific tasks

Pixel/local features should proceed to a controlled token-level branch for spatial grounding and explanation, but current evidence does not support changing the production predictor.

Retain for the next ablation: token mean/std for NDVI, NDWI, NDBI, BSI and physically meaningful VV−VH dB, plus 5×5 NDVI local standard deviation and per-token valid ratio. MNDWI is scientifically valid but is likely redundant with NDWI/NDBI in a minimal first set; test it as a swap. Treat VV texture as an optional SAR-specific study after speckle handling.

Reject now: duplicate scene means already in the 62-vector; all raw-band min/max/percentile combinations; division of dB VV by dB VH; unvalidated entropy/GLCM feature families; claims of object detection or causal change.

The 18-area result is mixed (test −0.1910 pp, validation +0.1728 pp). Its strongest evidence is improved spatial localization capacity and explanation potential, not dependable scene-level performance. A subsequent decision should use a preregistered, area-disjoint ablation with repeated seeds, CROMA+token fusion, per-class/support reporting, calibration checks, and unchanged Pass 3 test isolation.

Safe future explanations describe spectral/backscatter evidence and heterogeneity. Land-cover assertions, objects, and change attribution require trained and validated heads. Temporal differencing is feasible only after strict co-registration, cloud/validity masking, acquisition compatibility and uncertainty reporting.
