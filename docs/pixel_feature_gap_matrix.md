# Pixel feature gap matrix

The current 62 dimensions are 4 moments for each of 12 S2 and 2 S1 channels (56), plus scene means of NDVI, NDWI, MNDWI, NDBI, legacy clipped `VV−VH`, and legacy clipped `VH/VV` (6). The two SAR relationship entries are retained only for compatibility and are not physically interpretable dB ratios.

| Candidate | Already exists? | Resolution | New information? | Recommended level |
|---|---|---|---|---|
| B04/B08 values | Scene moments only | native 10 m | spatial layout | pixel/token |
| NDVI | scene mean only | 10 m operands | distribution and location | token |
| NDWI | scene mean only | 10 m operands | distribution and location | token |
| MNDWI | scene mean only | B11 resampled 20→10 m | distribution and location | token |
| NDBI | scene mean only | B11 resampled 20→10 m | distribution and location | token |
| BSI | no | mixed 10/20 m→10 m | complementary bare-soil-related contrast | token |
| VV/VH raw maps | scene moments | 10 m | spatial backscatter layout | pixel/token |
| VV−VH dB map | only a clipped legacy scene value | 10 m | physically meaningful local polarization contrast | token |
| NDVI local std 5×5 | no | 50 m support on 10 m grid | vegetation-pattern heterogeneity | token |
| VV local std | no | 30/50/70 m support | SAR heterogeneity, but speckle-sensitive | research only |
| edge strength | no | immediate neighbors | boundary/location evidence; likely correlated with local std | research only |
| valid-pixel ratio | counts exist at scene level | each 8×8 token | local data quality | token metadata |

Retain a minimal candidate set: token mean/std for NDVI, NDWI, NDBI, BSI, VV−VH and 5×5 NDVI local std. Reject duplicate scene means, raw min/max proliferation, dB division, and unvalidated entropy/GLCM descriptors.
