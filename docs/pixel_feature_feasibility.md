# Pixel/local feature feasibility (Pass 5B)

## Scope and evidence

This is a research-only feasibility study. It does not alter CROMA, the production model, the Pass 3 split, or the validated Pass 3 results. Evidence comes from the real BigEarthNet v2 sample `61_39`, the repository loaders, and an 18-area probe drawn from the pinned Pass 3 artifacts (10 train/4 validation/4 test areas). The diagnostic bundle is at `artifacts/pixel_feature_probe/61_39`.

## Raw-input audit

Sample `61_39` contains S2 in canonical order `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12` and S1 `VV,VH`. Native S2 shapes/resolutions are B02/B03/B04/B08 120×120 at 10 m, B05/B06/B07/B8A/B11/B12 60×60 at 20 m, and B01/B09 20×20 at 60 m. S1 VV/VH are 120×120 at 10 m. All are EPSG:32633 with bounds `[373200,5352000,374400,5353200]`; the common north-up affine is `[10,0,373200,0,-10,5353200,0,0,1]`. S2 declares nodata 0; S1 has no declared nodata. Raster masks report 14,400 valid common-grid pixels and zero invalid pixels per probed map.

The loader does not assume identical native grids. It validates CRS/bounds, uses B02 as the common grid, bilinearly reprojects spectral/SAR bands, and retains the categorical reference on its exact 10 m grid. S1 and reference must match the common grid exactly. Alignment therefore passes for this sample, while the 20/60 m bands necessarily contain resampled—not native 10 m—detail.

## Candidate inventory and scientific validity

| Feature | Formula/source | Units/range | Interpretation and caveats | Required valid pixels | Use |
|---|---|---|---|---|---|
| B04/B08 | raw red/NIR | source DN, nonnegative | spectral evidence; affected by atmosphere/illumination and resampling | band finite | pixel/token |
| NDVI | `(B08-B04)/(B08+B04)` | unitless, theoretical [-1,1] | vegetation-related contrast, not a vegetation/object detector | both bands; nonzero sum | token + explanation |
| NDWI | `(B03-B08)/(B03+B08)` | unitless [-1,1] | water-related evidence but vegetation/built surfaces can confound | both; nonzero sum | token + explanation |
| MNDWI | `(B03-B11)/(B03+B11)` | unitless [-1,1] | water-related contrast; B11 is resampled from 20 m | both; nonzero sum | token + explanation |
| NDBI | `(B11-B08)/(B11+B08)` | unitless [-1,1] | built-up-related evidence only; bare soil and dryness confound | both; nonzero sum | token + explanation |
| BSI | `(B11+B04-B08-B02)/(B11+B04+B08+B02)` | unitless [-1,1] | bare-soil-related contrast; not a soil classifier | four bands; nonzero denominator | token |
| VV/VH | raw channels | dB in this dataset | surface backscatter/scattering evidence; geometry, moisture and speckle confound | channel finite | pixel/token |
| VV−VH | `VV_dB-VH_dB` | dB | safe log-domain polarization contrast (linear-power ratio expressed in dB); not `VV/VH` division | both finite | token |
| local std | centered NaN-aware 3/5/7 windows | input units | local heterogeneity; scale dependent and edge/speckle sensitive | at least one finite window value | token |
| edge strength | gradient magnitude | input units/pixel | boundary evidence; not an object boundary detector | finite neighborhood | diagnostic/token candidate |

The code deliberately avoids dividing dB values. Near-zero index denominators become invalid rather than being forced to zero. Sample ranges were plausible: NDVI −0.6000…0.9312, NDWI −0.8848…0.6862, MNDWI −0.7652…0.8917, NDBI −0.8173…0.3152, BSI −0.5581…0.2447, VV −40.11…−2.04 dB, VH −39.91…−8.45 dB, VV−VH −15.69…22.64 dB.

## Local features and cost

NDVI and VV local standard deviation were measured at 3×3, 5×5 and 7×7. On `61_39`, each map took approximately 3.2, 4.3 and 6.7 ms respectively on CPU. Increasing windows raised the mean NDVI local standard deviation from 0.0483 to 0.0773 to 0.1024 and smoothed progressively broader boundaries; 5×5 is the balanced probe choice, not a final optimum. VV texture is noisier because no speckle-specific treatment was applied. Entropy/GLCM contrast/homogeneity were rejected from this first probe: quantization and directional choices require separate validation and would multiply redundant texture dimensions.

Raw map calculation took 0.18 ms, loading/alignment 85 ms, and aggregation of all 17 raw/local maps took 183 ms. The diagnostic bundle is 1.45 MB (uncompressed `.npy`, receipt, plot, experiment). Computation is CPU-suitable; no GPU is justified for these operations.

## Spatial sanity and hierarchy

The north-up RGB, B08, VV, valid mask, NDVI, NDWI, NDBI and BSI diagnostic preserves the same diagonal field/vegetation boundaries. Raster transform sign, array row/column convention, exact-grid tests, and visual inspection show no flip, transpose or apparent shift. Result: **PASS for sample 61_39**, not a universal survey of all 1,000 areas.

Recommended hierarchy: retain raw/derived maps transiently at pixel level; compute neighborhood evidence before aggregation; preserve per-token mean/std (and valid ratio where nodata exists) as the compact 225×F representation; aggregate robust token distributions only when a scene head needs them. Do not collapse spatial evidence to scene means before token reasoning.

## Temporal and explanation potential

The same deterministic maps can be computed independently on co-registered T1/T2 inputs. Differences should be made only after radiometric/acquisition comparability and cloud/validity checks; raw optical DN differences and SAR differences across incompatible geometries are unsafe. For language, the system may say a region has higher/lower vegetation-, water-, built-up-, polarization-, or heterogeneity-related evidence. It may not assert an object, land-cover class, causal change, or confidence without a validated trained head and temporal controls.

## Independent audit verdict

Formulas and bands pass; nodata is explicit; arrays are finite; S1/reference exact alignment and S2 reprojection pass; local windows are deterministic; token mapping passes; the probe uses disjoint areas and validation-only alpha selection. New information is spatial distribution/heterogeneity and BSI. Scene means of NDVI/NDWI/MNDWI/NDBI and raw-band moments are redundant with the legacy 62-vector. Predictive evidence is mixed and sample-limited. Overall classification: **B — recommended for specific tasks**, especially token grounding, explanations and future controlled change analysis; predictive integration requires a larger preregistered ablation.
