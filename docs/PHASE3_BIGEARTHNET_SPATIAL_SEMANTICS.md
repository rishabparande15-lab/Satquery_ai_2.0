# BigEarthNet.txt spatial semantics audit

**Decision (2026-09-19): UNRESOLVED / FAIL-CLOSED.** No BigEarthNet.txt point or box is approved for conversion to SatQuery analysis pixels or CROMA tokens. This audit concerns deterministic source geometry only; it does not establish learned semantic grounding.

## Sources and authority

| Source | Location inspected | What it establishes | Authority |
|---|---|---|---|
| [BigEarthNet.txt paper, arXiv:2603.29630v2](https://arxiv.org/pdf/2603.29630) | Section 3 and Fig. 1, pp. 2, 5–6 | Text annotations use BigEarthNet v2 S1/S2 pairs and derive spatial instances from pixel-level LULC reference maps; referring point instructions use a point within an instance and predict its enclosing box. Shows example numeric tuples. | Publisher, direct for generation intent; silent on coordinate convention. |
| [Publisher project page](https://txt.bigearth.net/) | Abstract and Overview | S1/S2 pairs are described as co-registered; boxes are used for referring expression detection. | Publisher, direct; no coordinate definition. |
| [Publisher dataset card](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt) | Parquet File Structure and How to use | `patch_id` identifies S2, `s1_name` identifies S1, `input` and `output` are text. Latitude and longitude refer to the patch center, not annotation geometry. The card's 120-pixel loader example is an image-loading choice. | Publisher, direct; no box/point raster convention. |
| [Publisher loader](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/blob/main/ben_txt_datamodule.py) | `BENImageReader.stack_and_interpolate`, `BENTxTDataset.__getitem__`, and default transforms | Loads/resizes S1/S2 bands; passes the source input and output as strings, only replacing `<point>`/`<ref>` tags. It does not parse or transform box/point coordinates. Default training transforms can flip images without transforming the textual geometry. | Publisher code, direct for loader behavior; not annotation-generation code. |
| [BigEarthNet v2 description](https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf) | pp. 1, 4–8 | S2 patches have pixel-level reference maps; S1 patches are paired one per S2 patch; files are georeferenced GeoTIFFs. The reference-map directory follows S2 patch identity. | Publisher, direct for source imagery; no BigEarthNet.txt coordinate convention. |
| [BigEarthNet v2 pipeline](https://github.com/rsim-tu-berlin/bigearthnet-pipeline) | README, Overview and Finalizing | Generates 1,200 m patches and reference maps, then aligns/adds S1 data and metadata. | Publisher-linked implementation, direct for v2 construction; does not specify text annotation coordinate encoding. |
| [SatQuery source ingestion](../src/annotation_foundation.py) | `parse_geometry` and `COORDINATE_CONVENTION_STATUS` | Parses finite source strings and validates observed unit-square bounds; labels geometry `source_unit_square_unmapped`. | Local implementation, direct for current behavior, not external semantics. |
| [SatQuery spatial contract](../src/spatial_contract.py) | `CoordinateSpace` and transforms | SatQuery `NORMALIZED_IMAGE` means continuous upper-left image-boundary coordinates, with half-open boxes. | Local contract only; cannot be assigned to publisher values by numeric similarity. |
| [SatQuery image-language foundation](../src/image_language_foundation.py) | sample assembly spatial reference | Preserves source geometry while leaving analysis geometry and token coordinates null. | Local implementation. |
| [Ravi's `Sat_Query`](https://github.com/raviasha/Sat_Query) | Local clone at `ac5541086c637ece2e9fd3c0a79cfa8bea23304d`, `src/satquery/match_annotations.py` | Matches `patch_id`/`s1_name`, retains publisher fields and exact strings, and does no coordinate conversion. `preprocessing.py`/`targets.py` define a separate 120×120 to 15×15 scientific grid. | Independent downstream implementation; no publisher semantics. Live remote HEAD could not be checked because the GitHub connection failed. |

The inspected public publisher materials expose no annotation-generation implementation that defines normalization or boundary rules. Absence from these inspected sources is an evidence gap, not proof that such code does not exist elsewhere.

## Findings by semantic question

| Question | Finding | Source | Confidence |
|---|---|---|---|
| Point meaning | Referring point detection supplies a point **within the target instance**; the paper says qualifying instances have a centroid within the instance. Whether the emitted point is exactly that centroid, a rounded centroid, a sampled interior location, or a pixel center is not specified. | Paper §3.1, p. 6 | High for interior/centroid selection criterion; low for exact encoding. |
| Box meaning | Output is an enclosing bounding box for a reference-map LULC instance. The four-number order is visually suggestive of two corners, but exact axis naming and boundary convention are not formally defined. | Paper Fig. 1 and §3.1; dataset preview | High for enclosing-instance intent; unresolved for encoding. |
| Origin and direction | Top-left, bottom-left, center, and y direction are not specified. | Paper, card, loader inspected above | Unresolved. |
| Axis identity | Neither `x=column/easting` nor `y=row/northing` is established for the text tuples. | Same | Unresolved. |
| `[0,1]` normalization | Publisher examples contain values 0 and 1. The denominator, scaling formula, rounding precision, and whether 1 denotes an outer edge or final pixel center are not specified. Numeric range alone is insufficient evidence. | Paper Fig. 1; dataset preview; local source audit | High for observed range; unresolved for formula. |
| Extent and dimensions | Text uses BigEarthNet v2 patch identities and reference-map instances. The publisher loader can resize bands to 120×120, but does not say that annotations were normalized against a particular raster band, reference map, crop, or resized tensor. | Paper §3; dataset card and loader | High for patch association; unresolved for coordinate reference. |
| Point coordinate type | Pixel index, pixel center, continuous image coordinate, and representative location cannot be distinguished. | Paper §3.1; no conversion in loader | Unresolved. |
| Box boundaries | Inclusive, exclusive, half-open, edge-coordinate, pixel-center, and rounding behavior are not stated. | Paper and card | Unresolved. |
| S1/S2 relation | The source S1/S2 patches are called co-registered and paired. This does not establish identical pixel grids, affine transforms, or which one the text geometry indexes. | Paper §3; BigEarthNet v2 description; v2 pipeline | High for pairing; unresolved for exact spatial identity. |
| SatQuery mapping | No authoritative transform to the 120×120 analysis grid, 15×15 CROMA grid, or 225 row-major tokens can be derived. | Sources above; local spatial contract | High decision confidence. |
| Existing conversion | Publisher loader and Ravi matcher retain text coordinates without pixel/token conversion. SatQuery deliberately leaves them unmapped. | Publisher `ben_txt_datamodule.py`; Ravi `match_annotations.py`; local code | High for inspected implementations; not a global claim about all code. |

There is no documented disagreement between publisher sources; their coordinate detail is incomplete. SatQuery's upper-left, half-open `NORMALIZED_IMAGE` convention is a separate internal definition, not a publisher statement. The publisher loader's image flips without geometry updates make its training image/annotation relation unsuitable as proof of a static pixel mapping.

## Required evidence to lift the block

Obtain a publisher definition or annotation-generation source revision showing: (1) tuple order and axis directions; (2) origin; (3) precise normalization denominator and rounding; (4) pixel-center versus boundary interpretation; (5) box endpoint and point-selection semantics; (6) the exact reference-map or image grid and any crop/resampling used; and (7) how that grid maps to S1, S2, and SatQuery's validated analysis raster. Verify that definition on source records and raster metadata before adding a named publisher coordinate space and any deterministic conversion.

## Current handling and impact

SatQuery keeps the exact source strings in `raw_source` and parsed numeric geometry in `box`/`point`, marks its frame `source_unit_square_unmapped` and convention `UNKNOWN`, and emits `analysis_geometry = null` and `token_coordinates = null`. The Phase 2D source audit recorded 25,303 valid spatial annotations and zero safely mapped annotations or CROMA token links. There is no automatic spatial correspondence and no learned grounding. VLM/grounding work must not treat these source tuples as localized supervision on analysis pixels or tokens until the missing evidence is supplied.

No production transform or scientific predictor was changed for this audit.

## Verification

The focused spatial, annotation, representation-linking, and image-language tests passed (`45 passed`). The full Python suite passed (`355 passed, 5 skipped`). `git diff --check` passed. The sealed-test regression artifact `artifacts/pipeline3_5000/representations/scientific_regression_phase2e/historical_checkpoint_inference.json` still records `mae_pp = 4.1034286464812215`, `rmse_pp = 9.5873311029768`, and `dominant_class_accuracy = 0.65` for 200 test areas. No retraining was run; these are artifact values, not fresh inference measurements.
