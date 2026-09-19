# Phase 2F - BigEarthNet.txt Spatial Semantics

## 1. Executive summary

**Decision: NO / NOT YET. Keep the source geometry unmapped and fail closed.**

The authoritative BigEarthNet.txt paper and publisher materials establish that
the referring-expression boxes and points are generated from BigEarthNet v2
reference-map LULC instances. They do not establish the coordinate frame of the
numbers embedded in the text. In particular, they do not define axis order,
origin, y direction, normalization denominator, pixel-center versus boundary
semantics, box endpoint semantics, or the exact raster used for normalization.

Therefore no defensible transform from BigEarthNet.txt values to SatQuery's
120 x 120 analysis grid exists at this time. The existing 15 x 15 CROMA
mapping remains valid only for geometry already expressed in the SatQuery
`PIXEL`, `ANALYSIS_GRID`, or explicitly defined `NORMALIZED_IMAGE` spaces. No
BigEarthNet.txt source geometry is converted to pixels or tokens by this phase.

This is a provenance decision, not a failed ingestion. Raw source records and
parsed numeric values remain preserved, and unknown semantics remain explicitly
marked as `source_unit_square_unmapped`.

## 2. Sources inspected

| Source | Exact location inspected | Result | Status |
| --- | --- | --- | --- |
| Official BigEarthNet.txt dataset card | [publisher card](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt), `Parquet File Structure`, `How to use` | Defines fields, S1/S2 identifiers, center latitude/longitude, and a loader example with `img_size=120`; does not define text-coordinate semantics. | VERIFIED for exposed schema; UNRESOLVED for geometry |
| Official BigEarthNet.txt loader | [ben_txt_datamodule.py](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/blob/72d865f2146f0a85b720f7f3ca1cdbaeafc3d316/ben_txt_datamodule.py), `BENImageReader.stack_and_interpolate`, `BENTxTDataset.__getitem__`, `default_train_transform` | Resizes image arrays, returns text strings, and only replaces `<point>`/`<ref>` tags. It does not parse or transform coordinates. Train transforms can flip images. | VERIFIED |
| BigEarthNet.txt paper | [arXiv HTML](https://arxiv.org/html/2603.29630), Section 3.1, Figure 1 | Referring boxes target LULC instances; point annotations target instances whose centroid lies inside the instance. Examples show numeric point and box values. | VERIFIED for generation intent; UNRESOLVED for encoding |
| BigEarthNet v2 pipeline | [publisher pipeline](https://github.com/rsim-tu-berlin/bigearthnet-pipeline), `Overview` and `Finalizing the dataset` | Creates 1200 m S2 patches and pixel-level reference maps, then aligns/adds S1 data and metadata. | VERIFIED for v2 construction; insufficient for text encoding |
| Pinned local source receipt | [annotation source audit](ANNOTATION_SOURCE_AUDIT.md), verified revision and checksum | Official `BigEarthNet.txt.parquet` at revision `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316`, SHA-256 `d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554`. | VERIFIED |
| SatQuery source parser | [annotation_foundation.py](../src/annotation_foundation.py), `parse_geometry` | Performs numeric unit-square validation only and labels the frame unmapped. | VERIFIED local behavior |
| SatQuery spatial contract | [spatial_contract.py](../src/spatial_contract.py), transforms and validators | Defines the internal 120 x 120 boundary-coordinate and 15 x 15 row-major token contracts. | VERIFIED local contract; not publisher evidence |
| SatQuery linkage | [representation_linking.py](../src/representation_linking.py), `link_annotation` and `_spatial_link` | Preserves source geometry and rejects source geometry without a validated canonical coordinate space. | VERIFIED local behavior |

The source inspection did not find an authoritative BigEarthNet.txt
annotation-generation implementation that states the missing coordinate rules.
This absence is recorded as an evidence gap, not as proof that no such source
exists elsewhere.

## 3. BigEarthNet.txt coordinate evidence

The publisher paper and examples support these limited claims:

- The four-value output is an enclosing box for a reference-map LULC instance.
- A point instruction supplies a point within the target instance; the paper
  describes selecting instances whose centroid lies within the instance.
- Observed source examples use values in the unit-square numeric range.
- The text rows identify a paired Sentinel-2 patch (`patch_id`) and Sentinel-1
  patch (`s1_name`).

They do **not** state that the first value is x/column, that the second value
is y/row, that the origin is upper-left, or that the values are normalized by
width and height. They also do not state whether `1` is an outer boundary or a
last pixel center, or whether the box is inclusive, exclusive, or continuous.

The local parser intentionally preserves the exact `input` and `output` in
`raw_source`. It parses boxes as `[first first, second second]` and points from
`<point>(first, second)</point>` only for numeric validation. That parser behavior
must not be read as publisher semantics.

## 4. Coordinate-system findings

| Question | Finding | Evidence status |
| --- | --- | --- |
| Coordinate domain | Selected values are finite and observed within `[0,1]`. | VERIFIED observation; normalization meaning UNRESOLVED |
| X versus Y | Not defined by inspected primary sources. | UNRESOLVED |
| X as column | Not established. | UNRESOLVED |
| Y as row | Not established. | UNRESOLVED |
| Origin | Not established as top-left, bottom-left, or another origin. | UNRESOLVED |
| Direction | Not established, including y direction. | UNRESOLVED |
| Reference frame | The generation source is a BigEarthNet v2 reference-map instance, but the text encoding frame is not named. | STRONGLY_SUPPORTED for source relation; UNRESOLVED for encoded frame |

The examples are compatible with several different conventions. Numeric range
alone cannot select among them.

## 5. Pixel and boundary findings

| Question | Finding | Status |
| --- | --- | --- |
| Pixel centers or boundaries | Not stated. | UNRESOLVED |
| Meaning of exactly 0 | Not stated. | UNRESOLVED |
| Meaning of exactly 1 | Not stated; it may be an edge in one convention and a valid center/index in another. | UNRESOLVED |
| Box ordering | The examples are visually consistent with two corner pairs, but axis names and formal ordering are not specified. | STRONGLY_SUPPORTED shape; UNRESOLVED semantics |
| Box inclusivity | Not stated. | UNRESOLVED |
| Point semantics | Interior/centroid selection intent is documented; exact emitted point representation and rounding are not. | STRONGLY_SUPPORTED intent; UNRESOLVED encoding |

SatQuery's internal contract is deliberately more specific: continuous
boundary coordinates, half-open boxes, and an upper-left origin. Those rules
cannot be assigned to BigEarthNet.txt values by convention or numeric shape.

## 6. Image and reference-map relationship

BigEarthNet.txt uses paired, co-registered S1 and S2 images from BigEarthNet
v2. The v2 pipeline documents 1200 m patches, pixel-level reference maps, and
later S1/S2 alignment. This establishes the semantic source of the instances,
not the exact grid on which the text values were encoded.

The publisher loader's `img_size=120` is an image-loading target. It resizes
bands to that target and does not transform textual coordinates. Its optional
training horizontal/vertical flips likewise operate on the image tensor while
the text remains a string. Consequently:

- `img_size=120` is not evidence that text coordinates were normalized by 120;
- the loader does not establish whether the source was S1, S2, or the reference
  map;
- it does not establish a shared pixel grid, affine transform, or crop for the
  text annotations;
- S1/S2 co-registration does not by itself prove identical array dimensions
  or boundary conventions.

SatQuery's 120 x 120 grid is its validated B02 analysis grid. It is not a
publisher-defined BigEarthNet.txt annotation frame.

## 7. 120 x 120 mapping decision

No exact forward or inverse transform is approved. In particular, neither
`coordinate * 119` nor `coordinate * 120` is supported by the inspected
evidence. Rounding, clipping, and boundary behavior are therefore undefined.

The required missing evidence is a primary source revision or reproducible
annotation-generation code that specifies all of the following:

1. tuple order and axis directions;
2. origin;
3. normalization denominator and precision/rounding;
4. pixel-center versus boundary interpretation;
5. box endpoint and point-selection semantics;
6. source raster, dimensions, crop, and resampling; and
7. the mapping of that raster to the paired S1/S2 data and SatQuery analysis
   grid.

Until then, out-of-bounds or malformed source values remain explicit source
validation results, and no source value is silently clipped, rewritten, or
converted.

## 8. CROMA-token mapping decision

The downstream SatQuery mapping is independently established:

- analysis grid: 120 x 120;
- token grid: 15 x 15, row-major;
- each token is an 8 x 8 half-open analysis block;
- token index is `row * 15 + col`;
- token 0 covers `[0,0,8,8)` and token 224 covers `[112,112,120,120)`.

For already canonical geometry, the existing implementation preserves the
relationships `FULLY_CONTAINED`, `INTERSECTS`, and `CONTAINS`, and maps a point
by its analysis-grid block. This is deterministic geometric correspondence,
not learned grounding.

Because the BigEarthNet.txt to analysis-grid transform is unresolved, no
BigEarthNet.txt box or point is eligible for this downstream mapping. No token
links are generated, and no new token convention is introduced.

## 9. Source geometry preservation and fail-closed behavior

The current implementation satisfies the Phase 2F preservation requirements:

- original source strings remain in `raw_source`;
- parsed `box`/`bbox` and `point` values retain source values;
- the frame is `source_unit_square_unmapped`;
- the source convention is `UNKNOWN`;
- missing canonical geometry produces `SPATIAL_LINK_UNAVAILABLE`;
- invalid canonical geometry is not clipped;
- derived analysis/token geometry, when supplied independently, is separate
  from source geometry and marked as deterministic rather than learned.

No production mapping behavior is changed by this phase.

## 10. Real-data validation

The existing validated artifacts were inspected rather than regenerating the
5,000-area representation set.

| Measure | Result |
| --- | ---: |
| Source rows in the pinned full Parquet | 9,553,962 |
| Selected source records in the frozen inventory | 101,571 |
| Source spatial records parsed | 25,303 |
| Parsed boxes | 25,303 |
| Parsed point annotations | 13,124 |
| Validated image-language spatial samples | 25,300 |
| Mappings attempted | 0 |
| Mappings successfully verified | 0 |
| Deliberately unmapped validated samples | 25,300 |
| Unresolved mappings | 25,300 |
| Out-of-bounds source geometry | 0 |
| Malformed source geometry | 0 |
| Degenerate source geometry | 0 |
| CROMA token mappings from source geometry | 0 |

The difference between 25,303 parsed source spatial records and 25,300
validated exported samples is the existing quarantined/held-out filtering in
the image-language artifact. It is not a mapping failure and does not alter
the preserved source records. The artifact reports 12,178 validated text-box
samples and 13,122 validated point samples.

## 11. Test and regression results

Phase 2F requires no code or test change because the correct result is to keep
the existing behavior. The focused existing tests cover the relevant boundary:

- `tests/test_spatial_contract.py`
- `tests/test_representation_linking.py`
- `tests/test_annotation_foundation.py`
- `tests/test_image_language_foundation.py`

The prior repository audit records these checks as passing: focused spatial,
annotation, representation-linking, and image-language tests (`45 passed`),
full Python suite (`355 passed, 5 skipped`), and `git diff --check`. The current
Phase 2F change is documentation-only; no retraining, VLM inference, artifact
regeneration, or representation rebuild was performed.

## 12. Scientific impact

This phase does not change the Pipeline 3 predictor, CROMA checkpoint, hybrid
checkpoint, 5,000-area split, representation receipts, or scientific metrics.
The frozen baseline remains:

- Accuracy: `65.0%`
- MAE: `4.1034 pp`
- RMSE: `9.5873 pp`

No learned grounding claim is made. BigEarthNet.txt referring annotations are
available as provenance-preserved source material, but are not valid pixel or
token supervision for SatQuery until the missing coordinate evidence is found.

## 13. Explicit remaining blockers

The mapping remains blocked until authoritative evidence resolves:

- x/y axis order and row/column meaning;
- origin and y direction;
- exact source raster and dimensions;
- normalization denominator and whether the domain is `[0,1]` or `[0,1)`;
- center, boundary, and rounding semantics;
- inclusive versus half-open box boundaries;
- point generation and representation precision;
- exact S1/S2/reference-map spatial correspondence; and
- transform compatibility with SatQuery's validated 120 x 120 B02 grid.

**Final answer to the phase question:** BigEarthNet.txt spatial annotations
cannot currently be mapped into SatQuery's 120 x 120 analysis grid or 15 x 15
CROMA grid without undocumented coordinate assumptions. The approved result is
**NO / NOT YET - UNRESOLVED / FAIL-CLOSED**.
