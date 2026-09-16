# Spatial Annotation Protocol for BigEarthNet.txt

## Status summary

- Coordinate convention: UNKNOWN
- Raster-grid mapping: UNKNOWN
- S1/S2 correspondence: UNKNOWN
- CROMA 15x15 mapping: NOT IMPLEMENTED
- Metadata dependency policy: DOCUMENTED, conservative, and split-safe
- Spatial training export: NOT IMPLEMENTED
- Non-spatial image-language export: IMPLEMENTED in Phase 6B.4

## Authoritative evidence and status

### Verified facts

- The source is the author-maintained BigEarthNet.txt Parquet pinned at revision `72d865f2146f0a85b720f7f3ca1cdbaeafc3d316`.
- The SHA-256 of the pinned file is `d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554`.
- The selected 5,000-area inventory has 101,571 matched source rows, 4,771 matched images, and 101,542 validated records.
- All box and point values in the selected rows parse as finite numbers in the source unit-square strings.
- The raw source strings remain untouched and are preserved in the canonical annotation records.

### Unknowns that remain explicitly unresolved

The publisher does not establish the following in authoritative form:

- axis origin
- x/y vs row/column semantics
- raster-grid relation
- S1/S2 grid correspondence
- boundary inclusivity/exclusivity
- whether coordinates are image-relative or geographic
- whether the numbers refer to 120x120 optical pixels, Sentinel-1 pixels, or another grid

Because these facts are not proven, this protocol intentionally keeps the coordinate convention at UNKNOWN and does not apply any CROMA token transformation.

## Geometry validation policy

The implementation validates the source strings only as numeric unit-square observations. The validation rules are:

- box strings are parsed as four finite numbers in `[x0 y0, x1 y1]` form
- point strings are parsed from `<point>(x, y)</point>`
- values must be finite and within [0, 1]
- box extents must be positive
- each point must be inside its paired box
- a malformed or non-finite source geometry is quarantined

No conversion is performed from those source values to image pixels, analysis-grid pixels, or CROMA tokens.

## Metadata dependency taxonomy

The repository uses a conservative taxonomy:

- VISUAL_ONLY
- SPATIAL_VISUAL
- METADATA_ONLY
- VISUAL_PLUS_METADATA
- UNKNOWN_DEPENDENCY

Rules:

- captions are treated as VISUAL_ONLY unless they explicitly mention metadata fields
- text/point box annotation tasks are classified as SPATIAL_VISUAL
- country/season/climate-zone MCQ questions are treated as METADATA_ONLY when they depend on metadata fields
- mixed cases remain VISUAL_PLUS_METADATA
- unsupported or ambiguous cases are UNKNOWN_DEPENDENCY

## Training eligibility policy

The current policy is intentionally conservative:

- VISUAL_ONLY -> ELIGIBLE_VISUAL_VQA
- SPATIAL_VISUAL -> REQUIRES_SPATIAL_VERIFICATION
- METADATA_ONLY -> EXCLUDED_FROM_VISUAL_VQA
- VISUAL_PLUS_METADATA -> REQUIRES_METADATA_AWARE_EXPERIMENT
- UNKNOWN_DEPENDENCY -> QUARANTINE

This prevents metadata shortcuts from being treated as visual supervision.

## Split protocol

- The frozen image-level split is preserved.
- Annotation records inherit the image split.
- Benchmark holdout rows are retained with explicit quarantining status.
- No annotation-level random split is created.
- Identical question-answer strings across different images are reported as CONTENT_OVERLAP, not image leakage.

## CROMA mapping status

CROMA token mapping is not implemented in this phase because the source-to-raster convention is still UNKNOWN.

Status: NOT IMPLEMENTED

## Provenance requirements

Any future spatial transformation must preserve the original source geometry and record:

- source geometry
- source coordinate convention
- source coordinate domain
- target raster/grid
- transformed geometry
- transformation method/version
- provenance hash
- mapping status
- validation status

## Stop conditions

This protocol stops and reports UNKNOWN rather than guessing if any of the following remain unresolved:

- coordinate origin
- axis semantics
- raster relationship
- S1/S2 correspondence
- boundary convention
- metadata dependency
- benchmark isolation

## Implementation status for this repository

Current repository status:

- source IDs, splits, and validity checks: VERIFIED
- geometry parsing and source unit-square validation: VERIFIED
- metadata taxonomy rules: IMPLEMENTED
- spatial mapping to analysis grid: UNKNOWN
- CROMA token mapping: NOT IMPLEMENTED
- visual VQA training: NOT STARTED

## This phase boundary

This phase explicitly does not begin VQA training, grounding training, image-text alignment, or any CROMA token mapping. It only establishes the protocol and the conservative audit state.
