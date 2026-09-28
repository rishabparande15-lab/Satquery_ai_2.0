# Phase 3K.1 — Grounding Dataset and Spatial-Semantics Audit

## Final status

**GROUNDING_TRAINING_BLOCKED**

This is a dataset, spatial-semantics, and provenance audit only. It creates no
grounding labels, model, pixel/token mapping, or external-data integration.

## 1. Objective

Determine whether an authoritative, reproducible remote-sensing source can
support text-to-region grounding in SatQuery. Grounding is kept separate from
classification, VQA, and captioning: it requires a text expression, image
identity, authoritative region geometry, and verified spatial semantics.

## 2. Existing BigEarthNet spatial annotations

BigEarthNet.txt supplies provenance-preserved text-box and point records. The
pinned source audit records 25,303 parsed spatial annotations (12,178 boxes
and 13,122 points in the validated image-language artifact), tied to
BigEarthNet image identity and split. It supplies neither a usable pixel nor
token target at present.

`SPATIAL_ANNOTATION_AVAILABLE` remains true, but
`PIXEL/TOKEN_MAPPING_UNVERIFIED` remains true as well.

## 3. Phase 2F constraints

Phase 2F is unchanged and fail-closed. The authoritative materials establish
that the source values refer to reference-map land-use/land-cover instances,
but do not establish their encoded coordinate space, axis order, origin, y
direction, normalization denominator, pixel-center/boundary rule, box endpoint
rule, source raster, or transform to SatQuery's B02 120×120 grid. Therefore:

- source strings and parsed numeric tuples stay `SOURCE_GEOMETRY_PRESERVED`;
- their frame stays `source_unit_square_unmapped` with convention `UNKNOWN`;
- no BigEarthNet box/point is mapped to a pixel, CROMA token, mask, or
  pseudo-label; and
- no coordinate assumption is inferred from unit-range values or loader
  `img_size=120`.

## 4. External grounding dataset landscape

The audit found releases that explicitly call their task visual grounding or
referring-expression comprehension/segmentation. It excludes generic RS VQA,
captioning, classification, and dialogue sources unless their released record
also carries geometry tied to the referenced image.

Primary sources inspected: [VRSBench repository](https://github.com/lx709/VRSBench),
[VRSBench paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/05b7f821234f66b78f99e7803fffa78a-Paper-Datasets_and_Benchmarks_Track.pdf),
[OPT-RSVG repository](https://github.com/like413/OPT-RSVG),
[GeoChat repository](https://github.com/mbzuai-oryx/GeoChat),
[GeoGround repository](https://github.com/VisionXLab/GeoGround), and the
[refGeo dataset card](https://huggingface.co/datasets/erenzhou/refGeo).

## 5. Candidate dataset inventory

| Candidate | Officially documented grounding material | Release, imagery, and geometry status | Training status |
|---|---|---|---|
| VRSBench | 29,614 images, 52,472 object referring expressions, detailed captions, and VQA; the paper evaluates visual grounding. Source objects derive from detection datasets including DOTA-v2 and DIOR. | Hugging Face dataset route and streaming are documented. Text is CC-BY-4.0, but the project states some DOTA imagery/annotations are academic-use-only. The public README says evaluation JSON boxes are normalized to 0–100, while origin, axis naming, endpoint/bounds semantics, per-record source imagery terms, immutable data revision, and comprehensive leakage evidence were not established here. | **BLOCKED.** |
| OPT-RSVG | Referring-expression visual grounding; 25,452 images and 48,952 image-query pairs. Published train/validation/test counts are 19,580 / 4,895 / 24,477. | Official code points data to Google Drive/Baidu Netdisk. The inspected README did not establish data license, immutable archive hash, annotation schema/coordinate convention, image-to-annotation manifest, or duplicate/geographic audit. | **BLOCKED.** |
| DIOR-RSVG | Used by OPT-RSVG as an RS visual-grounding dataset. | The inspected official OPT-RSVG materials link external delivery but do not establish an independently pin-able DIOR-RSVG record manifest, license chain, geometry convention, or split/leakage audit. | **UNKNOWN / BLOCKED.** |
| GeoChat_Instruct | The project documents 318k RS instruction pairs, including region captions, referring expressions, and region annotations; it demonstrates rotated-box referring-object detection. | Code, model, dataset, and evaluation scripts are announced, but the inspected release evidence did not establish a self-contained immutable imagery/annotation manifest, separate data licenses, coordinate convention, or leakage audit. Its generated instruction construction is not by itself authoritative geometry documentation. | **BLOCKED.** |
| refGeo / GeoGround | 161k image-text pairs and 80k images, with HBB, OBB, and mask tasks; refGeo combines four prior grounding sources plus AVVG. | Dataset card exposes `question_id`, `image_id`, `bbox`, `poly`, `question`, and dataset fields, supporting explicit linkage. It declares `s-lab-1.0`; the card preview currently reports generation/type inconsistency. The inspected evidence does not establish all upstream image terms, immutable data revision, per-file hashes/dimensions, HBB coordinate convention/bounds, mask provenance (the project says masks are SAM-generated), or leakage audit. | **BLOCKED.** |

No candidate is called a preferred or "best" source. This inventory is not an approval to download, merge, train, or evaluate any external dataset.

## 6. Geometry formats

| Source | Documented geometry | What is not safely assumed |
|---|---|---|
| BigEarthNet.txt | Numeric text boxes and points. | Whether tuples are pixel, normalized-image, geographic, or a particular box/point convention. |
| VRSBench | Object boxes; project material uses OBB terminology and states released evaluation boxes are normalized to 0–100. | Whether a record uses an HBB/OBB serialization, axis/origin, endpoint convention, or a safe conversion to SatQuery pixels. |
| OPT-RSVG / DIOR-RSVG | Bbox-producing referring-expression task. | Exact released bbox field order, units, origin, inclusivity, and image dimensions. |
| GeoChat_Instruct | Region annotations and rotated bounding-box outputs are documented. | Training-record geometry serialization and image-coordinate semantics. |
| refGeo | HBB, OBB/polygon, and mask are documented; card rows expose `bbox` and `poly`. | Bbox axis labels/bounds, pixel dimensions for each image, CRS, and whether the mask is source annotation rather than SAM-derived auxiliary data. |

No audited candidate establishes a point-grounding dataset usable by this phase.

## 7. Coordinate semantics

SatQuery requires explicit coordinate space, origin, axis order, bounds,
dimensions, and (for geographic geometry) CRS before deterministic mapping.

| Candidate | Coordinate-space finding | Mapping outcome |
|---|---|---|
| BigEarthNet.txt | Values observed in unit range, but frame and conventions are unresolved. | `MAPPING_UNVERIFIED`. |
| VRSBench | Released evaluation boxes are documented as normalized 0–100. Origin, x/y versus row/column, endpoint/bounds, source dimensions, and transform are not established by inspected material. | `MAPPING_UNVERIFIED`. |
| OPT-RSVG / DIOR-RSVG | Coordinate representation and image dimensions not established from inspected official release material. | `MAPPING_UNVERIFIED`. |
| GeoChat_Instruct | Region / rotated-box capability is documented, but released record convention is not established here. | `MAPPING_UNVERIFIED`. |
| refGeo | Card examples expose integer bbox/polygon values and `image_id`, but examples do not authoritatively define coordinate semantics or image dimensions. | `MAPPING_UNVERIFIED`. |

None of the inspected sources was verified as geographic geometry with an
available CRS; no transform is performed.

## 8. Image–annotation linkage

Grounding validity requires `image_id + annotation_id + geometry`, or a source
equivalent. BigEarthNet has source annotation identity and image linkage but
not usable geometry semantics. refGeo visibly has `question_id + image_id +
bbox/poly`, but its full data release and image assets were not materialized or
hash-verified. VRSBench, OPT-RSVG, DIOR-RSVG, and GeoChat advertise
image/expression/region association, but the inspected documentation alone is
not a verified per-record local manifest. Filename similarity was not accepted
as linkage proof.

## 9. Grounding task types

- Text → BBOX / referring expression: explicitly documented for VRSBench,
  OPT-RSVG, GeoChat, and refGeo.
- Text → oriented BBOX/polygon: documented for GeoChat and refGeo; VRSBench
  uses OBB terminology.
- Text → mask / referring segmentation: documented for refGeo; GeoGround says
  its AVVG masks are automatically generated by SAM, so they must retain that
  provenance if later audited.
- Text → point: no usable audited source verified.
- Text → region caption alignment: documented for GeoChat; it is not a
  substitute for verified model grounding.

Source geometry, deterministic coordinate conversion, and learned grounding
prediction are distinct objects. A source box is not a model prediction.

## 10. License

No candidate clears SatQuery's separate image-and-annotation license gate:

| Candidate | Image / annotation / code license assessment |
|---|---|
| BigEarthNet.txt | Existing source provenance is verified, but geometry semantics block use. |
| VRSBench | Annotation CC-BY-4.0 is documented; mixed upstream imagery includes DOTA material restricted to academic use. Full image/annotation source-chain terms are **PARTIAL**. |
| OPT-RSVG / DIOR-RSVG | Data-image and annotation terms were not established from inspected release material: **UNKNOWN**. |
| GeoChat_Instruct | Separate data/image/annotation terms were not established: **UNKNOWN**. |
| refGeo | Dataset-card license `s-lab-1.0` is declared, but all upstream imagery/annotation terms and generated-mask provenance are not established: **PARTIAL**. |

## 11. Source pinning

The audit did not identify a complete immutable data pin plus per-file manifest
for an externally usable grounding source. A code commit alone is insufficient
when imagery is elsewhere. VRSBench and refGeo have hosted dataset routes, but
an immutable revision and complete file/annotation hashes were not recorded in
this phase. OPT-RSVG uses external file hosts. Therefore external grounding
source pinning is **PARTIAL / UNKNOWN**, not verified.

## 12. Split and leakage

OPT-RSVG publishes train/validation/test counts, and VRSBench/refGeo expose
dataset split interfaces, but an official split alone is not a leakage audit.
No inspected candidate supplied a completed audit of image, region, scene,
annotation, geographic, and near-duplicate overlap suitable for SatQuery. No
external split is created, merged, or compared with the BigEarthNet 5,000-area
scientific split.

## 13. Bounded acquisition

No 3–5 example acquisition was authorized. It requires prior grounding
validity, complete licensing, and source pinning. VRSBench documents streaming
and refGeo has a hosted card, but neither route clears every preceding gate in
this audit. No multi-GB archive was downloaded.

## 14. CROMA token compatibility

CROMA has a validated internal 15×15 row-major token grid over the 120×120
SatQuery analysis grid. It can consume only geometry already expressed in a
verified SatQuery coordinate frame. Every audited external geometry remains
`MAPPING_UNVERIFIED` for this phase, so no source box, point, polygon, or mask
was mapped to CROMA's 225 tokens. CROMA tokens are spatial features, not a
grounding head or prediction.

## 15. S2 compatibility

The external candidates document RGB/aerial/overhead imagery or do not expose
the raw band contract. None is verified as canonical raw Sentinel-2
`[12,120,120]`. Any future integration is therefore `ADAPTER_REQUIRED`; the
S2 learned adapter/checkpoint remains unchanged.

## 16. SAR compatibility

No audited candidate establishes SAR grounding supervision with verified
geometry, raw VV/VH semantics, licensing, and provenance. VRSBench-SAR is
not accepted: its official repository notes annotations are automatically
generated by GPT-5.1 without human verification. SAR grounding status is
**NOT_VERIFIED**; the SAR projector remains unchanged.

## 17. Qwen grounding capability

SatQuery's verified Qwen path accepts RGB images and produces language. Its
grounding contract remains `NOT_VERIFIED`; no local coordinate decoder,
grounding head, region projector, segmentation model, or evaluated remote-
sensing grounding behavior exists. Qwen output must not be interpreted as a
bbox, point, mask, polygon, confidence, or evidence reference without a
separately validated implementation.

## 18. Provenance requirements

A future approved grounding record must retain: dataset/revision, image ID,
annotation ID, split, text, geometry, coordinate space and semantics version,
image dimensions, sensor, source path, image/annotation hashes, license, and
preprocessing version. A future learned prediction additionally requires model
and adapter revisions, prediction geometry, confidence only when measured, and
an evidence reference. No such prediction is created here.

## 19. Training readiness

All required gates must pass for one source: explicit grounding supervision,
real imagery, exact image–annotation linkage, authoritative geometry and
coordinate semantics, dimensions, licensing, immutable pin, official split,
leakage audit, reproducible provenance, bounded acquisition, and SatQuery
compatibility. The audited landscape leaves unresolved at least licensing,
pinning, full linkage verification, coordinate semantics, dimensions,
source/provenance, and leakage for every external candidate. BigEarthNet.txt
also remains blocked by its documented spatial-semantics gap.

Therefore the only valid result is **GROUNDING_TRAINING_BLOCKED**.

## 20. Phase 3K.2 gate

Phase 3K.2 may start only after an individual source clears all Section 19
conditions. It must first acquire at most 3–5 pinned examples, validate
image–annotation linkage and source geometry without conversion, record
coordinate semantics and dimensions, then decide whether a deterministic map
to the SatQuery grid is justified. Training, pseudo-labeling, and Phase 2F
changes remain out of scope until then.

## Immutability

No CROMA, scientific predictor/checkpoint, `physical_62d`,
`joint_croma_gap_768d`, `hybrid_830d`, split, representation artifact, receipt,
S2 adapter/checkpoint, SAR projector, temporal contract/artifact, or Phase 2F
file was modified. The frozen scientific baseline remains **65.0% accuracy**,
**4.1034286734 pp MAE**, and **9.5873312123 pp RMSE**.
