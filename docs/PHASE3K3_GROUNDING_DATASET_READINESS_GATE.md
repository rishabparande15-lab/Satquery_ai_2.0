# Phase 3K.3 — Grounding Dataset Re-evaluation and Acquisition Gate

## Final decision

**GROUNDING_TRAINING_BLOCKED**

No re-audited or newly reviewed source meets every SatQuery requirement for future text-to-region training. No external example was acquired because no candidate first cleared grounding validity, licensing, source pinning, and sample-level linkage gates.

## 1. Objective and 2. Prior grounding blockers

This re-audits authoritative RS grounding releases without training, conversion, labels, splits, or evaluation. Phase 3K.1's blockers remain: no complete evidence chain for geometry, license, pinning, linkage, provenance, and leakage. Phase 3K.2 remains fail-closed. BigEarthNet.txt stays `SPATIAL_ANNOTATION_AVAILABLE`, `SOURCE_GEOMETRY_PRESERVED`, and `MAPPING_UNVERIFIED`.

## 3. Candidate re-audit and 4. Newly identified sources

| Dataset | Grounding task / geometry | Current authoritative evidence | Status |
|---|---|---|---|
| VRSBench | Object referring / boxes; evaluation boxes documented as normalized 0–100. | Hosted train/validation annotation/image archives, evaluation referring JSON, and current visible commit `6cee296`; text annotations CC-BY-4.0 but DOTA content is separately academic-use-only. | **GROUNDING_TRAINING_BLOCKED** |
| OPT-RSVG | Referring expression / bbox. | Official repository documents 25,452 images, 48,952 image-query pairs, split counts, and external Drive/Netdisk delivery. | **GROUNDING_TRAINING_BLOCKED** |
| DIOR-RSVG / RSVG | Referring expression / bbox. | Official source documents 26,991 train, 3,829 validation, 7,500 test samples, XML query/bbox files, JPEG images, and ID split files; Google Drive delivery. | **GROUNDING_TRAINING_BLOCKED** |
| GeoChat / GeoChat_Instruct | Region captions, referring expression, rotated bbox dialogue. | Public code/data route, but no inspected self-contained immutable image/annotation manifest, coordinate serialization, image-license chain, or leakage audit. | **GROUNDING_TRAINING_BLOCKED** |
| refGeo / GeoGround | HBB, OBB/polygon, mask referring expression. | Hosted schema exposes `question_id`, `image_id`, `bbox`, `poly`, question, dataset; GeoGround documents 161k image-text pairs / 80k images. Viewer schema error, upstream image terms, full coordinate convention, data hashes, and leakage evidence remain incomplete. | **GROUNDING_TRAINING_BLOCKED** |
| GeoPixelD (new) | Pixel-level grounded conversation and phrase/mask grounding. | Official card declares CC-BY-4.0 and current card commit `5074aed`; it documents 53k+ phrases and 600k+ objects, but requires separate iSAID/DOTA imagery and 800×800 patch preparation. | **GROUNDING_TRAINING_BLOCKED** |

GeoPixelD is newly reviewed, not selected or ranked. Its separate upstream-image preparation, incomplete image-license chain, no complete immutable image+annotation manifest, incomplete mask semantics, and no bounded individual-object route leave the decision unchanged.

## 5. Image availability, 6. annotation availability, and 7. linkage

VRSBench publishes 8.36 GB train and 3.98 GB validation image ZIPs. DIOR-RSVG/OPT-RSVG use external Drive/Netdisk delivery. GeoPixelD directs users to separate iSAID/DOTA imagery. All candidates document task annotations. DIOR-RSVG provides XML query/bbox annotations, JPEG images, and ID split files; refGeo visibly has `question_id + image_id + bbox/poly`. This phase did not materialize any pinned image/annotation subset or verify source bytes and hashes, so these descriptions are not accepted as complete sample-level linkage. Filename similarity was never treated as evidence.

## 8. Geometry types, 9. coordinate semantics, and 10. image dimensions

VRSBench, OPT-RSVG, DIOR-RSVG, and GeoChat document boxes; GeoChat/refGeo document rotated boxes or polygons; refGeo and GeoPixelD document masks. Point grounding was not verified. VRSBench's 0–100 range does not establish origin, axes, dimensions, endpoint bounds, or preprocessing transform. Other candidates also lack one or more required coordinate-space, origin, axes, bounds, or dimension facts in inspected sources. GeoPixelD preparation names 800×800 patches, but this is not per-sample image/geometry validation. All external geometry remains **MAPPING_UNVERIFIED**; no conversion occurred.

## 11. License and 12. source pinning

| Dataset | License / pinning finding |
|---|---|
| VRSBench | Text CC-BY-4.0, DOTA upstream material academic-only; visible hosted commit `6cee296`, but no verified complete file manifest: **PARTIAL**. |
| OPT-RSVG | Image/annotation terms and immutable archive receipt not established: **UNKNOWN**. |
| DIOR-RSVG / RSVG | CC-BY-NC-4.0 / research-only; upstream DIOR chain and immutable data receipt not independently verified: **PARTIAL**. |
| GeoChat | Separate image/annotation terms and immutable data receipt not established: **UNKNOWN**. |
| refGeo | `s-lab-1.0` card license; upstream imagery/generated-mask chain and complete receipt incomplete: **PARTIAL**. |
| GeoPixelD | CC-BY-4.0 card / `5074aed`, but separately obtained iSAID/DOTA imagery and its complete immutable receipt remain unverified: **PARTIAL**. |

Code commits and card commits do not silently pin externally sourced imagery.

## 13. Official splits and 14. leakage audit

DIOR-RSVG has train/val/test ID files; OPT-RSVG publishes train/validation/test counts; VRSBench visibly has train/validation archives and evaluation JSON. No candidate exposed enough locally inspected pinned data for duplicate image/file/region/annotation, same-scene, geographic, or near-duplicate review. Every candidate is `LEAKAGE_AUDIT_INCOMPLETE`. No SatQuery split changed.

## 15. Bounded acquisition and 16. real sample validation

**BOUNDED_FIXTURE_BLOCKED** for every candidate. Required preceding gates remain incomplete, and available routes are multi-GB or external archives or separate image preparation. No archive was downloaded. Accordingly, real sample validation through `grounding_contracts.py` is **NOT RUN**; no image, query, geometry, dimensions, split, license metadata, or hash is claimed.

## 17. S2 compatibility and 18. SAR compatibility

Candidates are RGB/aerial or omit raw-band contracts; none is canonical raw Sentinel-2 `[12,120,120]`, so all are `ADAPTER_REQUIRED`. No candidate supplies verified SAR VV/VH grounding. VRSBench-SAR is not accepted as training evidence because its own project says its annotations are automatically generated by GPT-5.1 without human verification. S2 and SAR paths were not changed.

## 19. Multimodal/temporal grounding, 20. CROMA compatibility, and 21. Qwen compatibility

No candidate clears the audit for optical+SAR, temporal, or multi-image grounding. No geometry is mapped to SatQuery's analysis grid or CROMA 15×15/225 tokens. Qwen2.5-VL-3B-Instruct remains RGB-language-only in SatQuery: no coordinate decoder, region projector, mask head, or verified RS-grounding result exists.

## 22. Provenance

A future accepted record must preserve dataset/revision, image and annotation IDs, text, geometry/type/space, dimensions, sensor, source path, file/annotation hashes, split, license, and preprocessing. A prediction separately requires model/adapter revisions, predicted geometry/space, confidence/source, and evidence reference. Current contracts preserve this distinction; this phase creates no prediction.

## 23. Final training readiness

Every candidate fails at least one all-required condition: complete valid image+text+geometry linkage; authoritative coordinate semantics and dimensions; verifiable image/annotation license chain; complete immutable pinning; leakage audit; bounded 3–5 acquisition; or real sample structural validation. The only valid outcome is **GROUNDING_TRAINING_BLOCKED**.

## 24. Phase 3K.4 gate

**GROUNDING BRANCH FROZEN AT CONTRACT/DATA AUDIT LEVEL.** Controlled grounding-model execution is not authorized until one source clears every Section 23 condition.

## Scientific immutability

No CROMA, scientific predictor/checkpoint, `physical_62d`, `joint_croma_gap_768d`, `hybrid_830d`, scientific split/receipt, S2 adapter, SAR projector/artifact, temporal contract/receipt, or Phase 2F semantic was modified. The frozen baseline remains **65.0% accuracy**, **4.1034286734 pp MAE**, and **9.5873312123 pp RMSE**.
