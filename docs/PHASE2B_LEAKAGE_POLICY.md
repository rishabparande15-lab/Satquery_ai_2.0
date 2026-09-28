# Phase 2B Task-Safe Leakage Policy

## Policy identity

Task policies are defined in [src/leakage_policy.py](../src/leakage_policy.py) as `leakage_policy_v1`. They report risk and fail closed on structural split conflicts. They never delete records, relabel splits, or call repeated text contamination by itself.

## Shared rules

- The image is the minimum grouping unit for single-image tasks.
- All annotations and regions from one image inherit one split.
- An optical-SAR pair is one sample; its components cannot be separated.
- Features must retain source image, split, preprocessing version, and model/checkpoint provenance.
- Invalid or missing grouping identity is a validation failure, not a reason to guess.
- Exact or near-identical text is measured separately from image leakage.

## Task policies

| Task | Immediate grouping unit | Required policy |
|---|---|---|
| Scene classification / coverage | Image | Group all annotations, raster modalities, and derived features by image; preserve frozen Pipeline 3 split |
| Image VQA | Image | Group all Q&A records by image; report question and question-answer overlap separately |
| Captioning | Image | Group captions by image; report repeated caption/template overlap separately |
| Grounding | Image | All boxes, points, references, and regions from one image share its split |
| Temporal/change | Temporal scene identity | Group all dates of one geographic scene or validated sequence; never split temporal scene identity across partitions |
| Optical-SAR multimodal | Optical-SAR pair | Require both sensor identities to remain in one split and retain pair provenance |

## Leakage categories

### Image-level leakage

Any image ID present in more than one split is structural leakage and fails closed. Phase 2A found zero cross-split image leakage.

### Pair leakage

Optical and SAR identities are treated as one semantic sample. A pair identity split across partitions fails closed, even if one component has a different filename representation.

### Region leakage

Region annotations are not independent samples for split purposes. All regions from an image follow the image split. Spatial evidence regions generated from pixels/tokens retain the originating artifact and scene identity.

### Temporal leakage

Phase 2B defines the required policy but does not implement temporal data. Future temporal ingestion must supply a validated `temporal_scene_id`; acquisition date alone is insufficient. Same geographic scene across dates must be grouped according to the future benchmark’s sequence policy.

### Question and caption overlap

Repeated question strings, question-answer pairs, captions, or templates across splits are reported, not automatically removed. They may affect learned language generalization, but they are not image leakage unless the associated image/group identity also crosses splits.

### Feature/model leakage

Precomputed features must reference their source image and split plus preprocessing profile, model/checkpoint identity, and artifact checksum. Features from a held-out split must not be used for training or model selection. The policy module reports missing feature provenance where records declare features without provenance.

## The 2,001 QA overlaps

The verified Phase 2A artifact reports 2,001 identical question-answer contents across different experiment splits. These are repeated textual templates/content attached to different images. The same audit reports:

- image-level leakage: `0`
- duplicate image/question combinations: `0`
- test rows assigned to train: `0`

Therefore these records are content overlap, not demonstrated image contamination. They must remain available for audit and may require a stricter language-template holdout in a future VQA benchmark, but Phase 2B does not delete or relabel them.

## Enforcement boundary

The policy is immediately applicable to annotation and evidence audits. Temporal grouping is reserved until authoritative temporal scene IDs and sequences exist. Caption-template similarity beyond exact string equality requires a defined tokenizer/similarity protocol and is deferred. No learned model or training split is created in Phase 2B.
