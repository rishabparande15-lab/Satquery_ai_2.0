# Phase 2E Image-Language Foundation

## 1. ImageLanguageSample schema

`image_language_sample_v1` is the canonical, model-independent record. It contains stable sample/annotation/image identity, inherited split, task type, explicit modalities, typed verified representation references, exact source question/answer/choices/caption/text, optional source-only spatial reference, source provenance, validation state, and optional temporal identity fields. BigEarthNet.txt temporal fields are always null because no authoritative temporal identities exist.

The existing `image_language_record_v1` Phase 2A pooling layer remains unchanged and reusable. Phase 2E joins it to the authoritative image, representation, receipt, linkage, and split contracts rather than replacing it.

## 2. Task vocabulary

Current source mappings are `VQA_BINARY`, `VQA_MULTIPLE_CHOICE`, `CAPTIONING`, `GROUNDING_TEXT_BOX`, and `GROUNDING_POINT`. `TEMPORAL_VQA`, `CHANGE_DESCRIPTION`, and `CHANGE_GROUNDING` are reserved enum values only and are rejected by the BigEarthNet.txt adapter.

## 3. Modality vocabulary

Inputs distinguish `OPTICAL`, `SAR`, and `OPTICAL_SAR`. Current verified core inputs are optical-SAR physical features, joint CROMA GAP, and the evaluated hybrid. Raw optical/SAR and spatial token catalog rows are currently missing and are not fabricated.

## 4. Representation input contract

`RepresentationInput` requires a verified reference identity, explicit variant, modality, shape, dtype, logical URI, artifact and optional sample checksum, producer/model/preprocessing provenance, and spatial level. `VisionInputContract` exposes available modalities and representations, spatial resolution, dimensions, provenance, and preprocessing without choosing a model tensor format.

The default future visual profile identifies `joint_croma_gap_768d` as primary visual context and `physical_62d` as optional physical context. `hybrid_830d` is identified as the frozen scientific predictor input and is explicitly **not** assumed to be a VLM input.

## 5. VLM adapter interface

`VLMAdapter` and `VisionInputAdapter` define validation, input preparation, capabilities, and provenance. Generation and scoring default to unsupported. `VLMCapabilities` explicitly reports supported operations and VQA, captioning, grounding, image-only, multi-image, and optical-SAR support. No concrete VLM, tokenizer, prompt format, encoder, context window, weights, inference, or training is selected.

## 6. Prompt/template boundary

`PromptTemplate` contains template identity/version, task, required input fields, output format, and minimal deterministic rendering. Rendered prompts are marked derived and retain the source annotation ID. Source question, answer, choices, caption, and text are never overwritten.

## 7. Model-output contract

`ModelOutputType` reserves free text, categorical answer, multiple-choice answer, structured answer, and grounding output. `ModelOutput` carries adapter and provenance identity. These are output contracts only; no generation or scoring is implemented.

## 8. Grounding hook

`GroundingOutput` reserves phrase, region, bbox, mask, polygon, confidence, provenance, and an explicit learned-prediction flag. Existing geometric correspondence is not described as learned grounding. Phase 2E emits no grounding predictions.

## 9. Split/leakage policy

Samples inherit the authoritative image split. The grouping unit is `image_id`; annotations are never randomly split. All annotations for an image remain together, optical/SAR identity remains paired, representation split and provenance must agree, and benchmark/temporal grouping remains governed by the existing leakage policies. Real-data audit found zero cross-split image intersections.

## 10. BigEarthNet.txt dataset view

`scripts.build_image_language_foundation` streams validated annotations in canonical annotation-ID order, joins each to the authoritative image and verified core representation catalog, and writes `manifest.json`, `samples.jsonl`, `validation_report.json`, and `provenance.json`. It copies no raw imagery and does not duplicate the source annotation artifact. Quarantined Phase 2A rows remain excluded and counted.

Logical streaming views are `all`, `train`, `validation`, `test`, `binary_vqa`, `mcq_vqa`, `captioning`, and `grounding`. They are filters over the one canonical `samples.jsonl`, not independent datasets.

## 11. Benchmark adapter hooks

`BigEarthNetTextAdapter` is implemented. `RSVQAAdapter`, `VRSBenchAdapter`, and `CDVQAAdapter` are explicit future hooks whose assembly methods raise `NotImplementedError`. No external benchmark was downloaded or evaluated.

## 12. Provenance

Every sample traces to annotation ID/schema/source record and hash, image/dataset/split, and three stable typed representation reference IDs. Full artifact/sample checksums and producer/model/preprocessing metadata are resolved from the fingerprinted authoritative catalog instead of being copied into every annotation row. Canonical manifests use logical artifact references and contain no machine-local absolute paths.

## 13. Fingerprinting

The manifest records annotation SHA-256/schema, dataset and split fingerprints, representation catalog fingerprint, representation-link fingerprint, build version, ordering rule, sample SHA-256, and a canonical image-language fingerprint. Repeated builds must produce identical sample and manifest bytes.

## 14. Real-data statistics

- Validated samples: 101,542; excluded Phase 2A quarantines: 29
- Train: 93,208; validation: 4,175; test: 4,159
- Binary VQA source records: 37,503
- Multiple-choice VQA source records: 33,969
- Caption source records: 4,770
- Text-box source records: 12,178
- Point source records: 13,122
- Unique images: 4,770
- Samples with verified complete core set: 101,542
- Samples missing required core representations: 0
- Validated spatial samples deliberately unmapped: 25,300
- Provenance, split, malformed, and duplicate failures: 0

Sample SHA-256: `7db25d8b6f40034fd15795edb500ee3265ce11551ad1af540622782b99a39e31`. Image-language fingerprint: `2e2ccb5c85147519ee4b19d639a689408ba6ea01dd2f2ed98a70b847b3eb6ae0`. Compact typed references keep the canonical sample artifact at 220,685,042 bytes rather than duplicating full catalog metadata per annotation.

## 15. Limitations

This is not an evaluated VQA/captioning/grounding dataset or model. Raw optical/SAR and token representations are not materialized in the representation catalog. BigEarthNet.txt spatial coordinate semantics remain unknown. Multiple-choice options embedded in source questions are deterministically parsed by the preserved Phase 2A utility and identified as derived metadata; the source question and answer remain exact.

## 16. Future work

A later phase may choose and implement one or more VLM adapters, materialize required raw/token inputs, define benchmark-specific evaluators, and validate model outputs. Temporal work requires authoritative scene/observation/group identities first. None of these future capabilities is implied by Phase 2E.
