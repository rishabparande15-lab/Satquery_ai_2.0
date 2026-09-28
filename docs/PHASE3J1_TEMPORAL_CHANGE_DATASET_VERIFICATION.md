# Phase 3J.1 — Temporal / Change Dataset Verification Gate

Final status: **TEMPORAL_TRAINING_BLOCKED**

## 1. Objective

This audit evaluates authoritative sources for bi-temporal remote-sensing
change language supervision. It does not download imagery, train a temporal
model, alter Phase 3G/3H adapters, or create change labels.

## 2. CDVQA source

The primary source is the authors’ official [CDVQA repository](https://github.com/YZHJessica/CDVQA)
and the peer-reviewed 2022 TGRS paper, [*Change Detection Meets Visual Question
Answering*](https://doi.org/10.1109/TGRS.2022.3203314). The paper explicitly
introduces CDVQA on multi-temporal aerial images and says the dataset consists
of multitemporal image-question-answer triplets produced by automatic QA
generation. This establishes **TEMPORAL_SUPERVISION_VERIFIED** at the dataset
design level: questions target content changes between the two inputs, not two
unrelated images or locations.

The official repository publishes `Train`, `Val`, `Test`, and `Test2` image,
question, and answer JSON files under Apache-2.0. It has a mutable `main`
branch, no release/tag or image archive hash, and only seven commits visible;
the data source remains **SOURCE_PINNING_WEAK**. A specific Git commit could
pin those committed JSONs, but the audit did not find a source-pinned T1/T2
image release or a manifest that binds JSON rows to image hashes.

## 3. Dataset structure, pairing, and change annotation

| Property | Status |
|---|---|
| Real bi-temporal imagery required by task | VERIFIED from paper |
| Question/answer describes change between pair | VERIFIED from paper |
| Committed annotation partitions | VERIFIED: Train, Val, Test, Test2 JSON families |
| Exact pair ID / T1 ID / T2 ID schema | PARTIAL: image JSON files exist; fields require release-level inspection before use |
| Annotation ID schema | UNKNOWN |
| Image bytes and image manifest | BLOCKED: not in official repository/release located |
| Pair count / QA count | UNKNOWN from authoritative release material inspected |
| Sensor, bands, resolution, format, dates/interval | UNKNOWN |
| Change types | land-cover/content change at paper level; detailed taxonomy UNKNOWN |
| SAR modality | NOT AVAILABLE / not documented |
| Grounding geometry | NOT AVAILABLE in CDVQA evidence inspected |

The paper describes high-level change information from two input images. It
does not, in the public material inspected, provide timestamp fields or prove
pixel-level registration metadata. Temporal order is therefore
**T1/T2 UNKNOWN** for a record, and spatial relation is **UNKNOWN** under the
strict contract—not `COREGISTERED` merely because the task is change VQA.

## 4. License and split/leakage gate

The repository’s Apache-2.0 file is **VERIFIED for committed repository
content** (code and committed JSON). It does not establish a license for
unpublished/externally hosted T1 or T2 imagery. Thus T1 license,
T2 license, image redistribution permission, image acquisition provenance,
and imagery checksum are all **LICENSE_UNVERIFIED**. The annotation JSON can
be pinned only by a selected Git commit; no immutable archive/checksum is
published for it.

Train/Val/Test/Test2 annotation file families are present, but their exact
semantics, pair intersections, T1/T2 reuse, scene/geographic overlap,
near-duplicate risk, and image-level leakage cannot be checked without the
image manifests. No split was created or modified.

## 5. Compatibility with SatQuery

CDVQA is **ADAPTER_REQUIRED / UNKNOWN** for the existing S2 path: SatQuery
requires canonical raw 12-band S2 `[12,120,120]`, while the inspected CDVQA
materials do not identify sensor, band order, raw values, resolution,
preprocessing, or image dimensions. SAR is not documented; no optical–SAR
claim is made.

Conceptual future options only, not implementations:

```text
Option A: T1 tokens + T2 tokens -> learned temporal-difference/cross-attention
Option B: T1 + T2 -> shared temporal encoder -> language alignment
Option C: verified pair images -> documented multi-image Qwen pathway
```

Any future path must preserve T1 evidence, T2 evidence, explicit difference
evidence, timestamps/order, pair IDs, source hashes, and split lineage.
Generated text is not temporal evidence by itself.

## 6. Other authoritative temporal-language candidates

[ChangeChat-105k / DeltaVLM](https://github.com/hanlinwu/DeltaVLM) is an
author-published alternative, not a replacement silently merged into CDVQA.
Its documentation is materially more specific: pre/post `image_A/image_B`
pairs from the same LEVIR-CC tile; six interaction types (captioning,
classification, quantification, localization, open QA, dialogue); published
JSON field examples; 87,935 mixed training records; and test files. It states
annotations are CC-BY-4.0 on Hugging Face while images must be separately
obtained from LEVIR-CC under that upstream license. It therefore has explicit
temporal-language intent and split structure, but still requires an upstream
LEVIR-CC image-license and immutable image-manifest audit before sample
acquisition. It is not training-ready in this phase.

[QAG-360K / CDQAG](https://like413.github.io/CDQAG/) is another author-published
change QA-and-grounding benchmark: it reports more than 360K question/text
answer/mask triplets, 10 land-cover categories, and eight question types. It
is relevant for later change grounding, but its dataset release/license,
image source, splits, and record-level schema were not verified here.

## 7. Small sample acquisition, provenance, and final gate

**NOT RUN.** CDVQA fails the imagery-license, image-source/pinning,
pair-metadata, temporal-order, spatial-correspondence, and structural-input
gates. No legal/reproducible 3–5 pair acquisition route was established, so no
T1/T2 hashes, pair/annotation IDs, S2-boundary checks, or structural tests are
claimed.

To lift the gate, obtain an author-published, version-pinned image release
whose manifest joins every T1/T2 file and its timestamps to the official QA
record and split, with explicit imagery/annotation terms and checksums. Then
inspect no more than five pairs for temporal order, spatial correspondence,
duplicates, raw modality, and adapter-boundary compatibility before training.

BigEarthNet.txt remains single-image optical language data. CDVQA and every
external candidate remain separate from the scientific split and receipt
lineage. CROMA, scientific representations/checkpoints, Phase 2F, S2/SAR
adapters, and the locked baseline (65.0% accuracy; 4.1034286734 pp MAE;
9.5873312123 pp RMSE) are unchanged.

## 8. Phase 3J.2 architecture gate

Phase 3J.2 is blocked until one source clears all gates: explicit change
supervision, real pair bytes, exact pair/annotation linkage, temporal order,
spatial correspondence, verified licenses, pinned release, official split,
complete provenance, bounded sample acquisition, and structural validation.
