# Phase 2H - Representation, Compute and Storage Policy

## 1. Objective and decision

This phase defines the representation policy before concrete VLM work. It does
not select a VLM, download VLM weights, implement VQA/captioning/grounding, or
change the scientific Pipeline 3.

The central architectural decision is:

```text
scientific representation != VLM representation

raw data -> verified representation contract -> model-specific adapter
```

The current policy is **documentation-only**. Existing typed contracts already
carry representation type, modality, shape, dtype, spatial level, logical
artifact URI, checksum, producer, preprocessing, and provenance. A new generic
`RepresentationRequest` class is not justified until a concrete adapter has a
real requirement for lazy loading, batching, or format negotiation. Adding one
now would duplicate `TaskRequest`, `RepresentationRef`, `RepresentationInput`,
`VisionInputContract`, and `ArtifactResolver`.

Phase 2F remains unchanged: BigEarthNet.txt geometry is source-only and cannot
be mapped to the 120 x 120 analysis grid or 15 x 15 CROMA tokens.

## 2. Existing inventory

| Representation or source | Current evidence | Current consumer |
| --- | --- | --- |
| Sentinel-2 optical raw imagery | Validated native bands and B02-defined 120 x 120 analysis grid in the Pipeline 3 loader; not locally materialized in this workspace | Scientific preprocessing and future adapters |
| Sentinel-1 SAR raw imagery | Validated VV/VH paired data; not locally materialized in this workspace | Scientific preprocessing and future adapters |
| Reference raster | Validated scientific target input; not a learned grounding mask | Target construction and scientific evaluation |
| CRS, affine, bounds, resolution, identity metadata | Stored in manifests, scene contracts, and source provenance | Validation, alignment, reproducibility |
| `physical_62d` | 62 float32 dimensions, verified for all 5,000 areas | Scientific feature path; optional future physical context |
| Optical CROMA tokens | Contract/catalog type exists, but current catalog status is `MISSING` for all 5,000 areas | No current consumer |
| SAR CROMA tokens | Contract/catalog type exists, but current catalog status is `MISSING` for all 5,000 areas | No current consumer |
| Joint CROMA tokens | 225 x 768 spatial vectors are supported by the CROMA contract, but not materialized in the current catalog | Future model-specific adapter only |
| Optical/SAR/joint CROMA GAP | Current core materialization contains joint scene-level 768-D vectors; optical and SAR GAP families are optional/missing | Joint scene context and scientific predictor input |
| `joint_croma_gap_768d` | 5,000 verified, checksum-backed, receipt-backed vectors | Scientific predictor and future scene-level visual adapter |
| `hybrid_830d` | `physical_62d` concatenated with `joint_croma_gap_768d`; 5,000 verified | Current scientific predictor only |
| Regions/evidence | Deterministic evidence contracts exist; learned regions are unavailable | Analysis/evidence runtime, not VLM grounding supervision |
| Temporal pair/difference/fusion | Contract fields reserve temporal identity, but no authoritative paired observations exist | Deferred |
| Metadata embeddings | No implementation or verified consumer | Deferred |

The current core materialization is 5,000 areas with 4,600 train, 200
validation, and 200 test samples. It contains 157 resumable shards and 15,000
typed receipts for the three core representations.

## 3. Representation policy matrix

Storage values in this table are classified in Section 4. A missing value is
not an estimate; it means the artifact is not locally available or its format
is not fixed.

| Representation | Modality | Dimensions | Spatial information | Temporal information | Status and consumer | Future VLM relevance | Materialization policy | Decision |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| Raw optical image | optical | native bands; analysis output 12 x 120 x 120 | pixel/georeferenced after validation | scene timestamp when supplied | source/preprocessing only | high for image-capable adapters | selectively cached or streamed; no full duplicate | ON_DEMAND |
| Raw SAR image | SAR | 2 x 120 x 120 analysis output | pixel/georeferenced after validation | scene timestamp when supplied | source/preprocessing only | high for multimodal adapters | selectively cached or streamed; no full duplicate | ON_DEMAND |
| Reference raster | target/reference | source raster; validated target 120 x 120 | pixel target geometry | scene-specific | scientific target only | not a default VLM input | keep with scientific source receipt; never treat as grounding labels | KEEP |
| Metadata and CRS/affine | metadata/geospatial | structured | geospatial frame and provenance | acquisition metadata | manifests and scene contracts | adapter-dependent context | keep in manifests/receipts; load lazily | KEEP |
| `physical_62d` | optical + SAR-derived | 62 | scene-level statistics | no temporal encoding | mandatory scientific feature | optional physical context | already materialized and receipt-backed | KEEP |
| Optical CROMA tokens | optical | 225 x 768 | 15 x 15 token grid | none | catalog type, absent | possible future optical adapter | generate only for a validated consumer | DEFER |
| SAR CROMA tokens | SAR | 225 x 768 | 15 x 15 token grid | none | catalog type, absent | possible future SAR adapter | generate only for a validated consumer | DEFER |
| Joint CROMA tokens | optical + SAR | 225 x 768 | 15 x 15 token grid | none | supported by CROMA, absent from core materialization | high for spatial-capable future adapters | on-demand or bounded cache after adapter contract | ON_DEMAND |
| `joint_croma_gap_768d` | optical + SAR | 768 | scene-level pooled context | none | 5,000 verified; current scene visual representation | high for scene-level adapters | already materialized and receipt-backed | KEEP |
| `hybrid_830d` | optical + SAR | 830 | scene-level | none | current scientific predictor input | not automatically a VLM input | preserve existing materialization only | KEEP |
| Region representation | source/derived | unspecified | region-level | none | no verified source geometry mapping | future grounding only after evidence | no materialization without geometry contract | DEFER |
| Temporal pair | optical/SAR pair | unspecified | correspondence required | T1/T2 | no authoritative temporal identity | future change adapters | generate on demand after paired-scene contract | DEFER |
| Temporal difference/fusion | derived pair | unspecified | correspondence required | T1/T2-derived | not implemented | future model-specific input | no cache before model/evaluation contract | DEFER |
| Metadata embedding | metadata | model-dependent | none | possible dates/context | no implementation | adapter-specific | no representation before consumer exists | DEFER |
| Model input tensor | adapter-specific | model-dependent | adapter-defined | adapter-defined | not implemented | required only after model selection | generate on demand, cache only with receipt | ON_DEMAND |
| Model output | adapter-specific | model-dependent | output contract | adapter-defined | not implemented | future evaluation artifact | persist only as provenance-backed output | ON_DEMAND |

## 4. Storage accounting

### 4.1 Measured local storage

Measured from the current filesystem on 2026-09-19. No artifacts were copied or
regenerated for measurement.

| Artifact root | Files | Bytes | MiB | Classification |
| --- | ---: | ---: | ---: | --- |
| `artifacts/pipeline3_5000/representations` | 320 | 34,296,453 | 32.71 | MEASURED |
| `artifacts/pipeline3_5000/representations/representations` (`.npz` shards) | 157 | 15,671,729 | 14.95 | MEASURED |
| `artifacts/pipeline3_5000/representations/receipts/records.jsonl` | 1 | 16,522,696 | 15.76 | MEASURED |
| Per-shard representation receipts | 157 | 582,662 | 0.56 | MEASURED |
| `artifacts/representation_catalog/pipeline3_5000` | 4 | 55,915,684 | 53.33 | MEASURED |
| `artifacts/representation_links/bigearthnet_txt_phase2d2` | 4 | 277,860,220 | 264.99 | MEASURED |
| `data/raw/bigearthnet_txt` | 15 | 466,875,206 | 445.24 | MEASURED; annotation/source artifact, not imagery |
| Local raw Sentinel-1/Sentinel-2 5,000-area imagery | 0 | UNKNOWN | UNKNOWN | NOT LOCALLY AVAILABLE |
| Local 5,000-area CROMA spatial-token export | 0 | UNKNOWN | UNKNOWN | NOT LOCALLY AVAILABLE |

The representation root includes manifests and provenance in addition to the
listed shard and receipt components. The catalog and annotation-link sizes are
not representation tensor sizes and must not be counted as feature bytes.

### 4.2 Calculated payload sizes

These values are mathematical payload calculations, not filesystem
measurements. They assume float32 for numeric representation arrays and exclude
container compression, array headers, filesystem blocks, and receipt metadata.

| Representation | Calculation per sample | Calculated payload for 5,000 |
| --- | ---: | ---: |
| `physical_62d` | 62 x 4 = 248 bytes | 1,240,000 bytes (1.18 MiB) |
| `joint_croma_gap_768d` | 768 x 4 = 3,072 bytes | 15,360,000 bytes (14.65 MiB) |
| `hybrid_830d` | 830 x 4 = 3,320 bytes | 16,600,000 bytes (15.83 MiB) |
| One CROMA spatial token grid | 225 x 768 x 4 = 691,200 bytes | 3,456,000,000 bytes (3.22 GiB) |
| Optical, SAR, and joint token grids | 3 x 691,200 bytes | 10,368,000,000 bytes (9.66 GiB) |
| Normalized optical + SAR analysis tensors | 14 x 120 x 120 x 4 = 806,400 bytes | 4,032,000,000 bytes (3.75 GiB) |

The raw optical native payload depends on band dtypes and TIFF/container
metadata. Its pixel-count calculation is therefore not presented as a raw
storage measurement. Reference-map dtype and compression also remain source
dependent.

## 5. Compute and local-resource implications

The planning machine has an NVIDIA RTX 5060 with 8 GB VRAM and 24 GB system
RAM. No VLM benchmark was run.

- `physical_62d`, GAP, and `hybrid_830d` are small per sample and can be
  streamed from receipt-backed shards. A full uncompressed 5,000-sample batch
  is unnecessary; use bounded shard or mini-batch reads.
- One spatial CROMA grid is approximately 0.66 MiB of float32 payload per
  sample. A 5,000-area spatial cache is approximately 3.22 GiB before
  overhead, while model activations and future adapter memory are additional.
  Spatial tokens must therefore be lazy or bounded-cache inputs on an 8 GB GPU.
- Normalized 14-channel analysis tensors are approximately 0.77 MiB per
  sample before reference maps and metadata. Keep them on CPU/storage and move
  only the requested batch to VRAM.
- Raw imagery should remain selectively acquired and locally cached by
  content-addressed or receipt-backed artifacts. Do not keep duplicate raw,
  prepared, and model-input copies without a consumer.
- CROMA inference should retain the existing bounded batch and shard policies;
  no larger batch is implied by this analysis.
- Cache reuse is valuable for deterministic preprocessing and verified GAP or
  token outputs, but cache invalidation must follow source, preprocessing,
  model/checkpoint, dtype, shape, and configuration fingerprints.

## 6. Scientific predictor policy

`hybrid_830d` is the current scientific predictor representation:

```text
physical_62d + joint_croma_gap_768d -> hybrid_830d -> frozen scientific head
```

It is **KEEP/MATERIALIZED** because the frozen baseline depends on it. It must
not become the default VLM input merely because it is available. The predictor
and its feature receipts remain isolated from future model-specific adapters.

`physical_62d` and `joint_croma_gap_768d` may be exposed as verified optional
context to a future adapter, but that decision belongs to the future adapter's
validated input profile. No current language model consumes them.

## 7. VLM input policy and boundary

Future adapters may request validated optical, SAR, optical+SAR, scene-level
CROMA, spatial CROMA, or physical metadata through the existing
`RepresentationInput` and `VisionInputContract` abstractions. The adapter, not
the core catalog, owns conversion to model tensors, tokenizer inputs, context
windows, and batching conventions.

The core must not depend on EarthDial, GeoChat, SkyEyeGPT, Prithvi,
RemoteCLIP, or any other individual model. Model-specific derived inputs are
**ON_DEMAND** and must carry the same receipt/provenance requirements as core
representations.

## 8. Optical-SAR policy

The representation layer preserves three distinct concepts:

1. independent optical and SAR representations;
2. joint CROMA representation generated from both modalities;
3. learned fusion or evidence fusion performed by a future consumer.

The 830-D concatenation is a scientific predictor feature, not a universal
fusion contract. Future adapters may request optical-only, SAR-only, or joint
inputs without forcing either modality into `hybrid_830d`. No learned fusion is
implemented here.

## 9. Temporal policy

Temporal support remains a contract-level possibility, not a representation
claim. A future temporal input must preserve T1 identity, T2 identity,
acquisition timestamps, source fingerprints, modality availability, and a
validated spatial-correspondence status. The current BigEarthNet.txt view has
no authoritative temporal identities, and paired S1/S2 sensors are not a
temporal pair.

Temporal pairs, differences, and fused representations are **DEFER** decisions
until a dataset and correspondence contract exists. No temporal cache or
representation is created.

## 10. Grounding and spatial constraints

Existing CROMA token representations are valid model outputs with row-major
15 x 15 indexing. That does not make BigEarthNet.txt source geometry valid
token supervision. Phase 2F remains authoritative:

- source geometry is preserved but unmapped;
- no source annotation becomes an analysis pixel or token reference;
- learned grounding is not implemented; and
- region representations are deferred until coordinate evidence and a future
  evaluated grounding task exist.

Reference rasters remain scientific targets. They are not silently promoted to
grounding masks or VLM supervision.

## 11. Cache policy

| Cache category | Policy | Determinism/provenance | Invalidation |
| --- | --- | --- | --- |
| RAW | selective/on-demand; cache optionally | source URL/revision, item hash, acquisition receipt | source revision, checksum, selection plan |
| PREPROCESSED | cache optionally | source identity, preprocessing profile, shape/dtype, checksum | source, profile, grid, dtype |
| FEATURES | materialize only for verified consumers | existing typed receipts and sample/artifact hashes | source, feature implementation, model/checkpoint, config |
| TOKENS | on-demand or bounded optional cache | token shape/order, CROMA revision/checkpoint, input fingerprint | input, CROMA revision, preprocessing, dtype |
| MODEL INPUTS | on-demand | adapter and model configuration plus input references | model, tokenizer/format, requested batch/profile |
| MODEL OUTPUTS | on-demand persisted evaluation artifacts | model, input, dataset, split, configuration, checksum | any upstream identity or evaluation configuration |

The existing receipt/catalog system is the only authoritative persisted
representation cache. No second cache registry is introduced.

## 12. Provenance policy

Every persisted representation must retain or resolve through its receipt:

- source scene/sample identity and dataset/split fingerprints;
- source and artifact checksum where available;
- representation type, modality, shape, and dtype;
- preprocessing configuration/version;
- producer and model/checkpoint revision;
- generation configuration and run reference; and
- logical artifact URI and artifact checksum.

`ArtifactResolver`, `RepresentationRef`, `RepresentationInput`, the
representation catalog, and typed materialization receipts already enforce this
boundary. Absolute local paths remain outside logical contracts.

## 13. Decisions

| Decision | Representations | Reason |
| --- | --- | --- |
| KEEP | raw source identity/metadata, CRS/affine, reference raster provenance, `physical_62d`, `joint_croma_gap_768d`, `hybrid_830d`, existing receipts/catalogs | Existing scientific or provenance consumers depend on them. |
| MATERIALIZE | Existing three core 5,000-area representations only | Already verified and required by the frozen scientific pipeline; do not regenerate. |
| ON_DEMAND | raw optical/SAR, model-specific inputs, joint/optical/SAR token grids when a future adapter requests them | Avoid multi-gigabyte duplication without a validated consumer. |
| CACHE_OPTIONALLY | validated preprocessed tensors, future token outputs, future model outputs | Cache only behind source/config/checkpoint fingerprints and bounded storage. |
| DEFER | temporal pairs/differences, region representations, metadata embeddings, optical/SAR learned fusion, grounding-linked representations | Dataset, model, correspondence, or evaluation contracts do not yet exist. |
| REJECT | treating `hybrid_830d` as the universal VLM input; mapping BigEarthNet.txt geometry to tokens; using reference rasters as learned grounding labels | Violates scientific separation or the Phase 2F fail-closed rule. |

## 14. Tests and regression

No code or representation artifact changed in Phase 2H, so no new tests were
needed. Existing architecture, representation, receipt, materialization,
image-language, and spatial tests remain the relevant focused coverage.

The required full regression should remain at the Phase 2G result of `359
passed, 5 skipped`. The scientific baseline remains:

- Accuracy: `65.0%`
- MAE: `4.1034 pp`
- RMSE: `9.5873 pp`

No retraining, VLM download, CROMA regeneration, spatial remapping, or model
benchmark was performed for this policy.

## 15. Remaining blockers before VLM work

Before concrete VLM implementation, the project still needs a model/task
contract, input shape and modality decision, benchmark protocol, temporal
identity source if applicable, and a grounding geometry decision independent
of the current CROMA token grid. Those are Phase 3 research decisions.

The Phase 2H answer is therefore:

```text
Scientific predictor:
  keep the verified physical_62d + joint_croma_gap_768d -> hybrid_830d path.

Future VLM inputs:
  request raw or verified representations through existing typed contracts;
  load raw/tokens/model inputs on demand and cache only with receipts.

Temporal/grounding/region representations:
  defer until their source and evaluation contracts exist.
```
