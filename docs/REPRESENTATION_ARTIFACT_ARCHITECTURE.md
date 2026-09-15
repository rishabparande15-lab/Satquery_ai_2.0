# Representation and Artifact Architecture

## 1. Why representation references exist

Pipeline 3 produces scientifically useful values in several lifecycles: some
exist only during `run_analysis`, some are included as JSON values in its live
result, and some have independently persisted NumPy or JSON artifacts. Future
capabilities must be able to discover those states without guessing paths,
loading tensors unnecessarily, or claiming that discarded values are reusable.

A reference is metadata and identity. It is not necessarily the loaded tensor.

```text
                    SceneBundle
                        │
                 RepresentationSet
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
 RepresentationRef  RepresentationRef  RepresentationRef
   CROMA scene       CROMA tokens       physical
        │               │                │
        ▼               ▼                ▼
   ArtifactRef      ArtifactRef       ArtifactRef
        │
        ▼
 optional resolver/materializer
        │
        ▼
 actual representation
```

## 2. RepresentationRef

`RepresentationRef` is a frozen dataclass identifying a representation by
reference ID, scene, controlled type, modality, optional version, shape, dtype,
spatial semantics, producer, preprocessing, artifact, provenance reference, and
status. Its statuses are:

- `materialized`: the live `RepresentationSet` already carries the value;
- `available`: a verified `ArtifactRef` provides a materialization path;
- `not_addressable`: Pipeline 3 produced or used it, but the live result did not
  retain material or a stable artifact;
- `unavailable`: it was not produced.

An `available` reference is invalid without an available artifact. Shapes contain
positive dimensions and dtypes use a bounded numeric/JSON vocabulary.

Controlled representation types are raw optical, raw SAR, physical features,
optical CROMA, SAR CROMA, joint CROMA, pooled CROMA scene, hybrid, regions,
evidence, and metadata. Token versus scene CROMA is distinguished by explicit
spatial level rather than by an ambiguous new semantic label.

## 3. ArtifactRef

`ArtifactRef` is a separate frozen identity for a concrete NumPy array, JSON
document, image, report, or provenance receipt. It records a logical
`artifact://` URI and may record SHA-256, byte size, media/format, producer,
producer version, run, and provenance reference.

Absolute local paths are rejected as architectural URIs. For example,
`artifact://final-audit/sample-61_39/croma_joint_encodings.npy` remains stable
while each environment configures where the `final-audit` namespace lives.

## 4. RepresentationSet integration

`RepresentationSet.references` is an immutable tuple alongside the existing
loaded-material fields. `find` supports controlled type, modality, spatial level,
and addressable-only filtering. `with_reference` returns a new set and rejects
duplicate reference IDs. This preserves compatibility while clearly separating
metadata/reference from loaded material.

## 5. SceneBundle integration

`SceneBundle` carries the extended `RepresentationSet`; its existing serialization
includes the references, nested artifacts, spatial metadata, and provenance.
The deterministic compatibility adapter now describes live outputs without
altering the legacy result:

- loaded physical, pooled CROMA, hybrid, region, evidence, and metadata values
  are `materialized`;
- raw raster arrays and individual CROMA token/GAP outputs omitted by the live
  result are `not_addressable`;
- persisted receipts can separately create `available` references.

## 6. Spatial semantics

Spatial meaning is explicit and deliberately conservative:

| Level | Meaning |
|---|---|
| `pixel` | Raster grid material with CRS where known |
| `feature` | Feature vector; not a spatial map |
| `scene` | One whole-scene representation |
| `token` | Ordered 15×15 token grid; `[225,768]` for current CROMA |
| `region` | Deterministic connected evidence regions |
| `evidence` | Evidence spanning token, region, and claim references |
| `metadata` | Descriptive scene/processing metadata |

CROMA tokens are latent spatial tokens, not learned grounding or segmentation.
For verified `61_39` artifacts their row-major 15×15 grid maps 120×120 inputs in
8×8 cells; the mapping reference points to existing spatial-evidence scene
metadata. Pooled CROMA and hybrid vectors are scene-level. The 62-value physical
vector is feature-level.

## 7. Provenance

References link to the existing run or receipt rather than duplicating its
provenance. The final-audit receipt already records request and scene identities,
source hashes, CRS/grid, normalization profile, CROMA checkpoint hash, physical
schema, Phase 2 adapter, and hybrid configuration. Live references point to the
current analysis run; persisted references point to the logical receipt URI.

The trace is therefore `TaskResult → SceneBundle reference → artifact → existing
receipt/run → producer/preprocessing/input scene`.

## 8. Checksums

`reference_id` identifies the architectural reference. `artifact_id` identifies
the artifact record. SHA-256 identifies concrete artifact content. These are not
interchangeable.

The final-audit receipt contains actual SHA-256 values for every `.npy` file.
The spatial-evidence provenance contains a canonical content hash for the full
evidence object; it is not automatically treated as the byte hash of every split
JSON file. Live in-memory values have no invented checksum. `artifact_ref_from_file`
can calculate a real byte checksum when an existing file is deliberately added
to a logical namespace.

## 9. Materialization boundary

`ArtifactResolver` is intentionally small. A caller configures logical namespace
roots, then the resolver:

1. resolves an `artifact://` URI within its configured root;
2. rejects namespace and path traversal errors;
3. verifies optional byte size and SHA-256;
4. loads NumPy with pickle disabled; and
5. validates the loaded shape and dtype against `RepresentationRef`.

There is no database, cloud store, garbage collector, automatic default
persistence, or dynamic plugin system. Phase 6A.8 adds a local trusted JSON
catalog over verified opt-in persistence receipts; it reuses this resolver and
does not become another materializer.

Phase 6A.7 adds an opt-in live producer for these same contracts. See
[`LIVE_REPRESENTATION_PERSISTENCE.md`](LIVE_REPRESENTATION_PERSISTENCE.md).
Default analysis still creates no representation artifacts.

Phase 6A.8 adds discovery and lifecycle state for those verified artifacts. See
[`RECEIPT_CATALOG_ARCHITECTURE.md`](RECEIPT_CATALOG_ARCHITECTURE.md).

## 10. Current Pipeline 3 representations

This inventory comes from `analysis_engine`, `CROMAAdapter`, Phase 2/3.5 code,
the real final-audit receipt, and committed spatial-evidence artifacts—not names
alone.

| Representation/artifact | Producer | Actual shape/type | Spatial/semantic meaning | Current lifecycle and validation |
|---|---|---|---|---|
| Aligned raw optical | raster assembly | float32 `[12,120,120]` for `61_39` | pixel grid, Sentinel-2 bands | ephemeral live; verified final-audit `.npy` with SHA-256 |
| Aligned raw SAR | raster assembly | float32 `[2,120,120]` | pixel grid, VV/VH | ephemeral live; verified final-audit `.npy` with SHA-256 |
| Physical features | `modality_features.extract` | float32 `[62]` | scene feature statistics, not a map | values in live report; verified `.npy` |
| Optical CROMA tokens | official CROMA adapter | float32 `[225,768]` | latent 15×15 token grid | shape only live; verified final-audit `.npy` |
| SAR CROMA tokens | official CROMA adapter | float32 `[225,768]` | latent 15×15 token grid | shape only live; verified final-audit `.npy` |
| Joint CROMA tokens | official CROMA adapter | float32 `[225,768]` | latent joint token grid, no semantic claim | shape only live; verified final-audit `.npy` |
| Optical/SAR/joint GAP | official CROMA adapter | three float32 `[768]` vectors | scene-level latent representations | concatenated values live; individual verified `.npy` files |
| Pooled CROMA | Phase 2 / `_deep_features` | float32 `[2304]` | concatenated scene representation | values live; verified `.npy` |
| Hybrid representation | existing seeded `HybridFusion` | float32 `[192]` | scene-level untrained representation | values live; verified `.npy`; not a prediction |
| Pixel feature tokens | pixel/evidence pipeline | 225 JSON token records | deterministic token-local measurements | committed `tokens.json`; evidence validation |
| Regions | evidence pipeline | 17 JSON regions for `61_39` | deterministic connected evidence structure | live and committed; not segmentation |
| Evidence/claims | evidence pipeline | 3 claims for `61_39` | deterministic evidence only | live and committed; canonical evidence hash |
| Metadata/provenance | cube, orchestration, receipts | JSON | scene/grid/input/model/execution identity | live reports and receipts |
| Analysis report | `save_report` | schema 2.0 JSON | complete legacy response | persisted per UUID when requested; no current receipt checksum |

Raw arrays and learned tensors are deterministic for the pinned runtime, while
CROMA representations are learned latent features. Hybrid is deterministically
seeded but untrained. Evidence and interpretation are deterministic rules.

## 11. Currently addressable representations

The real ignored final-audit bundle is locally addressable through its receipt:
11 NumPy references covering raw optical/SAR, physical, pooled CROMA, hybrid,
three CROMA token arrays, and three GAP vectors. Every reference was materialized
with checksum, shape, and dtype verification. This validates the reference and
resolver design for `61_39`; it does not make that ignored audit directory a
production-wide artifact store.

Committed spatial evidence, region, token, interpretation, and provenance JSON
files can be assigned logical references with real byte checksums through the
file boundary. They are not silently coupled to machine-specific paths already
embedded in historical provenance.

## 12. In-memory-only or non-addressable representations

In the normal live `run_analysis` path, aligned raw arrays and the six individual
CROMA outputs are not retained as independently loadable values. The response
keeps only their shapes plus the pooled values. Their live references therefore
say `not_addressable`. The live physical, pooled CROMA, hybrid, evidence, region,
and metadata objects are `materialized` for that bundle but have no artifact URI
unless a caller explicitly persists and receipts them.

## 13. Future VQA and captioning requirements

VQA or captioning could reference scene-level CROMA, physical features, metadata,
and evidence, but neither capability exists. They still require a validated
question/text conditioning model, task datasets and splits, evaluation, output
contracts, epistemic behavior, and a production persistence policy for token
features. References alone provide no answering or captioning semantics.

## 14. Future grounding requirements

Grounding can discover verified optical/SAR/joint token arrays and the existing
token-to-pixel/region mapping. It still requires a validated grounding method and
task evidence linking language or predictions to tokens. CROMA token position is
not learned grounding, and deterministic evidence regions are not segmentation.

## 15. Future optical-SAR and temporal requirements

Optical-SAR reasoning can select modality-specific scene or token references but
still needs a validated reasoning head or rule contract; the current hybrid is
not a trained task output. Temporal analysis additionally needs two independently
identified scene bundles, acquisition times, co-registration diagnostics, a
pairing/transformation provenance record, temporal representations, and evaluated
change logic. This phase supplies none of those algorithms.

## 16. Known limitations

- Live analysis does not automatically persist or receipt representation arrays.
- The verified final-audit arrays are ignored validation artifacts, not guaranteed
  production storage.
- No central artifact catalog or lifecycle management exists by design.
- Historical evidence provenance contains absolute paths; new architectural
  identities do not repeat them, but historical files are unchanged.
- Report persistence has immutable UUID filenames but no content receipt/checksum.
- Token mapping is validated for the current 120×120/15×15 layout, not every
  future resolution or model.
- References do not add VQA, captioning, grounding, temporal analysis, prediction,
  or confidence.
