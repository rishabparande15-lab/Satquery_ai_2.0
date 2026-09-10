# Phase 2 integration contract

## Boundary audit

Phase 1 exposes `PreparedBatch`: canonical optical tensors `[N,12,120,120]`,
canonical SAR tensors `[N,2,120,120]`, ordered `SampleIdentity` values, and a
named normalization report. `FrozenCromaFoundation` validates six frozen
CROMA outputs: spatial optical, SAR, and joint encodings `[N,225,768]`, plus
pooled optical, SAR, and joint representations `[N,768]`. The 225 tokens are
row-major `15 x 15` tokens.

The existing advanced proof-of-concept pipeline extracts one 62-value physical
feature vector with `LocalRasterFeatureProvider`, obtains the three pooled
CROMA vectors, concatenates them in the existing order
`optical_GAP, SAR_GAP, joint_GAP` to `[N,2304]`, and passes physical and CROMA
vectors unchanged to `HybridFusion(62,2304)`, producing the existing 192-value
representation. The fusion module is an untrained representation prototype,
not a task model.

## Required transformation

`src.phase2_integration.adapt_phase1_output` is the typed bridge. It validates
the Phase 1 batch, validates all six CROMA tensors, validates physical feature
names and dimensions, and returns `Phase2PipelineInput`. The advanced pipeline
consumes its `physical_features` and `pooled_croma_features` fields. Spatial
features remain available as separate fields and are never substituted for
pooled features.

The adapter joins by the ordered `SampleIdentity.patch_id` values supplied by
the caller, never by directory order. It rejects duplicate or mismatched
identity lists, inconsistent feature rows, non-finite values, invalid band
orders, and mixed split batches. Optional physical features are represented by
an explicit missing list; required physical vectors still fail when absent.

## Preserved identity and provenance

Each output row retains patch ID, Sentinel-1 and Sentinel-2 IDs, split,
acquisition metadata, CRS, bounds, resolution, canonical channel order,
normalization profile, CROMA feature dimensions and checkpoint provenance,
physical feature names/schema, experiment configuration, and optional source
hashes. The adapter does not infer a split. The current three-sample proof of
concept supplies an explicit `--split` because its source manifest has no
official split assignment; this limitation is recorded rather than hidden.

## Spatial and physical requirements

The adapter enforces `[N,225,768]` for all spatial CROMA representations and
`[N,768]` for pooled representations. It preserves the official row-major
token convention and records it in the contract. Physical feature ordering is
the existing provider's reported ordering, including the legacy joint 62-value
schema and its documented SAR relationship limitations. Missing optional
features are reported; values are never fabricated.

## Tests

Phase 2 tests cover optical, SAR, joint, and pooled shape contracts; physical
feature ordering; identity and spatial mismatches; split isolation; provenance;
missing optional features; invalid values; deterministic use of the existing
fusion module; and a real-sample integration path when the external fixture is
available. Existing Phase 1 and advanced-pipeline tests remain unchanged.

## Scientific scope

This phase does not alter CROMA, `HybridFusion`, feature formulas, checkpoints,
prediction heads, evaluation results, or the reference baseline. GEE, temporal
analysis, retrieval, specialist models, and UI work remain out of scope.