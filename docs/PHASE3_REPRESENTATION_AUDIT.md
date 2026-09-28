# Phase 3 representation strategy and VLM input architecture

**Decision (2026-09-19): REPRESENTATION FOUNDATION: READY.** This is a contract and persistence decision for future adapters. No VLM, grounding model, temporal model, new feature store, or scientific pipeline change is part of this decision.

## Scope and evidence

Inspected `src/representation_materialization.py`, `representation_catalog.py`, `receipt_catalog.py`, `region_contract.py`, `representation_linking.py`, `image_language_foundation.py`, `architecture_contracts.py`, `live_representation_persistence.py`, `croma_adapter.py`, `phase1_foundation.py`, and `scripts/run_pipeline3_5000_baseline.py`; Phase 2B–2E documentation and tests; the frozen 5,000-area dataset/cache manifests, 157 shard receipts, and 15,000 typed receipts. The BigEarthNet.txt coordinate audit remains authoritative: its source geometries have `UNKNOWN` semantics and no analysis-pixel or CROMA-token mapping.

The frozen population is 5,000 single-time-area records (4,600/200/200). Dataset fingerprint: `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`; split fingerprint: `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`. CROMA base revision: `59505a6bcadbf36ba20767270154bf9f3067c5e7`; checkpoint SHA-256: `0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63`. The Phase 2D.2 materialization manifest records 157 source/hybrid shards, 15,000 typed receipts, and receipt-catalog SHA-256 `3deb01c15ecfa76f99098d54333bff0d0ce831b0d3af536d6b85cf9aea3deece`.

## Current inventory and role decision

Roles describe **uses**, not separate tensor formats. A representation may serve several roles, but its artifact identity and spatial level remain fixed. `canonical scientific` means frozen inputs used by the evaluated Pipeline 3 predictor. `adapter-eligible scene context` means a verified scene vector that an adapter may select, without claiming VLM suitability for an unselected model. `conditional pixel/token input` means a supported type that is produced or persisted only for a concrete consumer. `reserved temporal` means a contract requirement with no present artifact.

| Representation | Current shape/dtype and source | Information retained | Current persistence/consumer | Role and future VLM decision |
|---|---|---|---|---|
| `physical_62d` | `[62] float32`; deterministic local optical/SAR physical/statistical extraction | Joint modality provenance, scene summary; no addressable pixel/region grid or time sequence | Verified in source NPZ and typed receipts; Pipeline 3 hybrid; optional Phase 2E context | **Canonical scientific component; supplementary adapter-eligible scene context.** Keep model-independent feature identity. |
| `joint_croma_gap_768d` | `[768] float32`; frozen CROMA base on 12 optical + 2 SAR channels, followed by joint GAP | Fused optical/SAR scene embedding; spatial token positions are pooled away; individual sensor contributions are not recoverable from the vector | Verified in source NPZ and typed receipts; Pipeline 3 hybrid; Phase 2E `primary_visual` reference | **Canonical scientific component and currently available scene context.** An adapter must still declare compatibility; it is not a universal VLM input. |
| `hybrid_830d` | `[830] float32`; exact concatenation `physical[0:62] + joint_croma_gap[62:830]` | Two scene summaries with explicit segment provenance, but no pixel or token layout | Verified compressed hybrid NPZ and receipts; frozen 19-class scientific predictor | **Canonical predictor input only.** Available to an adapter that explicitly supports it, never the default VLM tensor by implication. |
| Optical/SAR GAP vectors | Each `[768] float32` from separate frozen CROMA branches | Sensor-specific scene identity; no spatial grid | Arrays already reside in the historical source cache; no separate 5,000-area typed catalog rows | **Conditional supplementary scene context.** Admit through verified receipts only if a consumer needs separate sensors; do not duplicate now. |
| Optical/SAR/joint CROMA encodings | Each `[225,768] float32`, 15×15 row-major | Spatial **feature** grid with 8×8 analysis-cell anchors and global attention context; not pixel measurements or grounding predictions | Supported by CROMA adapter and opt-in live persistence; 5,000-area catalog rows currently `MISSING` | **Conditional token input.** Generate lazily for an adapter needing tokens or an independently justified spatial task. |
| Raw optical/SAR | Source GeoTIFF bands; canonical in-memory arrays `[12,120,120]` and `[2,120,120]` float32 after validated loading | Pixel grid, bands, CRS/transform when validated, separate modalities; no language semantics | Existing source imagery/manifest, not duplicated into 5,000-area representation catalog; opt-in live persistence supports raw arrays | **Conditional raw input.** Load/validate on demand for an adapter requiring imagery; its own preprocessing is versioned. |
| Pixel/token physical features, pooled CROMA, regions/evidence, older `HybridFusion` | Types and live paths exist; shapes depend on producer | Producer-specific; do not equate older 192-D fusion or pooled 2,304-D features with frozen `hybrid_830d` | Optional catalog families truthfully `MISSING` for this population; live evidence may exist for individual scenes | **Conditional specialized inputs/evidence.** Do not promote to canonical 5,000-area VLM input. |
| Temporal representation | None in the 5,000-area catalog | No T1/T2 identities, alignment, difference, or temporal supervision | No persisted temporal representation | **Reserved.** No current temporal claim. |

The catalog's 15 representation families yield 75,000 rows: 15,000 `VERIFIED` core rows and 60,000 truthful `MISSING` optional rows (Phase 2D.2). Catalog type vocabulary and live persistence descriptors are broader than the three persisted core representations; a supported enum or `MISSING` row is not a materialized sample. `RepresentationSet` has optional `spatial_tokens`, and `SpatialReference` carries levels and optional token grid/CRS, but neither creates a spatial token artifact by itself.

All three core vectors are deterministic **for fixed inputs, code, configuration, and checkpoint**. Physical features are not model-dependent; CROMA GAP and the hybrid's GAP segment are checkpoint-dependent. The hybrid is frozen as an evaluated artifact. All three are scene-level and single-time; `OPTICAL_SAR` identifies input modalities, not a guarantee that a fused vector can be separated into optical and SAR components.

## Spatial distinctions

1. **Image information:** raw rasters carry pixel values and, when verified, a geospatial raster transform. They do not by themselves locate a phrase or class instance.
2. **Spatial representation:** CROMA's 225 vectors retain ordered token positions on the validated 15×15 analysis grid. Self-attention gives each token wider context, so a token is not a pure measurement of its nominal 8×8 cell. Optical, SAR, and joint token arrays are distinct.
3. **Deterministic source geometry:** independently verified `PIXEL`/`ANALYSIS_GRID` geometry can intersect the token grid under `spatial_contract.py`. BigEarthNet.txt geometry is `source_unit_square_unmapped`, so `representation_linking.py` returns `SPATIAL_LINK_UNAVAILABLE`; `image_language_foundation.py` leaves `analysis_geometry` and `token_coordinates` null.
4. **Learned grounding:** requires a model and evaluated image/text-to-region behavior. None is present or implied by raw pixels, CROMA tokens, or source box strings.

`physical_62d`, all GAP vectors, and `hybrid_830d` lose addressable within-scene location. No scene vector should be advertised as spatial grounding input without a model-specific reason and evaluation.

## Optical/SAR interface

`CROMAAdapter` runs the frozen base model with both sensors and can produce independent optical/SAR encodings and GAPs or a joint result. The verified `joint_croma_gap_768d` has `OPTICAL_SAR` provenance, but its numeric content cannot reconstruct either independent sensor embedding or the raw bands. Using only joint GAP removes token layout, spectral detail, SAR backscatter structure, and explicit sensor separation. Separate GAPs can expose sensor-specific scene summaries; tokens expose ordered features; raw bands preserve the most input detail. This is an information hierarchy, not a measured ranking of future VLM quality.

No single pre-VLM fusion policy is mandated. An adapter that accepts raw optical and SAR should receive separately identified verified inputs and perform its declared preprocessing/fusion. An adapter trained for frozen CROMA joint embeddings may select `joint_croma_gap_768d`. An adapter requiring token structure may request one or more CROMA token variants. The controller chooses by declared task/capabilities and verified availability; model-specific tensor packing belongs in `VisionInputAdapter.prepare_inputs`, not `SceneBundle` or the catalog.

## Future temporal contract

Two scene embeddings can support a coarse comparison only after the two observations have authoritative identities, compatible provenance, acquisition times, and alignment; concatenation does not confer temporal understanding. Paired token grids require verified common spatial support and temporal order. Explicit difference/relationship features would be derived, model- and alignment-dependent representations, not current core features.

A future temporal input should bind `(scene_id, t1_observation_id, t2_observation_id, ordered timestamps, per-time RepresentationRef(s), modality availability, common CRS/grid or alignment transform and validation status, preprocessing/model versions, split/group identity, derivation provenance)`. A difference artifact additionally needs its own producer/version/receipt and source-reference hashes. `SceneBundle.temporal_context`, `TaskRequest.temporal_context`, temporal capability flags, and the reserved image-language temporal fields are hooks, not validated temporal data. BigEarthNet.txt provides no authoritative T1/T2 grouping for this purpose.

## Model-independent vision input boundary

The existing `RepresentationInput` already supplies typed reference ID, representation type/variant, modality, shape/dtype, logical URI, artifact/sample checksums, producer/model/preprocessing versions, spatial level, and `VERIFIED` status. `VisionInputContract` groups available modalities and representations with spatial resolution, dimensions, provenance, and preprocessing. `VisionInputAdapter` performs `validate_inputs`, `prepare_inputs`, and `contract`; `VLMAdapter` declares capabilities. No model-specific tensor shape or tokenization is in the core contract.

For a future adapter, selection must require exact image/split identity, receipt-verified artifact, accepted representation type/variant, accepted modality combination, compatible spatial level/resolution, known preprocessing/model version, and task support. For temporal tasks it must additionally require the paired-observation contract above. Missing raw/token/temporal material is an explicit unavailable result until generated and verified. `input_profile.primary_visual = joint_croma_gap_768d` is the present **available scene-context preference**, not proof that any future VLM can consume it. `hybrid_is_default_vlm_input = False` remains correct. The current contracts are sufficient for this architecture decision; extend fields only when a selected adapter exposes a concrete missing requirement.

## Storage, generation and machine limits

Sizes below are **float32 payload arithmetic**, unless explicitly labeled measured compressed storage. They exclude NPZ/NPY headers, metadata, receipts, and raster source encoding. The current machine has 8 GB VRAM and 24 GB RAM. Historical Pipeline 3 physical+CROMA extraction took **910.732 s total** with **904,975,360 B peak GPU allocation**; the run did not separate physical, GAP, and token generation time. These are historical measurements, not a forecast for a future VLM or a full-token store. Phase 2D.2 measured 27.143 s on first verification/materialization and 23.075 s on full reuse; its host peak RAM was not instrumented.

| Representation | Size/sample | 5,000-area size | Generate time | VRAM implication | Persist? |
|---|---:|---:|---|---|---|
| `physical_62d` | 248 B | 1,240,000 B payload | Included in 910.732 s combined historical run; separate time unknown | CPU feature path; no new GPU inference to reuse | **Already persisted** in shared source shards. |
| `joint_croma_gap_768d` | 3,072 B | 15,360,000 B payload | Included in combined historical run; separate time unknown | Historical combined run peaked at 904,975,360 B GPU allocation | **Already persisted** in shared source shards. |
| `hybrid_830d` | 3,320 B | 16,600,000 B payload; **15,671,729 B measured compressed** | Phase 2D.2 first-run hybrid assembly 2.709 s; full verification 27.143 s | No CROMA/GPU rerun for concatenation | **Already persisted** for frozen predictor. |
| One CROMA GAP, optical or SAR | 3,072 B | 15,360,000 B payload per modality | No separate measurement; historical cache already contains arrays | No new inference to read existing arrays | No new typed copies until selected. |
| One CROMA token family | 691,200 B | 3,456,000,000 B payload (~3.22 GiB) | Full-token generation/storage time unknown | 8 GB VRAM impact unknown; bounded batches required | **Lazy/conditional**, per required family. Three families would be 10,368,000,000 B payload (~9.66 GiB). |
| Canonical in-memory raw optical | 691,200 B (`12×120×120`) | 3,456,000,000 B payload if duplicated | Load/reprojection time for whole set unknown | VLM VRAM unknown | **Reference existing rasters; load on demand.** |
| Canonical in-memory raw SAR | 115,200 B (`2×120×120`) | 576,000,000 B payload if duplicated | Whole-set time unknown | VLM VRAM unknown | **Reference existing rasters; load on demand.** |
| Temporal pair / derived difference | At least two source representations plus metadata; exact size model-dependent | Unknown | Unknown | Unknown | **Do not generate now.** |

The **measured** source cache is 44,280,954 B for all 157 compressed NPZ shards together; this includes physical, optical/SAR/joint GAP, targets, and identity data, so it cannot be assigned wholly to one representation. Its two typed core arrays are referenced in place. The current hybrid store adds 15,671,729 B compressed. A 5,000-area token store would be multi-GB before metadata and copies; simultaneous all-family in-memory loading would exceed the stated 8 GB VRAM even by raw payload arithmetic. No VLM generation latency, inference VRAM, or token-cache compression ratio has been measured.

## Persistence schedule and architecture

**Generate now:** nothing. Keep validating/reusing `physical_62d`, `joint_croma_gap_768d`, and `hybrid_830d` as the canonical frozen scientific core. Preserve the optional catalog rows as `MISSING` until a verified artifact exists.

**Only after a VLM is selected:** materialize exactly the raw sensor inputs, sensor-specific GAPs, or token families required by its declared input contract, first on a bounded evaluation subset. Use existing opt-in receipt-backed live persistence and catalog admission; decide any 5,000-area extension from measured reuse, generation time, RAM/VRAM, and evaluation value. Do not copy raw rasters merely to give them a representation label.

**Only for a spatial task:** require a verified geometry frame and evaluated model-specific need before caching token inputs or producing learned regions. Do not treat BigEarthNet.txt boxes/points as validated pixel/token supervision.

**Only for a temporal task:** establish T1/T2 identities, alignment, splits and provenance before creating paired or difference representations.

Current flow: validated raw sources and `SceneBundle` → receipt-backed representation catalog/`RepresentationSet` → `ImageLanguageSample` plus `VisionInputContract` → adapter-specific validation/preparation (future) → typed results/evidence. The catalog reports *availability*; the adapter selects among verified options. Raw and token paths remain opt-in. No new central registry, enum value, or persistence default is required.

## Rejected alternatives

- Make `hybrid_830d` the mandatory VLM input: it is the evaluated coverage-predictor feature order, and a future VLM's visual interface is unknown.
- Persist all three 225×768 token families for all 5,000 areas now: at least 10.368 GB raw payload without an established consumer or measured benefit.
- Duplicate all canonical raw imagery as float32 arrays: existing rasters and manifest already supply validated source identities; duplicate storage would reduce model flexibility rather than establish it.
- Infer grounding from CROMA tokens or BigEarthNet.txt numeric geometry: spatial features and unverified source coordinates provide no learned grounding evaluation or safe annotation transform.
- Treat two independently produced vectors as temporal evidence: ordered observations and alignment are not present.
- Add a new `VisionInput` class or VLM-specific fields now: the Phase 2E input contract and adapter boundary already express verified type, modality, spatial level, provenance, and model-specific preparation. A selected adapter may justify a narrow additive change later.

## Implementation and verification

This audit adds documentation only. No production module, test, cache, catalog, receipt, predictor, or split was modified. Focused tests passed (`69 passed`); the full pytest suite passed (`355 passed, 5 skipped`). `git diff --check` passed. A read-only `load_verified_receipts` validated all 15,000 receipts and the source-cache manifest hash matched its materialization record. A read-only catalog rebuild yielded 75,000 rows (`15,000 VERIFIED`, `60,000 MISSING`), all 5,000 images with complete core representations, and the established catalog fingerprint `0f5ce8feceff2fe8f2a75e1570b36edabf8a4716c57159243ffc6063eafcfd30`. Dataset, split, receipt, catalog, and scientific fingerprints are unchanged. The sealed historical checkpoint artifact retains MAE `4.1034286464812215` pp, RMSE `9.5873311029768` pp, and accuracy `0.65` on 200 test areas; no inference or retraining was run here.
