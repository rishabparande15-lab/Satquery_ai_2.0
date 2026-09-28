# Phase 3K.2 — Grounding Architecture and Fail-Closed Contract

## 1. Objective

**IMPLEMENTED:** a model-neutral, fail-closed validation boundary for future remote-sensing visual grounding. It contains no model, training, label creation, source-coordinate conversion, or external dataset integration.

## 2. Current grounding blockers

Phase 3K.1 remains **GROUNDING_TRAINING_BLOCKED**: no audited source cleared authoritative geometry semantics, licensing, immutable pinning, complete image–annotation linkage, provenance, and leakage gates. BigEarthNet.txt geometry remains source-preserved and mapping-unverified.

## 3. Grounding request

`GroundingRequest` requires request ID, image ID, query/referring text, modality, requested target type, declared `CoordinateSpace`, provenance, and optional authoritative image dimensions, annotation ID, and metadata. Target types are `POINT`, `BBOX`, `MASK`, `POLYGON`, and `REGION`; none implies training.

## 4. Grounding result

`GroundingResult` retains its request, grounding status, geometry origin, optional learned geometry, mapping and validation status, confidence/source, evidence reference, and model provenance. It cannot recast source geometry as a model prediction. A verified result requires valid geometry and verified mapping; the contract-only path does not emit either.

## 5. Geometry types

The contract supports existing typed `Point` and `BBox`, a typed `Polygon`, and `MaskReference`. A mask is a reference plus dimensions, coordinate space, provenance, and optional checksum—not mask bytes or a synthetic mask.

## 6. Coordinate spaces

The contract reuses `PIXEL`, `ANALYSIS_GRID`, `CROMA_TOKEN`, `NORMALIZED_IMAGE`, and `GEO` from `spatial_contract.py`. Geometry without a declared recognized space is rejected. No new coordinate space or implicit conversion exists.

## 7. Geometry validation

Points and boxes reuse existing finite/order/extent/bounds validation; polygons require at least three finite, non-zero-area vertices; mask dimensions must agree with declared image dimensions when both are supplied. Invalid geometry is retained with `MALFORMED`, `DEGENERATE`, or `OUT_OF_BOUNDS` status for audit and is never clipped or repaired. The adapter rejects invalid output.

## 8. Source geometry versus learned grounding

`GeometryOrigin.SOURCE_GEOMETRY` and `LEARNED_PREDICTION` are separate. The existing `region_contract.py` remains the source/deterministic-region store; `GroundingResult` permits output geometry only as a learned prediction and does not claim it is verified evidence.

## 9. Mapping contract

`MAPPING_UNVERIFIED` is the default. It cannot be emitted by the contract-only adapter as grounded evidence. Coordinate conversion requires independently verified source semantics and dimensions; none is performed here.

## 10. CROMA token boundary

CROMA's 15×15 / 225-token grid is unchanged. The contract creates no token IDs and does not activate BigEarthNet.txt mapping. Spatial features and token correspondence are not learned grounding.

## 11. Evidence model

The explicit non-interchangeable evidence kinds are `SOURCE_GEOMETRY`, `MODEL_GROUNDING`, `PIXEL_EVIDENCE`, `REGION_EVIDENCE`, `TOKEN_EVIDENCE`, and `METADATA_EVIDENCE`. Generated text or model geometry does not automatically become pixel, region, or token evidence.

## 12. Confidence model

Confidence source is `MODEL_PROVIDED`, `CALIBRATED`, or `UNKNOWN`. Unknown source requires `UNKNOWN` confidence. A numeric value must be finite in `[0,1]` and have a non-unknown source. This phase calibrates nothing.

## 13. Provenance

Request/result metadata carries source dataset/revision through request provenance, image and optional annotation ID, coordinate space/dimensions, model/adapter/preprocessing information through model provenance, and optional evidence reference. Missing facts remain `UNKNOWN` rather than inferred.

## 14. Model-swappable adapter

`GroundingAdapter` defines request/output validation, preparation, prediction, capabilities, provenance, and resource estimation. `ContractOnlyGroundingAdapter` exposes the interface but raises `GroundingModelUnavailable` for prediction. It has no dependency on Qwen, GeoChat, EarthDial, EarthGPT, VRSBench, or a specific model.

## 15. S2, SAR, and temporal compatibility

Capabilities are conservative: optical, multispectral, and SAR are `ADAPTER_REQUIRED`; optical+SAR and temporal are `NOT_IMPLEMENTED`. Qwen grounding remains `NOT_VERIFIED`. Existing temporal contracts are unchanged; T1/T2/changed-region grounding is a placeholder only.

## 16. Resource contract

The contract reports one image and `UNKNOWN` for estimated VRAM, region count, and mask cost. It does not invent resource measurements.

## 17. Fail-closed rules

The implementation rejects undeclared spaces, malformed request metadata, source geometry presented as a learned result, incompatible target/geometry types, numeric confidence with unknown source, verified status without valid geometry/mapping, and adapter emission of invalid or mapping-unverified geometry. It never clips geometry or maps unresolved source coordinates.

## 18. Structural tests

`tests/test_grounding_contracts.py` covers request/provenance/space checks, point/bbox/polygon/mask validation, retained invalid geometry without clipping, source-vs-learned separation, Phase 2F/CROMA mapping protection, confidence, resource fields, modality declarations, and the model-swappable unavailable adapter. Synthetic geometry claims no grounding performance.

## 19. Implemented versus deferred

**Implemented:** request/result contracts, geometry validation, coordinate declarations, provenance/evidence/confidence semantics, resource surface, adapter interface, and fail-closed protection.

**Deferred:** learned grounding, VQA integration, training/evaluation, S2/SAR/optical-SAR/temporal grounding, and CROMA grounding from unresolved BigEarthNet.txt geometry.

## 20. Phase 3K.3 gate

Phase 3K.3 is **GROUNDING DATASET RE-EVALUATION / ACQUISITION GATE**. It may proceed only when one source clears established geometry, licensing, provenance, linkage, split, leakage, and bounded-acquisition requirements.

## Scientific immutability

No CROMA, scientific predictor/checkpoint, `physical_62d`, `joint_croma_gap_768d`, `hybrid_830d`, split, representation artifact, receipt, Phase 2F, S2 adapter/checkpoint, SAR projector, or temporal artifact was modified. The frozen baseline remains **65.0% accuracy**, **4.1034286734 pp MAE**, and **9.5873312123 pp RMSE**.
