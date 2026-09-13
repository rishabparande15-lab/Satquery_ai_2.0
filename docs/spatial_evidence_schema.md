# Structured spatial evidence schema

`src.evidence_schema` is the authoritative Pass 5C schema and query API. Schema version is `spatial_evidence_v1`; mapping version is `north_up_row_major_120_to_15_v1`.

The root contains `scene`, 225 ordered `tokens`, deterministic `regions`, supported `claims`, modality-specific `sensor_views`, and `provenance`. A token contains its index, `[row,column]`, half-open pixel bounds, projected map bounds, block size, valid ratio, selected physical/local features, CROMA references, and predictions. Each feature records value, unit, sensor, `DERIVED` type, aggregation, valid count/ratio, feature version, and source artifact.

CROMA entries reference the actual `[225,768]` arrays by path, hash and row indexing. Their status is `LATENT`; dimensions are never assigned semantic labels. Predictions accept explicit per-token records, but sample `61_39` reports `UNAVAILABLE` because no trained task-head output exists for it. Reference-map labels are not relabelled as predictions.

Evidence categories are `MEASURED`, `DERIVED`, `MODEL_PREDICTED`, and `INFERRED`. Current raw inputs are upstream measurements, token features are derived, absent task outputs are unavailable, and rule-supported statements are inferred. Evidence strength (`STRONG`, `MODERATE`, `WEAK`, `UNAVAILABLE`) is descriptive and explicitly not calibrated confidence.

`EvidenceQueries` exposes `get_scene_summary`, `get_token_evidence`, `get_feature_map_evidence`, `get_class_regions`, `get_sensor_evidence`, and `get_supported_claims`. Missing features/sensors return `UNAVAILABLE`; they are never replaced with zeros.

The JSON artifacts are intentionally self-contained enough for an LLM or future VQA controller to query without raw pixels while retaining links and hashes needed for independent reconstruction.
