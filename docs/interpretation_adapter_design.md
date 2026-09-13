# Pass 5E Interpretation Adapter Design

`src/interpretation_adapter.py` is a fail-closed layer after `src/evidence_schema.py`. The flow is:

`validated inputs → existing features/CROMA → spatial_evidence_v1 → exact claim validation → controlled routing/templates → answer + traceability`

The router normalizes a question, rejects forbidden and unrouted forms, and selects claim IDs. Claim acceptance then independently checks the schema contract. Templates receive only the accepted claim, referenced regions, selected mode, and whether a location was requested. Scene-relative upper/lower/left/right positions are calculated from row/column coordinates already stored in region token extents, while projected extents are preserved in the trace. The adapter does not promote image-row direction to a geographic compass direction. Missing geometry suppresses the location clause.

Simple mode returns normal sentences without raw numeric dumps. Technical mode adds feature IDs, token indices, region IDs, and the explicit uncalibrated limitation. Each structured claim carries its text, epistemic status, strength, confidence status, sensor, source features, tokens, regions, derived scene positions, and source artifacts. Evidence provenance is nested unchanged.

The semantic SHA-256 covers the complete deterministic result except its own hash. Runtime measurements are deliberately outside that semantic object. `validate_interpretation` verifies status, claim inclusion in the answer, allowlisting, evidence provenance, epistemic status, and hash integrity.

The API adds `interpretation` and compact `spatial_evidence` fields alongside existing results. Existing prediction fields and the historical deterministic explanation are not rewritten. The browser displays summary, evidence basis, sensors, region/token references, the technical answer, and provenance using text-only DOM operations.

Limitations: this is not an LLM, VLM, VQA system, object detector, classifier, learned grounding model, segmentation model, or calibrated uncertainty estimator. The region layer remains deterministic threshold grouping.
