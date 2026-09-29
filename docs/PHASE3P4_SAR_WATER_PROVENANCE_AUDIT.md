# Phase 3P.4: independent water-provenance audit

Status: `PHASE3P4_BLOCKED_NO_INDEPENDENT_REFERENCE`.

This fail-closed, data/provenance decision gate loaded the exact 100 Phase 3P.3
identities (50 water-present and 50 water-absent). It performed no training, no
model inference, no parameter update, no routing change, and no TEST access.

## Preregistered audit rule

An acceptable validator had to be an independently sourced, patch-aligned water
reference with recorded provenance, CRS, bounds, valid overlap, and resolution.
The fixed thresholds were water present at fraction >= 0.05, water absent at
<= 0.01, and ambiguous in between. Without a qualifying reference, the
deterministic decision is `UNVERIFIABLE`.

The pool could pass only with at least 60 confirmed records, at least 20
confirmed TRAIN records per class, five confirmed VALIDATION records per class,
agreement >= 0.95, verified alignment, provenance completeness, valid
opposite-label shuffle pairs, and zero TEST access.

## Result

No independent patch-level water reference was available locally. The existing
BigEarthNet `Reference_Maps` were rejected because they belong to the same
CLC-derived target/reference lineage that generated the Phase 3P.3 candidate
labels; using them would be circular validation.

All 100 records are therefore `UNVERIFIABLE`: confirmed water 0, confirmed
non-water 0, ambiguous 0. Original-label agreement is not computable. Spatial
alignment was not performed because there was no independent geospatial layer
to align. No confirmed TRAIN or VALIDATION manifests were created, and no
shuffle pair remains valid for a future training manifest.

## Decision

Classification: `SAR_WATER_POOL_NEEDS_MORE_PROVENANCE`.

Bounded standalone SAR retraining is not justified. The exact next phase is to
obtain a separately sourced, patch-aligned water reference or a documented
independent human review with supporting reference material for every retained
candidate, then rerun this immutable Phase 3P.4 protocol. SAR routing and
S1+S2 fusion remain blocked.
