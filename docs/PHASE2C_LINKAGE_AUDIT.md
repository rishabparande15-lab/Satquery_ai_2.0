# Phase 2C Linkage Audit

## Verdict

Phase 2C is complete for the annotation-to-representation linking foundation. The layer is split-safe, provenance-aware, deterministic, and explicitly separates geometric correspondence from semantic interpretation or learned grounding.

## Existing abstractions reused

The implementation reuses Phase 2A annotation records, Phase 2B coordinate spaces and token mapping, `RepresentationRef`/`ArtifactRef`, receipt-compatible logical artifact identity, and existing spatial evidence conventions. Pipeline 3 and the existing evidence chain were not modified.

## New components

- [src/representation_linking.py](../src/representation_linking.py)
- [tests/test_representation_linking.py](../tests/test_representation_linking.py)
- `AnnotationRepresentationLink` schema and explicit link/status enums
- deterministic bbox/point-to-token correspondence
- representation availability and provenance checks
- deterministic `links.jsonl` artifact writer
- optional temporal identity fields reserved but unpopulated

## Real-data statistics

The real Phase 2A artifact contains:

- Total annotations: `101,571`
- Canonical image IDs: `4,771`
- Image-level linkable annotations: `101,571`
- Binary QA: `37,518`
- Multiple-choice QA: `33,979`
- Captions: `4,771`
- Text boxes: `12,179`
- Point boxes: `13,124`

The source spatial values remain `source_unit_square_unmapped`; therefore real source box and point records are not converted to analysis pixels or tokens.

The full deterministic artifact was generated at the ignored runtime location `artifacts/representation_links/bigearthnet_txt/`:

- Link records: `101,571`
- Image-level links: `101,571`
- Bbox-to-token links: `0`
- Point-to-token links: `0`
- Missing representation cases: `101,571`
- Spatial link failures: `25,303`
- Split failures: `0`
- Provenance failures: `0`
- `links.jsonl` SHA-256: `46e037b72c2bbc6de02c2815f708ca427306b6b0f672b7c208d2c133a81a9d89`

All representation cases are explicitly missing because the full annotation inventory was not paired with a complete per-image representation catalog. This is a truthful availability result, not fabricated linkage.

A representative five-record real-data link artifact covered all observed task families:

- Image-level links: `5`
- Bbox-to-token links: `0`
- Point-to-token links: `0`
- Missing representation cases: `4`
- Spatial link failures: `2`
- Split failures: `0`
- Provenance failures: `0`
- Deterministic `links.jsonl` SHA-256: `10ea3889b9c122a7bdb1b3711d25dc199dc1ca73dd922ba15d7c85d87b43dc20`

The two spatial failures are expected and correct: Phase 2A does not authorize conversion of source unit-square geometry.

## Spatial-linking results

Fixture and canonical-geometry tests verify:

- `[4,4,12,12]` maps to tokens `0,1,15,16`.
- Intersection fractions are recorded for token and bbox coverage.
- `FULLY_CONTAINED`, `INTERSECTS`, and `CONTAINS` relationships remain distinct.
- Boundary points map deterministically without clipping.
- Out-of-bounds geometry becomes `SPATIAL_LINK_UNAVAILABLE` and an invalid link.
- Text-only annotations never receive invented spatial references.

## Split safety and provenance

Adversarial tests verify:

- annotation/manifest split mismatch fails closed
- missing split metadata produces `SPLIT_UNKNOWN`
- wrong representation scene produces `PROVENANCE_MISMATCH`
- missing representations produce `REPRESENTATION_MISSING`
- representation artifact URI/checksum/provenance fields are retained
- source geometry and source coordinate semantics are preserved separately

No cross-split representation link was produced in the real sample. Phase 2A remains at zero image-level leakage and zero split conflicts.

## Determinism

Repeated generation of the representative real-data link artifact produced byte-identical `links.jsonl` and identical SHA-256 fingerprints. Link IDs are derived from linkage version, annotation identity, image/split identity, linked representation identities, and spatial reference.

## Tests

Focused Phase 2C tests:

- `python -m pytest -q tests/test_representation_linking.py`
- `6 passed`

Coverage includes text-only links, real spatial-source preservation, canonical bbox/token linkage, point boundaries, out-of-bounds geometry, missing image/representation behavior, split mismatch, provenance mismatch, and deterministic artifacts.

The full repository regression is recorded after final validation in the completion result.

Full regression after the Phase 2C changes:

- `python -m pytest -q`
- `332 passed, 5 skipped, 0 failed`
- `361 warnings`, consisting of existing `rasterio` deprecation warnings and the known Windows pytest cache ACL warning.

## Backward compatibility

No scientific calculations or model behavior changed. Existing Pipeline 3 preprocessing, CROMA adapter, evidence schema, interpretation adapter, receipt catalog, API, frontend, Phase 2A annotation validation, and Phase 2B spatial validation remain unchanged.

## Known limitations

- Real BigEarthNet.txt spatial annotations cannot yet be mapped to analysis coordinates because their source coordinate convention is not authoritative.
- Representation split metadata is not embedded in every historical `RepresentationRef`; missing metadata correctly produces `SPLIT_UNKNOWN`.
- No temporal identities exist, so temporal fields remain null.
- This layer does not validate model semantics, learned grounding, segmentation, or language correctness.

## Explicit non-goals

VQA training, VLM training, caption generation, learned grounding, segmentation, temporal reasoning, CDVQA, agent planning, Pipeline 3 changes, checkpoint changes, retraining, and scientific baseline changes are intentionally excluded.
