# Trusted Receipt Catalog and Artifact Lifecycle

## 1. Purpose

Phase 6A.8 adds a small local discovery and lifecycle index for already verified
Pipeline 3 representation artifacts. The catalog answers which representation
exists for a scene/run and whether it is currently reusable. It does not compute,
modify, or become a source of scientific values.

```text
SceneBundle -> ReceiptCatalog -> RepresentationRef -> ArtifactRef
            -> ArtifactResolver -> verified NumPy representation
```

## 2. CatalogRecord

`CatalogRecord` wraps the existing immutable `RepresentationRef` and receipt
`ArtifactRef` with only lifecycle metadata: UTC `created_at`, `active`, `invalid`,
or `expired` status, optional `expires_at`, and an invalidation reason. Artifact
ID, URI, checksum, byte size, scene, run, representation type, modality, shape,
dtype, producer, provenance, and spatial metadata remain in the existing nested
contracts rather than being copied into a second schema.

Identity stays separated:

- `artifact_id` is catalog/artifact identity;
- `ArtifactRef.uri` is the logical address;
- SHA-256 is concrete content identity; and
- `RepresentationRef` is scientific representation metadata and identity.

No filesystem path is scientific identity.

## 3. Storage format

Each enabled live persistence root contains `receipt_catalog.json`. It is a
human-readable, deterministic JSON document with `catalog_version: 1` and a
sorted `records` array. Writes use a same-directory temporary file, flush and
`fsync`, then atomic replacement. The file can be inspected, copied, backed up,
and reopened without a server or database.

## 4. Registration

`register` and `register_many` accept only constructed `RepresentationRef` and
receipt `ArtifactRef` values. Registration requires checksum, size, shape,
dtype, run identity, and consistent receipt provenance. The configured resolver
verifies the receipt and loads/verifies every NumPy representation before one
atomic catalog update.

Registering exactly the same reference is idempotent. Reusing an artifact ID,
logical URI, or representation reference ID for different metadata/content is
a conflict and never overwrites the prior record. Independent versions coexist
through the already explicit scene/run logical path; no extra version semantics
are invented.

## 5. Lookup and list

`lookup` filters by scene ID, representation type, modality, run ID, artifact
ID, logical artifact URI, and spatial level. It returns active records by
default. `list_representations` returns their existing `RepresentationRef`
values for a scene/run. Invalid and expired records require `active_only=False`
and therefore cannot be accidentally treated as reusable.

## 6. Validation and materialization

`validate` delegates to `ArtifactResolver.load_numpy`, the Phase 6A.6 boundary
that checks namespace containment, existence, byte size, SHA-256, safe NumPy
loading, shape, and dtype. The catalog has no second loader. `materialize`
validates first and refuses non-active or corrupt entries.

## 7. Invalidation and lifecycle

Lifecycle transitions are intentionally small:

```text
ACTIVE -- validation failure --> INVALID
ACTIVE -- retention reached --> EXPIRED
```

Validation failure records a stable reason and atomically persists `invalid`.
An elapsed retention deadline becomes `expired` during lookup/validation. These
states are excluded from normal lookup. Neither transition deletes the catalog
row or artifact file. Reactivation and cleanup are deliberately absent.

## 8. Trust and security boundary

The catalog is internal Python infrastructure and has no public API mutation
route. Registration does not accept a path or URI string as proof of trust. It
accepts the typed references produced by trusted persistence, checks controlled
`artifact://` syntax through the contracts, and resolves only explicitly
configured namespaces. Absolute paths, traversal, unknown namespaces, malformed
checksums, non-positive shapes, unsupported dtypes, object arrays, unknown JSON
fields, malformed records, inconsistent provenance, and duplicate conflicts
fail closed. NumPy always uses `allow_pickle=False`.

The index file itself is trusted local state. Filesystem permissions, signing,
multi-user authorization, and hostile concurrent writers remain operational
concerns outside this phase.

## 9. Scene/run isolation

Persistence URIs and IDs include scene and analysis/run IDs. Lookup applies all
provided filters together, so scene A cannot resolve scene B and `t1` cannot
resolve `t2`. Tests persist two contents under separate temporal run IDs and
materialize each independently. Acquisition metadata remains in provenance;
future temporal code should use acquisition identity in addition to run identity.

## 10. SceneBundle and RepresentationSet

`attach_to_scene` returns a new immutable `SceneBundle` whose existing
`RepresentationSet` is supplemented with active catalog discoveries. It filters
by the bundle scene and its analysis ID when present. The catalog discovers
persisted references; `RepresentationSet` remains the scene's representation
description. Neither replaces the other.

## 11. Persistence integration

Enabled `RepresentationPersistenceSession.finalize` now performs:

1. deterministic NumPy serialization;
2. immutable write/reuse;
3. checksum, size, shape, dtype, and exact-value verification;
4. deterministic receipt write/reuse;
5. resolver verification of receipt and representations; and
6. atomic registration in `<storage_root>/receipt_catalog.json`.

Catalog registration never precedes verification. If cataloging fails after
artifact creation, finalization raises `artifacts persisted but catalog
registration failed` and does not publish a successful `PersistenceOutcome`.
Files are not automatically deleted or rolled back.

## 12. Restart and reproducibility

A fresh `ReceiptCatalog` reconstructs only allowlisted dataclass fields through
their normal validators. Restart coverage reopens the JSON file, discovers a
prior scene/run token reference, validates it, and materializes the exact
`[225,768]` float32 array without knowing a filesystem path.

## 13. Corruption behavior

Tests modify bytes without changing size, truncate files, delete files, and
load malformed/inconsistent catalog documents. Each artifact failure becomes a
persistent `invalid` record and disappears from active lookup. Malformed catalog
metadata prevents catalog opening rather than yielding partially trusted rows.

## 14. Performance

The implementation reads the small JSON index once when constructed and uses
an in-memory tuple for lookup. On the Phase 6A.8 development machine, 1,000
scene/run-filtered lookups over a one-record catalog took 0.001060 seconds
(approximately 1.06 microseconds each); the test retains a deliberately broad
two-second regression ceiling. Registration/validation deliberately
pay artifact hashing and NumPy verification cost because trust is more important
than write speed. This is not intended for large or distributed catalogs.

## 15. Real `61_39` validation

The Phase 6A.7 real workflow remains the validation path for the six CROMA
representations: optical, SAR, and joint token arrays plus their GAP vectors.
Actual runtime metadata—not catalog assumptions—must establish shapes and
dtypes. The Phase 6A.8 rerun discovered, validated, reopened after catalog
restart, and materialized all six actual outputs: three float32 `[225,768]`
token arrays and three float32 `[768]` GAP arrays. Physical dimension remained
62, pooled CROMA 2304, hybrid 192, spatial evidence 17 regions/3 claims,
interpretation `ANSWERED`, and prediction null. The normalized scientific
before/after payloads were identical at SHA-256
`e49f2cce3a9cf97e33ba0eaa137e0fdf0e87b82bd681a80a365f3ea5b29292be`.

## 16. Future VQA usage

A future trusted capability can request `scene_id`, `run_id`, `joint_croma`,
`optical_sar`, and `token`, then materialize the verified representation through
the resolver. It never needs a storage path. This is readiness, not VQA.

VQA still needs a question-conditioned model, question/image fusion, trained
task head, dataset integration, RSVQA evaluation, answer validation, evidence
attachment, unsupported-question handling, and calibration policy.

## 17. Future grounding usage

The catalog preserves `SpatialReference`, including token-grid, CRS when known,
mapping reference, and mapping version. This keeps the chain from token position
to spatial mapping to potential region available. It does not supply learned
phrase-token relevance and does not claim grounding.

## 18. Future temporal usage

Separate T1/T2 run artifacts coexist under distinct run-qualified identities.
Future temporal analysis still requires acquisition pairing, co-registration,
transformation provenance, temporal models/rules, evaluation, and evidence that
cites both parent observations. None is implemented here.

## 19. Scientific and API invariants

Catalog work occurs after scientific computation and receipt verification. It
does not change CROMA, physical features, hybrid representations, prediction,
evidence, interpretation, API response fields, or report schema. The catalog is
not exposed through `/api/analyze` or `/api/report` and never exposes storage
paths.

## 20. Known limitations

- Local JSON assumes one trusted writer at a time; there is no cross-process lock.
- No background expiration scan, cleanup, deletion, quota service, or repair.
- No signatures or tamper-evident catalog journal.
- Records are loaded into memory, suitable for a small local index only.
- Expiration is metadata/state; files remain present.
- Acquisition-specific lookup beyond existing scene/run/provenance is future work.

The governing order remains: Pipeline 3 computation, verified representation,
receipt/checksum, then catalog discovery. The catalog answers where a verified
representation is and whether it is reusable; it never decides its scientific
value.
