# Opt-in Live Representation Persistence

## 1. Problem being solved

Pipeline 3 computes optical, SAR, and joint CROMA token/GAP tensors but its
legacy result retains only representation shapes and pooled scene values. Phase
6A.6 could truthfully describe those live tensors only as `not_addressable`.
This phase adds an optional side effect that preserves selected already-computed
arrays as verified immutable artifacts. It changes no scientific calculation.

## 2. Why persistence is opt-in

The default `run_analysis(request)` call remains unchanged and creates no tensor
files, receipts, references, or extra API fields. Persistence costs storage,
checksum I/O, CPU/GPU transfer for selected tensors, and validation time. It is
therefore available only through an explicit internal Python policy; frontend
and public API clients cannot accidentally enable it.

## 3. Policy design

`RepresentationPersistencePolicy()` is disabled. Enabling it requires all of:

- `enabled=True`;
- a non-empty explicit set of `PersistableRepresentation` values;
- a physical `storage_root` supplied by trusted internal code;
- a simple logical artifact namespace;
- mandatory SHA-256 verification; and
- a positive byte limit, 64 MiB by default.

There is no “persist all” boolean. The frozen policy rejects enabled empty
selections, missing roots, unsafe namespaces, disabled checksums, and invalid
limits. `CROMA_REPRESENTATIONS` is an explicit six-item selection, not a dynamic
wildcard.

## 4. Supported representations

The numeric writer supports the six core CROMA outputs and five optional values:

| Selection | Live material | Approximate `61_39` `.npy` size | Spatial meaning | Decision |
|---|---:|---:|---|---|
| Optical/SAR/joint CROMA tokens | three `[225,768]` float32 tensors | ~691 KB each | latent 15×15 token grid | primary supported use |
| Optical/SAR/joint GAP | three `[768]` float32 vectors | ~3.2 KB each | scene-level latent | primary supported use |
| Raw optical | `[12,120,120]` float32 | ~691 KB | aligned pixel grid | supported, explicit only |
| Raw SAR | `[2,120,120]` float32 | ~115 KB | aligned pixel grid | supported, explicit only |
| Physical features | `[62]` float32 | ~376 bytes | feature-level, not a map | supported, low cost |
| Pooled CROMA | `[2304]` float32 | ~9.3 KB | scene-level concatenation | supported |
| Hybrid | `[192]` float32 | ~896 bytes | scene-level untrained representation | supported |
| Regions | live JSON | variable | deterministic region structure | not duplicated by numeric persistence; already present in evidence/report artifacts |

Nothing is selected by default. Raw arrays are supported for controlled research
reproducibility but are not part of the recommended six-output CROMA selection.

## 5. Artifact generation

`RepresentationPersistenceSession` observes existing arrays only when its policy
selects them. It makes a contiguous defensive copy and waits until the normal
analysis result has been constructed before finalizing. Each array is serialized
once with `numpy.save(..., allow_pickle=False)` under:

```text
<configured root>/<scene id>/<analysis id>/<controlled filename>
```

The architectural identity is instead:

```text
artifact://<namespace>/<scene id>/<analysis id>/<controlled filename>
```

The root never enters `ArtifactRef`, `RepresentationRef`, the receipt, or the
legacy result.

## 6. Receipt and checksum semantics

Every representation receipt row records selection, reference/artifact IDs,
logical URI, actual content SHA-256, byte size, shape, dtype, representation
type, modality, spatial semantics, producer/model, preprocessing, run ID, and
provenance link. The receipt is deterministic JSON and has its own `ArtifactRef`,
size, and SHA-256.

SHA-256 is computed over the actual `.npy` bytes. After writing, each artifact is
read again, hashed, loaded without pickle, checked for shape/dtype, and compared
exactly with the captured in-memory array before it is exposed as `available`.

## 7. Immutability

Artifacts and receipts use exclusive creation. An existing target with identical
content is verified and reused. Different content at the same logical scene/run
location raises an immutable-artifact conflict; it is never overwritten. Normal
analysis IDs naturally create new run-versioned collections.

External filesystem modification cannot be prevented by a local Python contract,
but subsequent resolution detects size or checksum changes and fails closed.

## 8. Provenance

The receipt reuses existing Pipeline 3 provenance: analysis ID, public input
identifiers, retrieval source, preprocessing operations, physical schema, CROMA
source, and execution timestamp. CROMA references also record the existing model
identity and actual checkpoint SHA-256. Each reference links back to the receipt;
the receipt identifies the run and input scene.

## 9. SceneBundle integration

The advanced internal capability path returns `AdaptedAnalysis`, containing the
legacy result, populated `SceneBundle`, and `TaskResult`. Successfully persisted
references replace matching `not_addressable` or in-memory metadata references
inside `RepresentationSet`. Unselected references retain their truthful prior
state. The SceneBundle and TaskResult provenance reference the receipt. The
unchanged legacy result receives no persistence field.

Use `run_deterministic_scene_analysis_with_context(..., persistence_policy=...)`
when the caller needs the representation-bearing SceneBundle. The existing
`run_deterministic_scene_analysis` remains usable and may receive the same policy
when only the TaskResult is needed.

## 10. Materialization

Phase 6A.6 `ArtifactResolver` maps an explicitly configured namespace to its
physical root. A future consumer performs:

```text
RepresentationRef → ArtifactRef → configured ArtifactResolver → verified ndarray
```

Resolution verifies namespace containment, artifact existence, size, checksum,
NumPy format, reference shape, and dtype. Pickle loading is disabled.

## 11. Storage namespace and catalog

The namespace is logical and environment-specific. A development run might use
`artifact://satquery-live/61_39/<run-id>/croma_joint_encodings.npy` while trusted
configuration maps `satquery-live` to a local directory. No database, global
catalog, cloud URI, automatic cleanup, or implicit root is introduced. Phase
6A.8 creates a small `receipt_catalog.json` in the configured root and registers
the completed references only after artifact and receipt verification. See
[`RECEIPT_CATALOG_ARCHITECTURE.md`](RECEIPT_CATALOG_ARCHITECTURE.md).

## 12. Security boundaries

- Public requests cannot supply a persistence policy or storage root.
- Namespaces, scene IDs, and run IDs accept only simple logical identifiers.
- Resolved paths must remain under their configured namespace root.
- Filenames come from a controlled enum-to-descriptor mapping.
- NumPy object arrays and duplicate captures are rejected.
- Deserialization always uses `allow_pickle=False`.
- Selection size is checked before directories or artifacts are written.
- Missing selected outputs, checksum mismatch, metadata mismatch, tampering, and
  conflicting writes fail closed.

## 13. Performance and storage measurements

On the development machine, real `61_39` with the six CROMA selections produced
six arrays plus one receipt totaling **2,098,381 bytes**. The three token arrays
dominate storage; GAP vectors are small.

Cold persistence-off execution measured 20.98 seconds. The immediately following
warm persistence-on run measured 2.73 seconds, so those two values are not a fair
overhead comparison because model loading/cache state differs. After warming the
same process, persistence-off measured 0.79 seconds and the capability persistence-
on path measured 2.34 seconds: an observed **1.55 second** addition on this machine.
That includes checkpoint hashing, device-to-CPU copies, six writes, receipt output,
and complete reload verification. These are operational measurements, not portable
latency guarantees.

## 14. Real 61_39 validation

Persistence off wrote zero artifacts. Persistence on for the explicit six CROMA
outputs wrote three token tensors with actual shape `[225,768]`, three GAP vectors
with actual shape `[768]`, and one receipt. All six materialized values matched
their captured in-memory values exactly. Token modality identity was preserved as
optical, SAR, and optical-SAR joint; token grids remained 15×15 row-major latent
structure, not segmentation or grounding.

Both modes retained physical dimension 62, pooled CROMA dimension 2304, hybrid
dimension 192, 17 regions, three claims, interpretation `ANSWERED`, null model
confidence, and no task prediction.

## 15. Scientific invariants

The normalized scientific fingerprint was identical with persistence disabled
and enabled:

```text
b7901bc61b5f50196919e41c6ca96bf03d612ceda9668efa8dbef44bf446caa0
```

Normalization excludes run UUIDs, timings, cache state, and request-derived
semantic hashes while including physical/CROMA/hybrid values, evidence,
interpretation, model-result, confidence, and temporal state. Persistence is an
optional side effect around the existing computation and does not alter the
legacy result.

## 16. VQA implications

A future VQA capability can request optical, SAR, or joint CROMA token references,
resolve verified tensors without knowing Pipeline 3 paths, and combine them with
scene-level representations. VQA still requires a validated question-conditioned
model, training/evaluation datasets, task head, question/multimodal fusion,
answer-evidence validation, unsupported-question behavior, and confidence/
calibration policy. None is implemented here.

## 17. Grounding implications

Persisted tokens plus the existing token-grid mapping can support controlled
grounding research. A future system still needs a validated mapping from language
phrases to token relevance and then to regions, masks, or boxes, with task-specific
evidence. Latent token position alone is not learned grounding.

## 18. Temporal implications

Artifact identity contains scene and analysis/run IDs, so separate T1 and T2 runs
cannot be silently conflated. Temporal work still needs paired SceneBundles,
acquisition timestamps in task identity, co-registration validation, transformation
provenance, temporal representations, change logic, evaluation, and evidence that
cites both parents. No temporal analysis is added.

## 19. Known limitations

- Persistence is synchronous and local; selected writes add measurable latency.
- Checkpoint hashing is intentionally performed for persisted CROMA provenance.
- The catalog is a small single-writer local JSON index; there is no garbage
  collection, quota service, background expiry worker, or concurrency lease
  beyond exclusive immutable artifact creation.
- A process crash during a new exclusive write can leave an incomplete file that
  later verification rejects; automatic repair is out of scope.
- Regions/evidence remain in their existing JSON/report lifecycle rather than
  being duplicated by this numeric writer.
- Public API clients cannot request or discover these internal artifacts yet.

## 20. Future improvements

Phase 6A.8 supplies the small trusted catalog and lifecycle metadata. A later
operational phase may add carefully scoped locking, backup, and explicit cleanup;
it should not add VQA until task models, datasets, evaluation, evidence, and
epistemic contracts are independently ready.
