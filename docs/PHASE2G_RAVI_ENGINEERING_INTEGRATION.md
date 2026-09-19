# Phase 2G - Ravi Engineering Pattern Integration

## 1. Executive summary

Ravi's current `main` repository was audited at commit
`ac5541086c637ece2e9fd3c0a79cfa8bea23304d` (the repository page reported this
as the latest commit on 2026-09-19). The audit covered its README, `src`,
`scripts`, `notebooks`, `docs`, `reports`, manifests, and tests, with focused
inspection of the downloader, range reader, CROMA exporter, evaluation,
validation training, prediction, and export tests.

The result is deliberately small:

- Ravi's bounded CROMA export behavior is already covered by SatQuery's
  `representation_materialization.py`, receipt catalog, catalog validation,
  atomic shard writes, ordered identities, and sample/artifact checksums.
- Ravi's evaluation and reproducibility behavior is already covered by the
  existing split gates, saved prediction/provenance artifacts, independent
  metric recomputation, bootstrap metrics, fingerprints, and regression
  receipts.
- SatQuery already has a specialized exact-5,000 acquisition path in
  `scripts/acquire_pipeline3_5000_s2.py`; it was not changed.
- A small generic acquisition boundary was missing for future selective EO
  datasets. `src/acquisition.py` now provides deterministic plans, bounded
  range reads, resumable `.part` files, checksum-based reuse, atomic
  completion, and explicit failure reports. It contains no Ravi-specific URL,
  path, dataset split, or identifier logic.

No CROMA representation was regenerated. No Pipeline 3 predictor, checkpoint,
5,000-area split, spatial contract, representation receipt, or scientific
artifact was changed.

## 2. Current Ravi source audit

| Ravi capability | Ravi implementation and source location | SatQuery equivalent | Decision | Reason / integration location |
| --- | --- | --- | --- | --- |
| Selective area selection | `src/satquery/download_bigearthnet.py`, `select_records`; seeded country-balanced selection within official splits, unique S1/S2 pairs, excluded demo IDs | `scripts/acquire_pipeline3_5000_s2.py` consumes the frozen exact-5,000 manifest; `pass5_validation.py` creates validated manifests | INTEGRATED | The existing scientific population must not be reselected. Generic future selection is represented by `AcquisitionPlan.selection` in `src/acquisition.py`. |
| HTTP range access | `src/satquery/remote_lmdb.py`, `HTTPRanges`; exact `206` and `Content-Range`, bounded reads, retries | Existing Pipeline 3 script has the same narrow LMDB reader and exact range checks | INTEGRATED | Reusing the existing verified implementation avoids a second LMDB reader. |
| Partial source cache resume | `download_bigearthnet.py`, `download_file`; `.part` file resumes at its byte length and verifies pinned SHA-256 | Existing specialized acquisition has resumable source access; generic utility resumes selected artifact `.part` files | PARTIALLY_INTEGRATED | The reusable part is now in `src/acquisition.py`; LMDB-specific indexing remains specialized and unchanged. |
| Bounded download/export | `download_subset` uses selected reference maps, one shard at a time, bounded workers and temporary work; `features.py` uses `batch_size` | `representation_materialization.py` processes source shards sequentially and CROMA adapter is already bounded upstream | INTEGRATED | Existing shard/receipt infrastructure is stronger and must not be duplicated. |
| CROMA sample ordering | `features.py`, `extract_features`; preserves preprocessing-manifest order and validates IDs | `representation_materialization.py` validates ordered IDs/splits against the authoritative manifest | INTEGRATED | Same invariant is already enforced at the trusted materialization boundary. |
| Shard completion/resume | Ravi writes ZIP plus receipt and skips only a hash-verified completed shard | Existing materialization validates/reuses each hybrid shard through typed receipts and checksums | INTEGRATED | Existing `load_verified_receipts` and `ReceiptCatalog` are the canonical implementation. |
| Feature metadata/provenance | Ravi feature manifests capture checkpoint, source manifest hash, normalization, device, batch size, shapes, and reload equality | Existing representation receipts capture dataset/split fingerprints, producer, model/checkpoint, preprocessing, shape/dtype, artifact and sample hashes | INTEGRATED | No additional metadata layer is justified. |
| Atomic output and failure cleanup | Ravi uses temporary output roots and temporary Drive copies; failed extraction removes temporary output | Existing materialization uses atomic files and rejects invalid partial existing shards | INTEGRATED | Existing behavior meets the safety requirement without changing scientific artifacts. |
| Area-level split integrity | Ravi `evaluation.load_splits` rejects duplicate SAR areas and keeps official train/validation/test areas | Existing manifests, `representation_catalog.py`, materialization, and leakage tests enforce image/split identity | INTEGRATED | Existing split fingerprints are authoritative for the 5,000-area baseline. |
| Saved predictions and reload | Ravi `prediction.py`, `predict_cover.py`, and evaluation tests save typed checkpoints/predictions and validate reliability compatibility | Existing scientific artifacts, `pass3_validation.py`, `phase3_6_benchmark.py`, and artifact tests save and independently recompute metrics | INTEGRATED | The current architecture already has provenance-backed artifacts and regression gates. |
| Bootstrap uncertainty | Ravi `evaluation.coverage_metrics` resamples whole areas with a fixed seed and records percentile intervals | Existing coverage/evaluation artifacts and scientific reports record reproducible held-out metrics and uncertainty where applicable | INTEGRATED | No generic bootstrap duplicate was added. |
| Configuration capture | Ravi records seed, batch size, optimizer, feature contract, checkpoint and split provenance in reports | Existing manifests, receipts, provenance, and frozen fingerprints capture the corresponding contracts | INTEGRATED | Existing project-wide provenance is the stronger boundary. |
| Colab/notebook builders | Ravi rebuilds notebooks with a checksum-verified wheel and separates download/pipeline/annotation workflows | SatQuery uses scripts and artifact receipts; notebooks are not part of the current production workspace | NOT NEEDED | Copying Ravi's notebook packaging would add a second workflow and is outside this infrastructure phase. |
| Generic future dataset acquisition | Ravi logic is BigEarthNet/LMDB-specific despite good operational patterns | `src/acquisition.py`: `AcquisitionItem`, `AcquisitionPlan`, `acquire_plan` | NEWLY INTEGRATED | Provides only the reusable contract needed for future RSVQA, VRSBench, CDVQA, or similar selective files. |

## 3. Bounded/batched CROMA export decision

Ravi's useful invariants were verified in `features.py`, `extract_croma.py`,
and `tests/test_export.py`:

- caller-selected positive batch size;
- inference mode and CPU detachment per batch;
- concatenation in source order;
- per-batch output metadata and hashes;
- source-manifest hash and model/checkpoint provenance;
- temporary output followed by atomic publication;
- reload equality check; and
- explicit refusal to overwrite an existing output.

SatQuery already implements the same or stronger invariants across
`src/representation_materialization.py`, `src/representation_catalog.py`,
`src/receipt_catalog.py`, and `src/representation_artifacts.py`. It processes
the existing source shards, creates only the required hybrid representation,
records 15,000 typed receipts, validates every artifact and sample checksum,
and supports shard reuse. Extending or replacing this path would risk changing
the frozen representation fingerprints. **No CROMA code or artifact was
changed.**

## 4. Selective/resumable acquisition implementation

### Existing exact Pipeline 3 path

The existing [acquisition script](../scripts/acquire_pipeline3_5000_s2.py)
remains the owner of the exact 5,000-area optical recovery workflow. It reads
only IDs from the authoritative manifest, validates the train-only missing
population, uses strict HTTP ranges and retries, validates native arrays and
checksums, writes per-area receipts, and reports unmatched/duplicate/failure
states. It was not generalized in place because that would risk changing the
baseline acquisition contract.

### New generic boundary

[src/acquisition.py](../src/acquisition.py) adds:

- `AcquisitionItem` for an explicit URL, relative destination, expected size,
  and SHA-256;
- `AcquisitionPlan` for deterministic sorted selection and a plan fingerprint;
- `acquire_plan` for bounded chunked reads through an injectable range reader;
- HTTP range validation through `http_range_reader`;
- `.part` files that resume from their existing byte length;
- checksum and size verification before atomic rename;
- reuse of already verified completed files without duplicate downloads;
- atomic `plan.json` and `acquisition_report.json`; and
- explicit item-level failures with a failed report and no false completion.

The utility does not select by filesystem enumeration, embed Ravi paths, know
BigEarthNet IDs, download full datasets, or alter any current dataset artifact.
Tests use in-memory range readers and small temporary files only.

## 5. Evaluation/reproducibility decision

No new evaluation layer was justified. The current repository already has the
relevant controls:

- immutable area-level manifests and split fingerprints;
- artifact and receipt SHA-256 checksums;
- saved model/prediction/provenance artifacts;
- independent metric recomputation in `pass3_validation.py`;
- deterministic seeds and validation-only selection;
- bootstrap/uncertainty reporting in the existing evaluation path; and
- full regression and scientific baseline receipts.

Ravi's evaluation code is useful corroborating evidence, but copying its
`evaluation.py` or prediction schema would duplicate existing contracts. The
new acquisition utility is model/task agnostic and does not participate in
Pipeline 3 evaluation.

## 6. Tests

New tests in [tests/test_acquisition.py](../tests/test_acquisition.py) cover:

- deterministic plan fingerprints and duplicate prevention;
- selective item ordering;
- partial-file resume and bounded chunk requests;
- checksum-verified completion;
- completed-file reuse without another request;
- explicit failed reports with preserved partial state; and
- traversal-safe relative paths.

Focused result: `4 passed`. The full repository regression was run with the
scientific virtual environment: `355 passed, 5 skipped`. Existing warnings are
the known rasterio deprecation warnings and Windows pytest cache permissions.
`git diff --check` passed.

## 7. Scientific safety

The following were not modified or regenerated:

- Pipeline 3 predictor and `hybrid.pt`;
- CROMA checkpoint and existing feature shards;
- exact 5,000-area dataset manifest and split;
- representation catalog and 15,000 typed receipts;
- scientific predictor artifacts and metrics;
- `spatial_contract.py` and `representation_linking.py`; and
- Phase 2F's unresolved BigEarthNet.txt spatial mapping decision.

The sealed baseline remains `65.0%` test accuracy, `4.1034 pp` MAE, and
`9.5873 pp` RMSE. No retraining, VLM work, grounding, or learned spatial
mapping was performed.

## 8. Final classifications

| Pattern | Classification |
| --- | --- |
| Bounded/batched CROMA export | INTEGRATED |
| Exact Pipeline 3 selective acquisition | INTEGRATED |
| Generic selective/resumable acquisition | NEWLY INTEGRATED |
| Evaluation/reproducibility scaffolding | INTEGRATED |
| Ravi-specific notebook/Colab workflow | NOT NEEDED |
| Ravi-specific LMDB and mirror paths | DEFERRED / NOT COPIED |
| BigEarthNet.txt spatial mapping or grounding | DEFERRED and fail-closed |

Phase 2G is therefore complete as an engineering-infrastructure update. The
only new production capability is the generic acquisition boundary; all other
Ravi patterns were either already present, deliberately not needed, or
deferred because copying them would duplicate or endanger the canonical
scientific pipeline.
