# Phase 3J.6 — Temporal Training Decision and Branch Freeze

## Final decision

**TEMPORAL_TRAINING_BLOCKED**

**TEMPORAL_BRANCH_FROZEN_AT_EXECUTION_LAYER**

The evidence through Phases 3J.1–3J.5 does not justify temporal-model training. Temporal contracts and a real RGB execution boundary are complete, but no audited source satisfies every non-relaxable data-readiness condition. No temporal model, fine-tuning, benchmark, change detector, or grounding implementation is authorized by this decision.

## Evidence reviewed

This decision reviewed the Phase 3J.1 CDVQA verification, Phase 3J.2 ChangeChat/DeltaVLM audit, Phase 3J.3 contract, Phase 3J.4 RSRCC readiness gate, Phase 3J.5 execution report, the temporal adapter and focused tests, the RSRCC fixture manifest, and the RSRCC execution receipt.

| Source | Supervision / imagery / linkage | Time, correspondence, registration | License, pinning, splits, leakage, acquisition | Compatibility and result |
|---|---|---|---|---|
| CDVQA | Change-QA supervision is verified at dataset-design level; image bytes are unavailable from the audited official source; pair fields are only partial and annotation IDs are unknown. | T1/T2 order, timestamps, sensor, correspondence, and registration are unknown at record level. | Committed JSON has Apache-2.0, but image terms and image pinning are unverified. Annotation partitions exist, but no image-level leakage audit or bounded fixture route exists. | Sensor/bands are unknown; not compatible with S2/S1 paths. **BLOCKED.** |
| ChangeChat-105k + LEVIR-CC / DeltaVLM | Explicit change captioning/VQA supervision; ordered A/B paths and record linkage are verified; separate answer IDs are not established. LEVIR RGB image bytes are available only as a full archive. | Pre/post order and spatial correspondence are verified; record timestamps, intervals, and reproducible pixel registration are absent. | CC-BY-4.0 annotations and author-linked Apache-2.0 imagery are verified; Git revisions are pinned. ChangeChat validation and completed duplicate/geographic leakage audit are partial. A bounded fixture is blocked by the 2.68 GB archive. | RGB, not native S2/S1. **BLOCKED.** |
| RSRCC | Co-hosted RGB before/after images, `CHANGE_VQA` rows, exact UUID pair keys, and row locators are verified. No source-issued annotation ID exists. | Before-to-after semantic order and spatial correspondence are verified. Dates, intervals, sensor/source-scene identity, and reproducible registration evidence are absent. | Apache-2.0 is declared; revision `7898de7bfd08bc404d9a92e1caaa9dce91b0c3ea` is pinned. Train/val/test exact pair overlap is zero; geographic/near-duplicate audit is unavailable. Three per-file-hashed pairs were acquired. | RGB only, separate from native S2/S1. Strongest structural fixture, not training-ready. **BLOCKED.** |
| QAG-360K / CDQAG | Change QA/grounding is reported, but release, imagery, split, and record schema were not verified. | Not established. | License, pinning, provenance, and bounded route not established. | **UNKNOWN / BLOCKED.** |
| RSCC | Change-description intent and pre/post imagery are claimed, but per-pair metadata was not established. | Not established at record level. | Upstream-term dependency and split protocol remain unverified. | **UNKNOWN / BLOCKED.** |

## Phase 3J.5 execution interpretation

**PROVEN:** a real RSRCC T1/T2 RGB pair loaded; `TemporalInput` validated the pair; `TemporalRGBAdapter` prepared two RGB inputs; the existing Qwen processor accepted them; FP16 CUDA generation completed; and provenance was retained.

The executed pair was `d259f34c_8693_418e_9dab_ef14b860319f`, using `Qwen/Qwen2.5-VL-3B-Instruct` revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. It generated eight tokens in 40.74490 seconds total, with 7,802,813,952 bytes peak CUDA allocation. The receipt remains at `artifacts/test_fixtures/rsrcc_execution_phase3j5.json`.

**NOT PROVEN:** temporal learning, change detection, change-VQA accuracy, temporal generalization, pixel-registered change reasoning, multispectral temporal reasoning, or SAR temporal reasoning. The generated language is not annotation truth or objective change evidence.

## Training-readiness recheck

The table evaluates the strongest currently auditable candidate, RSRCC, against the unchanged all-gates requirement. A `PASS` does not compensate for a `FAIL`, `PARTIAL`, or `UNKNOWN` elsewhere.

| Required condition | Status | Evidence / unresolved point |
|---|---|---|
| Authoritative T1 imagery | PASS | Co-hosted RSRCC `before` PNGs, revision-pinned and locally hashed. |
| Authoritative T2 imagery | PASS | Co-hosted RSRCC `after` PNGs, revision-pinned and locally hashed. |
| Exact pair IDs | PASS | UUID filename stems form ordered pair keys. |
| Annotation IDs | FAIL | No source-issued annotation ID; only CSV-row locator and annotation hash. |
| Explicit change supervision | PASS | `CHANGE_VQA` rows. |
| Temporal ordering | PASS | Dataset-documented before-to-after semantic order. |
| Timestamps / intervals | FAIL | No record-level dates or intervals. |
| Spatial correspondence | PASS | Source documents temporally aligned pairs; retained as `SPATIALLY_CORRESPONDING`. |
| Registration evidence | FAIL | No CRS, transforms, residuals, or reproducible registration procedure. |
| Image license | PASS | Apache-2.0 declared by the co-hosted dataset. |
| Annotation license | PASS | Apache-2.0 declared by the co-hosted dataset. |
| Immutable source pinning | PASS | Dataset revision is recorded above. |
| Official train split | PASS | Published train metadata. |
| Official validation split | PASS | Published validation metadata. |
| Official test / evaluation protocol | PASS | Published test metadata and official split partition. |
| Leakage audit | PARTIAL | Exact ordered-pair / UUID split overlap is zero; geographic and near-duplicate fields are absent. |
| Per-file provenance | PARTIAL | Fixture hashes and paths exist, but sensor and source-scene provenance are absent. |
| Reproducible acquisition | PASS | Individual immutable-revision objects supported bounded, hashed acquisition. |

Unresolved non-relaxable requirements are: source-issued annotation IDs; record-level timestamps/intervals; reproducible registration evidence; sensor/source-scene provenance; and a geographic/near-duplicate leakage audit. These gaps also prevent a valid native S2 or S1 temporal claim: RSRCC is RGB.

## Branch freeze

`TEMPORAL_BRANCH_FROZEN_AT_EXECUTION_LAYER` means:

- temporal input/representation contracts are implemented;
- real RSRCC RGB pair execution through Qwen is demonstrated;
- learned temporal training and fusion are deferred;
- change benchmark evaluation is deferred; and
- no further temporal model implementation should proceed until a dataset clears the unchanged readiness gate.

This freezes speculative temporal-model work, not the completed execution boundary. A future temporal VLM may be substituted through the model-swappable interface only after data readiness is established.

## Conditions to reopen temporal training

Temporal training may be reconsidered only when one source provides all of:

1. auditable T1/T2 bytes with exact ordered pair identifiers and per-file provenance;
2. exact source-issued annotation linkage/identifiers and explicit temporal change supervision;
3. record-level timestamps or acquisition intervals;
4. documented spatial correspondence plus reproducible registration evidence;
5. verified image/annotation licensing, immutable pinning, and reproducible bounded acquisition;
6. official train/validation/test protocol with exact-pair, geographic, and near-duplicate leakage audit; and
7. a modality-compatible, separately validated adapter path without changing frozen S2/S1 or scientific paths.

## Immutability verification and next capability

This documentation-only phase did not modify temporal contracts, the RSRCC fixture or execution receipt, S2 learned adapter/checkpoint, SAR projector, CROMA, scientific predictor/checkpoints, representations, receipts, splits, or Phase 2F semantics. The frozen scientific baseline remains **65.0% accuracy**, **4.1034286734 pp MAE**, and **9.5873312123 pp RMSE**.

The next independent SatQuery requirement is **GROUNDING**: audit text, point, bbox, mask/polygon, evidence-linked grounding, benchmark/data availability, and spatial provenance. No grounding is implemented by this phase.
