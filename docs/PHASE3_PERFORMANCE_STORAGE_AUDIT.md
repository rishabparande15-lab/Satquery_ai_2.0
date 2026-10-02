# Phase 3 performance, memory, VRAM and storage audit

**Scope (2026-09-19):** the frozen 5,000-area Pipeline 3 core and the resource contract for a future image-language adapter. Target machine: Windows, RTX 5060 with 8 GB VRAM, 24 GB host RAM. This is a documentation decision; no CROMA extraction, retraining, VLM loading, cache regeneration, or dataset change was performed. Values labeled *payload arithmetic* are uncompressed array calculations, not measurements of disk, process RAM, or GPU residency.

## 1. Actual resource inventory

| Item | Evidence and observation | Limit |
|---|---|---|
| Historical physical + CROMA feature extraction and validation | 910.732 s for 5,000 areas, 5.49 areas/s, 329.4 areas/min; peak `torch.cuda.max_memory_allocated` 904,975,360 B (about 0.843 GiB). [Training report](PIPELINE3_5000_TRAINING_REPORT.md), [source builder](../scripts/run_pipeline3_5000_baseline.py). | Combined physical, raster loading, inference and validation run. Per-stage time, model-only allocation, allocator reserved memory, other GPU processes and system-wide VRAM: **UNKNOWN / NOT MEASURED**. Historical hardware/run, not a new RTX 5060 benchmark. |
| Historical five baseline fits/evaluation | 91.837 s, 38,397,952 B peak PyTorch GPU allocation. Training probe batch size 1,024; [training report](PIPELINE3_5000_TRAINING_REPORT.md). | Full device usage and host peak RAM: **UNKNOWN / NOT MEASURED**. This is probe training, not CROMA or VLM training. |
| CROMA checkpoint | `<machine-local-data-root>\checkpoints\CROMA_base.pt` measured 777,563,846 B on disk; frozen checkpoint SHA-256 `0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63` recorded in [materialization report](PHASE2D2_REPRESENTATION_MATERIALIZATION.md). CROMA base loaded with `modality=both`, resolution 120, inference mode. | Disk serialization is **not** weight residency. Weight-only CPU/GPU footprint and activation/buffer split: **UNKNOWN / NOT MEASURED**. |
| Feature source cache | 157 compressed NPZ shards, 44,280,954 B total; observed 72,220–283,814 B/shard; manifest 37,098 B. Includes physical, three CROMA GAP arrays, targets and identity metadata; no token arrays. | Compressed bytes cannot be assigned entirely to one representation. |
| Hybrid store | 157 compressed NPZ shards, 15,671,729 B total; observed 25,602–100,480 B/shard. | Additional persisted artifact for frozen predictor, not a second copy of source physical/GAP arrays. |
| Metadata | 15,000 typed receipt records in `records.jsonl`: 16,522,696 B. 157 shard receipt JSON files: 582,662 B total. Materialization manifest: 584,476 B. Local representation catalog: 75,000-row `records.jsonl` 55,912,312 B; its manifest, provenance and report add 3,372 B. | Catalog is a reproducible local inventory, not 75,000 materialized vectors. These measurements are current file lengths, not compressed sizes. |
| Source archives | Historical report records 511,289,395 B of nested source cache; current files are S1 ZIP 504,950,273 B and reference ZIP 6,339,122 B. | These archives are source imagery/reference, distinct from the 44 MB feature cache. Native optical GeoTIFF collection total: **UNKNOWN / NOT MEASURED**. |
| Phase 2D.2 admission/materialization | First run 27.143 s (source validation 1.276 s, hybrid assembly 2.709 s); full reuse 23.075 s, 157/157 reused. CPU regression 0.730 s with 95,154,176 B observed RSS increase and no GPU use. | Materialization peak host RSS: **UNKNOWN / NOT MEASURED**. Current manifest records the *resume* timing, not first-run creation timing. |
| Scientific CPU regression, later Phase 2E artifact | RSS 513,495,040 B before, 609,341,440 B after, delta 95,846,400 B; GPU peak 0. | A before/after delta is not peak RAM. These are separate runs and must not be combined into a memory bound. |

The feature builder defaults to **32 areas per persisted shard**, **8 areas per CROMA inference batch**, float32 canonical arrays, `torch.inference_mode()`, and a CUDA device unless overridden. It processes 156 full shards and one 8-area shard; the last inference mini-batch in each shard is bounded by 8. Training's 1,024 batch applies to the small probe, not CROMA. The measured extraction throughput is end-to-end for that historical run. CROMA-only inference throughput: **UNKNOWN / NOT MEASURED**.

## 2. Representation cost model

The 5,000-area column below is **uncompressed payload arithmetic** unless it says measured. It excludes raster containers, compression, metadata, Python objects, temporary copies and allocator overhead. “Existing” distinguishes on-disk source imagery from typed persisted representations.

| Representation / stage | Shape per area | Dtype | Bytes/area | 5,000-area payload | Existing? | Persist? |
|---|---:|---|---:|---:|---|---|
| Native optical sensor bands | B01/B09: 2×20×20; B02/B03/B04/B08: 4×120×120; B05/B06/B07/B8A/B11/B12: 6×60×60 | uint16 verified by acquisition path | 160,000 | 800,000,000 | Source GeoTIFFs | Preserve source; do not duplicate as representation |
| Native SAR VV/VH | 2×120×120 in sampled source ZIP TIFF | float32 observed in one VV TIFF; entire archive dtype **UNKNOWN / NOT MEASURED** | 115,200 *if both bands match sample* | 576,000,000 *conditional* | Source ZIP | Preserve source; no duplicate |
| Canonical, reprojected optical | 12×120×120 | float32 | 691,200 | 3,456,000,000 | Produced per loaded area by builder | Ephemeral |
| Canonical, reprojected SAR | 2×120×120 | float32 | 115,200 | 576,000,000 | Produced per loaded area by builder | Ephemeral |
| Normalized CROMA optical input | 12×120×120 | float32 | 691,200 | 3,456,000,000 | Produced per inference call | Ephemeral |
| Normalized CROMA SAR input | 2×120×120 | float32 | 115,200 | 576,000,000 | Produced per inference call | Ephemeral |
| `physical_62d` | 62 | float32 | 248 | 1,240,000 | Shared source NPZ, verified | Already persisted; reuse |
| `joint_croma_gap_768d` | 768 | float32 | 3,072 | 15,360,000 | Shared source NPZ, verified | Already persisted; reuse |
| Optical or SAR GAP (each) | 768 | float32 | 3,072 | 15,360,000 each | Arrays in source NPZ, not typed catalog entries | No new copy without consumer |
| `hybrid_830d` | 830 | float32 | 3,320 | 16,600,000; measured compressed shards 15,671,729 | Verified hybrid NPZ | Already persisted for predictor |
| One optical, SAR, or joint CROMA spatial-token family | 225×768 | float32 | 691,200 | 3,456,000,000 (about 3.22 GiB) | No 5,000-area token store | Lazy; conditional |

Optical native arithmetic is `[(2×20²)+(4×120²)+(6×60²)]×2 = 160,000` B/area, using the actual band list and shapes in [acquisition code](../scripts/acquire_pipeline3_5000_s2.py). It is **not** a measured total of compressed GeoTIFF files. The SAR source row is explicitly conditional because only one TIFF header was inspected; the canonical loaded SAR row is established by [builder code](../scripts/run_pipeline3_5000_baseline.py). All three token families together would be 10,368,000,000 B (about 9.66 GiB) before metadata. Their compression ratio and generation time: **UNKNOWN / NOT MEASURED**.

## 3. VRAM on the 8 GB target

| Workload | Weights | Input/output payload | Activations and temporary buffers | Fit conclusion |
|---|---|---|---|---|
| Existing CROMA inference | Checkpoint disk size measured; GPU weight allocation **UNKNOWN / NOT MEASURED**. | At batch 8, normalized optical + SAR tensor payload is 6,451,200 B. Three float32 GAP outputs would total 73,728 B/batch. If all three 225×768 token outputs were resident, raw output payload alone would be 16,588,800 B/batch. | Components **UNKNOWN / NOT MEASURED**; combined historical PyTorch peak was 904,975,360 B. | Existing historical run completed under its environment. On the specified RTX 5060: **UNKNOWN** until measured there. |
| Larger CROMA batches | Same model identity if unchanged. | Inputs scale arithmetically at 806,400 B/area; outputs scale with areas and selected outputs. | Attention activations, allocator behavior and fragmentation need not scale linearly. | **UNKNOWN** for a changed batch; retain batch 8 unless measurements justify change. |
| Full 5,000-area token extraction | Same existing model for CROMA, if chosen. | One token family is 3.456 GB raw across the dataset; three are 10.368 GB. Bounded batches can be emitted and released. | Token path's measured peak and serialization cost **UNKNOWN / NOT MEASURED**. | A full all-family token array cannot reside together in 8 GB by payload alone. Bounded generation may fit, but is unproven and has no current consumer. |
| Future VLM | Model/checkpoint identity and weight residency **UNKNOWN / NOT MEASURED**. | Input resolution, token count, modality and output size **UNKNOWN / NOT MEASURED**. | Forward activations, KV cache, runtime workspace, dtype/quantization overhead and offload cost **UNKNOWN / NOT MEASURED**. | **UNKNOWN**. A compressed checkpoint fitting on disk, or nominal quantized weight size under 8 GB, is insufficient evidence. |

The historical `max_memory_allocated` counter omits unallocated-but-reserved CUDA memory and other processes. It is not a guaranteed capacity margin on an 8 GB card. Do not subtract the checkpoint's disk length from the GPU budget or infer a safe VLM size from the CROMA result.

## 4. RAM and access pattern on the 24 GB target

The feature builder streams the 5,000-area population **by shard**, but loads all 32 canonical optical/SAR arrays, physical vectors and targets for a shard into Python lists before processing CROMA batches of 8. Canonical optical+SAR payload for 32 areas is 25,804,800 B. Normalization makes float32 tensor copies; `torch.cat` and device transfer create additional transient copies, so 25.8 MB is not peak host usage. Inference outputs are moved to CPU NumPy arrays and retained only for the current shard before compressed writing. This is a bounded pattern, and no measured memory failure justifies refactoring it.

`load_cache` validates each shard checksum, appends all requested arrays to lists, then concatenates them into whole-population arrays. That temporarily holds shard arrays and concatenated copies; training then forms split matrices. The complete compressed feature cache is 44.3 MB and core float32 vectors are small, so eager loading is reasonable for this scientific predictor. It should **not** be copied as the read pattern for a future multi-GB raw-image or token store. The materializer reads, validates and concatenates one 32-area shard at a time; receipt and catalog generation also hold metadata structures. Their peak RSS is **UNKNOWN / NOT MEASURED**. The 55.9 MB catalog JSONL is an inventory cost, not tensor RAM, but whole-catalog loading may cost more than file length in Python objects; that peak is **UNKNOWN / NOT MEASURED**.

For future large representations: read only the requested sample/shard and modalities; bound workers, prefetch, batch, decoded arrays and in-flight outputs by measured RSS; release GPU/CPU intermediates after a shard; avoid multiplying float32 copies or loading 5,000 token tensors at once. These are execution requirements, not a mandate to rewrite current code.

## 5. Storage and lazy-generation policy

| Artifact | Classification | Reason / trigger |
|---|---|---|
| Native optical GeoTIFFs, SAR and reference source ZIPs | **PERSIST** as source / **REPRODUCIBILITY-ONLY** for unused copies | Preserve verified upstream data and hashes; load selected bands/areas lazily. No second canonical float32 corpus. |
| Reprojected and normalized imagery | **EPHEMERAL** | Specific preprocessing and model inputs can differ. Recreate per bounded batch. |
| `physical_62d`, joint CROMA GAP | **PERSIST** existing shared source shards | Frozen scientific inputs; receipts reference source arrays without duplication. |
| Optical/SAR GAP | **LAZY admission** of existing source arrays | Already present but do not make redundant typed stores. Admit with provenance only for a selected consumer. |
| `hybrid_830d` | **PERSIST** | Frozen predictor needs exact concatenation. It is not the default VLM input. |
| CROMA spatial tokens | **LAZY** | Generate only for a selected token-consuming adapter, grounding, or justified spatial analysis; start with bounded samples and one needed family. |
| CROMA and predictor checkpoints | **PERSIST / REPRODUCIBILITY-ONLY** | Retain frozen identity and reproducibility; load only for relevant execution. |
| Shard/sample receipts and manifests | **PERSIST** | Checksum, ordering, source, preprocessing and split traceability. |
| Representation catalog | **REPRODUCIBILITY-ONLY**, rebuildable | Local inventory of verified/missing rows; no implied need to copy it to a second store. |
| Evaluation predictions and reports | **REPRODUCIBILITY-ONLY** | Retain authoritative regression evidence and hashes. |
| Future VLM intermediates, captions/grounding outputs | **EPHEMERAL** by default; persist only with explicit evaluated consumer/provenance | No VLM task or artifact is selected. |

Temporal pairs/differences remain **LAZY and blocked** until T1/T2 identities, source hashes, spatial co-registration and split semantics are verified and a task needs them. Grounding outputs remain blocked on valid geometry frame/supervision and an evaluated grounding consumer; CROMA token positions alone do not establish grounding.

## 6. Shards, resume and cache reuse

Current source and hybrid design uses 157 `np.savez_compressed` shards: 156×32 plus 1×8. Sequential inference/writing is bounded by shard, and source/hybrid readback hashes a shard then opens it. Random sample access uses the manifest/receipt's shard and index, then decompresses that shard; it is not a per-sample random-access format. At current shard size and observed maximum of 283,814 B source or 100,480 B hybrid, that tradeoff is suitable for the scientific predictor and occasional sample lookup. Future VLM preprocessing can keep bounded batches independently of persistence shard size; a different size needs a measured I/O or memory problem first.

The Phase 2D.2 materializer validates source manifest identity, source SHA-256, ordered IDs/splits, required keys, shape, float32 dtype and finite values; existing hybrid reuse additionally checks receipt checksum and exact content contract. Writes use temporary files, fsync and atomic rename. A corrupt or mismatched final shard fails closed and requires explicit removal/regeneration, so partial final artifacts are not silently accepted. All 15,000 sample receipts bind artifact SHA-256 plus a distinct per-sample hash. The catalog admits only verified core rows.

**Future new extraction cache contract:** key and validate dataset/source version and source hashes, exact area/image identity and ordering, split, representation family/version, producer and preprocessing configuration, modality, spatial/temporal configuration, model/checkpoint identity when model-dependent, dimension, dtype, sample/shard checksum and completion status. Match all applicable fields before reuse; fail closed on missing or mismatched provenance. The historical `build_features` early reuse shortcut checks only area IDs before skipping a shard. Its completed downstream cache has stronger admission, receipts and catalog validation, but that shortcut must not be copied into a new extraction path. This is a small documented validation contract; the existing receipts/catalog already implement most of it, so no duplicate cache framework is warranted now.

## 7. Device and future VLM gate

Current Pipeline 3 commands already select CUDA/CPU with `--device`, CROMA batch with `--inference-batch-size`, shard size with `--shard-size`, and probe batch in training code. A global `ExecutionProfile` with device, dtype, workers, memory ceilings and offload policy would have no selected VLM or measured demand to drive its semantics. **No new runtime configuration is implemented.** A future adapter must declare its own execution requirements at the existing adapter/input boundary and be tested before admission:

1. State exact model/checkpoint, supported device and dtype/quantization, required source modalities or representations, image sizes, token/sequence bounds, and intended batch/prefetch/workers. Preserve source/checkpoint/preprocessing hashes.
2. Estimate weight residency, input and output payload, activations, KV cache and temporary workspace separately. Mark unknown estimates **UNKNOWN / NOT MEASURED**; include RAM for CPU offload and decoded/preprocessed copies.
3. On the target 8 GB VRAM / 24 GB RAM machine, run a bounded representative cold-load and warm-inference profile with actual selected inputs, then a worst-case permitted batch/sequence. Record process RSS peak, CUDA allocated **and reserved** peaks, total/free device memory, latency, offload and failure behavior. Avoid competing GPU workloads during the measurement.
4. Admit only a configuration that completes within **measured available** VRAM/RAM with operating-system and other-process headroom, and whose latency and output behavior meet the selected task's acceptance criteria. Reduce batch/sequence, change dtype, or explicitly evaluate offload only if measured results require it. No fixed numerical headroom or quantization promise is inferred from current CROMA measurements.

Until a concrete VLM and task exist, all VLM weight, activation, KV-cache, offload, throughput and fit claims are **UNKNOWN / NOT MEASURED**.

## 8. Implementation gaps and rejected optimizations

**Genuine gap:** future extraction must not inherit the historical ID-only early reuse shortcut. The required provenance-safe cache contract above is documented for the future extraction boundary. The current frozen 5,000-area materialization, receipts and catalog already validate the completed cache; modifying Pipeline 3 now would cross the task boundary and could change reproducibility. No present code gap requires an implementation change.

**Implemented changes:** this audit document only.

| Proposed change | Measured problem / cause | Decision and regression risk |
|---|---|---|
| Increase CROMA batch or change shard size | No measured throughput bottleneck attributable to those settings; new peak unknown. | Reject now. May increase activation memory or alter cache layout. Benchmark first. |
| Persist 5,000-area token families | No consumer or extraction benchmark; 3.456 GB raw per family. | Reject. Storage, validation and accidental full-load risk. |
| Save normalized optical/SAR tensors | No repeated consumer; source imagery remains available. | Reject. Adds 4.032 GB raw for both modalities, tied to current preprocessing. |
| Quantize/offload a future VLM | No selected model or memory profile. | Reject as current implementation. Potential accuracy/latency changes need task-specific evaluation. |
| Rewrite scientific cache loading as streaming | 44.3 MB compressed cache and no measured RAM failure. | Reject. Raises regression risk for frozen training/evaluation behavior. |
| Add global execution framework or second receipt system | Existing CLI, adapter boundary, receipts and catalog cover present use. | Reject. Adds configuration and provenance divergence without measured benefit. |

Any future optimization must identify a measured issue, root cause, proposed change, regression risk and expected measurable improvement before implementation.

## 9. Tests and regression verification

`python -m pytest -q --basetemp=.pytest_tmp_perf_storage_full` passed: **355 passed, 5 skipped, 0 failed**. The custom base directory avoids the existing inaccessible `.pytest_tmp` in this Windows worktree. A read-only `load_verified_receipts` call validated **15,000** typed receipts. A read-only `build_catalog` from the materialization manifest reproduced **5,000** complete core areas, **75,000** catalog rows (15,000 `VERIFIED`, 60,000 `MISSING`), and SHA-256 `0f5ce8feceff2fe8f2a75e1570b36edabf8a4716c57159243ffc6063eafcfd30`. `git diff --check` passed; the new document was also checked for whitespace because it is untracked.

Frozen dataset fingerprint `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`; split fingerprint `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`; materialization receipt SHA-256 `3deb01c15ecfa76f99098d54333bff0d0ce831b0d3af536d6b85cf9aea3deece`. The current source-cache manifest SHA-256 matches the materialization record, `3e37a44b0dd291b15f503a9e1b2b3c32ff9f80ac278fa414a8689ac740e101ed`. The [frozen final scientific fingerprint](../experiments/pipeline3_5000/final/fingerprint.json) recomputed from its canonical scientific basis is unchanged at `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`. Historical scientific result: MAE 4.1034286465 pp, RMSE 9.5873311030 pp, dominant-class accuracy 65.0% on the sealed 200-area test, from the existing [inference artifact](../artifacts/pipeline3_5000/representations/scientific_regression_phase2e/historical_checkpoint_inference.json). No new inference is implied.

**WHAT WE CAN SAFELY PERSIST NOW:** the existing validated core vectors, source imagery/checkpoints, manifests, receipts, catalog and scientific evidence.

**WHAT MUST REMAIN LAZY:** raw raster loading, canonical/normalized tensors, optional sensor GAP admission, and spatial-token generation for a justified consumer.

**WHAT MUST WAIT FOR A CONCRETE VLM:** adapter-specific input preprocessing, model load, dtype/quantization/offload decision, measured VRAM/RAM gate and task output storage.

**WHAT MUST WAIT FOR TEMPORAL WORK:** paired observations, co-registration and temporal representations.

**WHAT MUST WAIT FOR GROUNDING:** validated geometry frame and supervision, token-to-region use and grounding outputs.

PERFORMANCE / STORAGE FOUNDATION: READY
