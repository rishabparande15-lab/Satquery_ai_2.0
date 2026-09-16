# Phase 6R.2B Fresh 5,000-Area Scientific Reproduction

Date: 2026-09-16  
Repository commit: `2f7b241e4f4687d2aae263faf4b1d5c08eae9e5e`

## 1. Executive summary

**Final verdict: RED**

The isolated scientific Pipeline 3 model reproduced successfully. Fresh CPU
inference with the historical checkpoint completed all 5,000 cached
representations with zero failures or skips. The sealed 200-area test result
was MAE `4.10342865` percentage points, RMSE `9.58733110` percentage points,
and dominant-class accuracy `130/200 = 65.0%`. Fresh seed-17 training selected
epoch 54 and reproduced the same result to sub-micro numerical tolerance.

The overall verdict is RED because the required production connection is
broken. `POST /api/analyze` does not invoke the capability registry, cannot
discover the full-identity 5,000-area dataset, does not load the trained 830D
probe, and cannot trace its output into evidence, interpretation, and a
`TaskResult`. A real 5,000-dataset sample request returned HTTP 422. The
requested `225 x 19` trained output also does not exist: the frozen checkpoint
is a scene-level linear probe with output shape `[N, 19]`.

No scientific source, architecture, feature, label, split, evaluation
definition, historical checkpoint, or production dataset was modified.

## 2. Environment

Only `.venv-scientific\Scripts\python.exe` was used.

| Component | Observed value |
|---|---|
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| PyTorch | 2.14.0+cpu |
| pandas | 3.0.5 |
| pyarrow | 25.0.1 |
| rasterio | 1.4.4 |
| matplotlib | 3.11.2 |
| CUDA available | false |
| CUDA device count | 0 |
| PyTorch CUDA version | null |
| GPU name / memory | unavailable |
| CUDA tensor operation | not run; CUDA unavailable |

Execution therefore used CPU. PyTorch was not modified.

## 3. Dataset integrity

The independent exhaustive audit reopened the rasters and archive members and
passed:

- selected metadata areas: 5,000;
- complete S1 areas: 5,000;
- complete S2 areas: 5,000, comprising 60,000 TIFFs;
- complete reference maps: 5,000;
- complete multimodal areas: 5,000;
- missing optical areas: 0;
- duplicate area IDs and S1 identities: 0;
- duplicate whole-area or cross-area S2 content: 0;
- cross-split S2/reference duplicates and spatial overlaps: 0.

The audit checked native dimensions, uint16 S2 dtype, CRS, bounds, north-up
orientation, finite values, nodata/masks, valid pixels, identity
correspondence, and all twelve optical bands. Seven same-area cross-band
equality groups and 103 within-split duplicate reference groups remain
explicitly recorded; neither is cross-area S2 duplication or cross-split
leakage.

Dataset fingerprint:
`7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`

Result: exact historical match.

## 4. Split integrity

The fresh audit found 4,600 train, 200 validation, and 200 test areas. Split
intersections were empty. The split fingerprint freshly matched:

`2232ac5bc65d3ed20c6f39deb6037bb8e05410b247fd6c9e538eddf9feb86c`

Fresh training loaded exactly 4,600 train and 200 validation rows and reported
`test_loaded = false` and `test_used_for_selection = false`.

## 5. Checkpoint verification

Historical checkpoint:
`D:\Satquery_ai datasets\comparison\pipeline3-5000\models\baseline\hybrid.pt`

| Property | Verified value |
|---|---|
| File SHA-256 | `0694dd82d4a3c03663ed7128adb4be6df729178e07b4ff8a1ec4295ed912130f` |
| Model-state SHA-256 | `85d9390c66a887276db24a7cadb58c398770cde2e17488a1a9c42e16819da634` |
| Weight / bias shapes | `[19,830]` / `[19]` |
| Feature | hybrid |
| Input | physical 62D followed by joint CROMA GAP 768D |
| Outputs | 19 scene-level class proportions |
| Seed / selected epoch | 17 / 54 |
| Historical test flag | false |

The prompt's historical hash is the canonical model-state hash, distinct from
the serialized checkpoint file hash. Both matched their authoritative config.
The file was loaded read-only and was not overwritten.

## 6. Fresh 5,000 inference

The audit harness reused the production cache loader, probe loader, state hash,
prediction function, and evaluation function unchanged.

| Measurement | Result |
|---|---:|
| Areas attempted / completed | 5,000 / 5,000 |
| Failed / skipped | 0 / 0 |
| Physical matrix | `[5000,62]` |
| Joint CROMA GAP | `[5000,768]` |
| Hybrid matrix | `[5000,830]` |
| Predictions | `[5000,19]` |
| Cache validation/load | 0.580 s |
| Probe inference | 0.00313 s |
| Metric evaluation | 0.186 s |
| Total harness runtime | 0.829 s |
| Observed RSS increase | 96,641,024 bytes |
| Peak GPU memory | 0 |

This was fresh checkpoint inference over verified existing representations,
not a fresh 5,000-area CROMA recomputation. A controlled cache-bypass extraction
was performed separately. No area failed silently.

## 7. Historical-checkpoint metric reproduction

Only the sealed 200-area test split was evaluated.

| Metric | Historical | Fresh | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|
| MAE (pp) | 4.1034 | 4.10342865 | 0.00002865 | 0.000698% |
| RMSE (pp) | 9.5873 | 9.58733110 | 0.00003110 | 0.000324% |
| Accuracy | 65.0% | 65.0% | 0 | 0% |

Exact accuracy was `130/200`. The whole-area bootstrap 95% MAE interval was
`[3.83756011, 4.37412376]` pp. Per-class metrics are preserved in
`historical_checkpoint_inference.json`.

The maximum difference from the previously saved test predictions was
`6.2585e-7`; no metric definition, ordering, checkpoint, or stale prediction
mismatch was found.

## 8. Cache/artifact audit

The full inference used the existing cache at
`D:\Satquery_ai datasets\comparison\pipeline3-5000\scene_features`.
Its manifest identifies the exact dataset/split fingerprints, CROMA checkpoint
SHA-256 `0238d8...b63`, source revision
`59505a6bcadbf36ba20767270154bf9f3067c5e7`, 157 shards, and 5,000 rows. The
production loader recomputed and verified every shard hash and all unique area
identities.

One real test area was recomputed from raw rasters twice on CPU with cache
bypass. Physical features matched the cache exactly. Maximum fresh CPU versus
historical GPU-cache differences were:

- physical: `0`;
- optical GAP: `1.12504e-6`;
- SAR GAP: `6.55651e-7`;
- joint GAP: `9.53674e-7`.

The repeat CPU runs were exactly equal for all token and GAP tensors. No mixed
fresh/cached row entered either training or full checkpoint inference; the
full matrices came from the verified cache, while the bypass result was an
independent comparison only.

## 9. Fresh training

Fresh artifacts were written only under:

`D:\Satquery_ai datasets\comparison\pipeline3-5000\revalidation_6R2B`

The frozen configuration used seed 17, AdamW, learning rate 0.001, weight
decay 0.01, batch size 1024, maximum 60 epochs, patience 10, train-only
standardization, and the unchanged 4,600/200/200 split. Training took 84.33 s
on CPU. The hybrid curve contains all 60 train/validation points and selected
epoch 54 at validation loss `1.5265722275` and validation MAE `4.40641643` pp.

Fresh hybrid checkpoint:

- file SHA-256: `b6eadb5b08453e2a1d262767b350e918e4db56bbf0faae358ce0f9946295610a`;
- state SHA-256: `4fa4f16e9267087f5ef20bef30687808bc2918e1fd57be5f94aa376dadca1331`;
- configuration SHA-256: `d99d1726e7668a534496732ce27afe5f9d053350d43b5546b64bd07d22d540f1`.

The fresh state hash differs from the historical state because CPU retraining
produced parameter differences up to `3.8147e-6`; this was not hidden or
altered to force a hash match.

## 10. Fresh training test metrics

After epoch selection was frozen, the saved fresh checkpoint produced:

| Metric | Historical | Fresh training | Absolute difference |
|---|---:|---:|---:|
| MAE (pp) | 4.1034 | 4.10342872 | 0.00002872 |
| RMSE (pp) | 9.5873 | 9.58733139 | 0.00003139 |
| Accuracy | 65.0% | 65.0% (`130/200`) | 0 |

Bootstrap 95% MAE CI: `[3.83756040, 4.37412382]` pp. Fresh-training versus
historical-checkpoint test predictions differed by at most `9.53674e-7`.
Metrics recomputed from the saved prediction file matched the report exactly;
maximum difference was `0.0`.

Operational disclosure: an initial identical final-scoring command completed
its in-memory computation but was denied before writing any output to `D:`.
No result was observed or used and no retraining occurred. The unchanged,
already-frozen command was rerun with write permission solely to persist the
metrics, predictions, fingerprint, and receipt.

## 11. Leakage audit

| Category | Status | Evidence |
|---|---|---|
| Data | PASS | Empty split intersections; no cross-split raster duplicates or positive spatial overlaps |
| Label | PASS | Reference values are separate targets; the 62D extractor accepts only optical/SAR; reference-only perturbation changed features by 0 |
| Feature | PASS | Physical and CROMA features derive from input rasters; standardization is train-only |
| Training | PASS | Test rows were not loaded for training/selection; historical config records no test-based selection |
| Cache | PASS | All 157 hashes verified; identities complete; raw bypass matched within measured CPU/GPU tolerance |

No demonstrated material scientific leakage was found.

## 12. CROMA validation

The real cache-bypass sample produced:

- optical encodings `[1,225,768]`;
- SAR encodings `[1,225,768]`;
- joint encodings `[1,225,768]`;
- optical/SAR/joint GAP `[1,768]`;
- finite tensors and the expected 15 by 15 token grid;
- exact repeat-run equality, maximum absolute difference `0.0`.

The production model's patch flatten/reshape contract is row-major. No token
truncation was observed. First and repeated CPU passes took 1.124 s and 1.042 s.

## 13. Physical feature validation

The production vector contains exactly 62 ordered values: mean, standard
deviation, minimum, and maximum for each of 12 optical bands and VV/VH,
followed by NDVI, NDWI, MNDWI, NDBI, VV-minus-VH, and VH-over-VV means. Optical
ordering was B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12;
SAR ordering was VV, VH.

Source inspection shows that `LocalRasterFeatureProvider.extract` accepts only
optical arrays, SAR arrays, and their band orders. Runtime extraction remained
bit-identical after changing only a reference-map value. Therefore no reference
map, target statistic, or class frequency enters the 62D vector.

## 14. End-to-end connectivity

**FAIL.** The requested real 5,000-area trace cannot be completed.

The scientific branch is valid through:

`S2/S1 -> preprocessing -> CROMA/physical -> 830D hybrid -> scene-level 19D probe`.

It does not continue through the requested `225 x 19` output, region/scene
aggregation, evidence, interpretation, and `TaskResult`. The trained model is a
scene-level `[830] -> [19]` probe, and no adapter connects it to the production
evidence or capability system. The independent direct capability-registry path
passed two real-data tests on the legacy three-sample fixture, but that path
uses a deterministic untrained 192D `HybridFusion`, not this 830D trained
checkpoint. It cannot serve as proof of Pipeline 3 model connectivity.

## 15. Evidence/provenance

The cached scientific artifacts retain dataset, split, CROMA checkpoint, source
revision, feature configuration, and area identities. The new prediction and
training artifacts have checksums and a one-time test receipt.

The Phase 6R.2A limitation remains: the restored 4,000 optical areas have
verified acquisition receipts, while the original 1,000 have none. The exact
dataset and split fingerprints match, but a historical 5,000-receipt state is
not claimed.

No provenance attachment from the trained 830D probe into API evidence or a
`TaskResult` exists; that connectivity property fails.

## 16. API connectivity

**FAIL.** A live request used the real selected area
`S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11`.

- HTTP status: 422;
- response schema: 2.0;
- response status: rejected;
- runtime reported by response: 1.181 s;
- warning: local sample discovery failed;
- prediction: null;
- confidence: uncalibrated, model confidence null;
- evidence and interpretation: unavailable.

The API's non-strict discovery reduces full patch identities to row/column
suffixes, causing duplicate-band collisions in the 1,000-area root. Source
inspection also proves that `/api/analyze` directly calls legacy
`analysis_engine.run_analysis`; it does not traverse the capability registry.
Even a successful legacy request would expose representation-only untrained
fusion and no trained task prediction.

## 17. Performance

Measured fresh times were:

- exhaustive dataset audit: 412.51 s, dominated by raster/archive I/O and hashing;
- verified cache load: 0.580 s;
- full 5,000 probe inference: 0.00313 s;
- historical-checkpoint metric evaluation: 0.186 s;
- single-area raster I/O/resampling: 0.0437 s;
- single-area physical features: 0.00474 s;
- single-area CPU CROMA: 1.124 s first pass;
- fresh training: 84.33 s;
- persisted fresh-test scoring: 1.352 s.

The dominant scientific bottleneck is CROMA representation extraction. The
historical cache manifest records 910.73 s and 904,975,360 bytes peak GPU
memory for its original build, but that is historical—not a fresh Phase 6R.2B
measurement. Full current-host CPU CROMA recomputation was not performed; the
controlled measured bypass and cache verification were used instead. No
optimization was attempted.

## 18. Failure handling

Twenty-five targeted temporary-fixture checks passed:

- every missing optical band fails explicitly;
- missing VV or VH fails explicitly;
- missing reference fails explicitly;
- wrong dimensions, CRS, grid, NaN, Inf, and corrupt TIFFs fail explicitly;
- wrong CROMA token shape and invalid phase-boundary contracts fail explicitly;
- a corrupted feature shard is rejected by checksum;
- an incompatible checkpoint is rejected by its contract.

No production raster or checkpoint was modified. The API's real-dataset
failure was explicit (HTTP 422), not a silent substitution or misleading
prediction.

## 19. Reproducibility

Machine-readable evidence is under
`experiments/pipeline3_5000/revalidation_6R2B/` and the new training directory.

Scientific revalidation fingerprint:

`37a54e07bf238c7987e223112953c4870610a1e9746166a3cd80419b435ec01b`

The fingerprint covers repository commit, environment-bound dataset/split and
checkpoint identities, configuration, seed/epoch, historical and fresh test
metrics, cache identity and bypass tolerances, saved-prediction recomputation,
leakage results, and the API connectivity failure. Saved prediction metrics
recompute exactly.

## 20. Limitations

1. CUDA is unavailable; all fresh scientific execution used CPU.
2. The full 5,000 representation matrix was verified from the existing cache;
   only a controlled real-area subset was freshly recomputed from raw rasters.
3. Original 1,000-area acquisition receipts remain unavailable.
4. The serialized fresh checkpoint and state hashes differ slightly from the
   historical checkpoint despite sub-micro prediction agreement.
5. The first frozen test-scoring attempt encountered a post-computation write
   permission failure and was identically rerun only to persist evidence.
6. The production API/capability/evidence path is not connected to the trained
   830D Pipeline 3 probe and cannot consume the 5,000-area root.

## 21. Final verdict

**RED**

Dataset integrity, historical-checkpoint metrics, fresh training, saved metric
recomputation, CROMA repeatability, physical-feature independence, and leakage
checks passed. Nevertheless, the phase definition requires genuine end-to-end
connectivity. The real API rejected the dataset sample, bypasses the capability
registry, and has no path from the trained 830D checkpoint to evidence,
interpretation, or `TaskResult`. This is a critical connection defect.

Phase 6R.2B stops here. No VQA, captioning, grounding, CDVQA, optimization,
commit, or push was performed.
