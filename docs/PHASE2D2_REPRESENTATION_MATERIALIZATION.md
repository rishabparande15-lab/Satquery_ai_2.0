# Phase 2D.2 Representation Materialization

## 1. Dataset identity

The only admitted population is the authoritative 5,000-area Pipeline 3 dataset: 4,600 train, 200 validation, and 200 test areas. Dataset fingerprint: `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`. Split fingerprint: `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`. Ordering comes from `experiments/pipeline3_5000/dataset_manifest.json`, never filesystem order.

## 2. Source artifacts

Machine inventory found the complete historical cache at `D:\Satquery_ai datasets\comparison\pipeline3-5000\scene_features`: one manifest and 157 NPZ shards (44,280,954 bytes). All shard hashes, 5,000 ordered identities, split values, shapes, dtypes, and finite values were verified. The cache manifest SHA-256 is `3e37a44b0dd291b15f503a9e1b2b3c32ff9f80ac278fa414a8689ac740e101ed`.

The frozen predictor is `hybrid.pt`, SHA-256 `0694dd82d4a3c03663ed7128adb4be6df729178e07b4ff8a1ec4295ed912130f`, state-dict SHA-256 `85d9390c66a887276db24a7cadb58c398770cde2e17488a1a9c42e16819da634`.

## 3. Representation types

The core set is `physical_62d`, `joint_croma_gap_768d`, and `hybrid_830d`. Optional catalog families remain `MISSING`; token tensors, separate optical/SAR GAPs, and raw rasters were not duplicated.

## 4. Materialization strategy

`src.representation_materialization` reuses physical and joint CROMA arrays in-place after validation. It creates only `hybrid = concatenate(physical, joint)` in the exact evaluated order. It does not run new preprocessing, CROMA inference, training, or evaluation logic. The command is:

```powershell
python -m scripts.materialize_pipeline3_representations --source-cache-manifest "D:\Satquery_ai datasets\comparison\pipeline3-5000\scene_features\manifest.json"
```

## 5. Sharding

The established bounded shard size of 32 is retained: 156 shards contain 32 samples and the final shard contains 8. Each shard receipt records ID, global start/end indexes, ordered image IDs/splits, artifact shape/dtype/hash, producer, model, preprocessing, and dataset identity.

## 6. Resume strategy

Existing hybrid shards are reused only after artifact checksum, keys, exact ordered IDs/splits, shape, dtype, and finiteness checks. Invalid or partial final artifacts are never overwritten silently; explicit removal/regeneration is required. New NPZ and JSON/JSONL files use temporary files, flush/fsync, and atomic rename.

## 7. Receipt schema

`representation_shard_receipt_v1` produces one typed receipt for every `(image_id, representation_type, variant)` association. There are 15,000 unique `default`-variant receipts. Each records the shard, array key, sample index, image/split identity, shape, dtype, artifact checksum, distinct sample checksum, dataset/checkpoint/preprocessing provenance, and producer. The trusted `src.receipt_catalog.validate_materialization_receipts` boundary performs deterministic lookup validation and rejects duplicates or invalid artifacts.

## 8. Checksum strategy

Artifacts use file SHA-256. Samples use `SHA256(dtype || canonical_shape || C-order bytes)`. The receipt catalog SHA-256 is `3deb01c15ecfa76f99098d54333bff0d0ce831b0d3af536d6b85cf9aea3deece`. A shard checksum is explicitly marked as an artifact checksum and is never described as a per-sample checksum.

## 9. CROMA checkpoint identity

Model: official CROMA base, source revision `59505a6bcadbf36ba20767270154bf9f3067c5e7`, checkpoint SHA-256 `0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63`, loader `src.croma_adapter.CROMAAdapter`, modality `both`, image resolution 120. The source cache was produced with `torch.inference_mode()` and records 225 x 768 token semantics and 768-D GAP outputs.

## 10. Physical feature identity

The 62-D vectors are the exact existing output of `src.gee_features.LocalRasterFeatureProvider.extract`. No band order, formula, statistic, aggregation, or normalization changed.

## 11. Hybrid representation

The 830-D vector is float32 `physical[0:62]` followed by `joint_croma_gap[62:830]`. It is the input consumed by the frozen evaluated probe, not the separate 192-D `HybridFusion` infrastructure.

## 12. Storage requirements

Raw float32 estimates were 1,240,000 bytes physical, 15,360,000 bytes joint CROMA GAP, and 16,600,000 bytes hybrid. Physical/CROMA incurred zero new storage because the verified 44,280,954-byte cache was reused. Compressed hybrid shards use 15,671,729 bytes. Manifests, shard receipts, per-image receipts, and scientific regression artifacts bring the Phase 2D.2 materialization tree to 33,829,002 bytes. Raw optical/SAR data are referenced, not copied.

## 13. Performance

Historical physical+CROMA extraction took 910.732 s (5.49 areas/s, 329.4 areas/min) with 904,975,360 bytes peak GPU allocation; its manifest did not separate physical and CROMA stage times. Phase 2D.2 first-run verification/materialization took 27.143 s: 1.276 s source validation and 2.709 s hybrid assembly, with the remainder used for hashing, receipts, and atomic persistence. Catalog generation/validation took 3.586 s and Phase 2C link regeneration took 28.619 s. A full resume took 23.075 s and reused all 157 hybrid shards. Scientific CPU regression took 0.730 s with an observed RSS delta of 95,154,176 bytes and zero GPU use. Materialization host peak RSS was not instrumented; no GPU was used during Phase 2D.2 admission/assembly.

## 14. Determinism

The second run reused all shards and reproduced the exact receipt bytes/SHA-256 (`3deb…eece`). Hybrid shard bytes and sample hashes were unchanged. Operational duration fields in the top-level run manifest are measurements and are expected to change; scientific artifacts, ordering, metadata receipts, and catalog rows are deterministic.

## 15. Failure handling

Missing/corrupt shards, checksum changes, missing arrays, identity/order/split mismatches, wrong shapes/dtypes, non-finite values, duplicate typed identities, invalid sample indexes, receipt tampering, and interrupted final artifacts fail explicitly. The completed real run recorded zero sample or shard failures.

## 16. Coverage

All 5,000 areas have all three verified core representations: train 4,600/type, validation 200/type, and test 200/type. `representation_catalog_v1` contains 75,000 rows: 15,000 `VERIFIED` core rows and 60,000 truthful optional `MISSING` rows. Its current fingerprint is `0f5ce8feceff2fe8f2a75e1570b36edabf8a4716c57159243ffc6063eafcfd30` after the core types were explicitly labeled `optical_sar` for Phase 2E input validation.
