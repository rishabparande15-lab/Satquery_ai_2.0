# Phase 2D.2 Materialization Audit

## 1. Expected 5,000 areas

The authoritative manifest and cache both resolve exactly 5,000 unique ordered IDs with the required 4,600/200/200 split and frozen dataset/split fingerprints.

## 2. Successfully materialized areas

5,000 areas were verified. Existing physical and CROMA GAP vectors were admitted; 157 hybrid shards were generated and then successfully reused on a repeat run.

## 3. Failed areas

Zero. No area was skipped or substituted.

## 4. Core representation coverage

- `physical_62d`: 5,000 verified (4,600 train, 200 validation, 200 test)
- `joint_croma_gap_768d`: 5,000 verified (4,600/200/200)
- `hybrid_830d`: 5,000 verified (4,600/200/200)
- Images with complete core set: 5,000

## 5. Optional representation coverage

The remaining 12 catalog families have 60,000 `MISSING` rows. They were not generated merely because the schema supports them.

## 6. Receipt verification

15,000 typed, unique image/type/default-variant receipts passed artifact checksum, deterministic sample-index, ordered identity, split, shape, dtype, and sample checksum validation. Receipt catalog SHA-256: `3deb01c15ecfa76f99098d54333bff0d0ce831b0d3af536d6b85cf9aea3deece`.

## 7. Catalog fingerprint

`representation_catalog_v1` has 75,000 records and SHA-256 `0f5ce8feceff2fe8f2a75e1570b36edabf8a4716c57159243ffc6063eafcfd30`: 15,000 `VERIFIED`, 60,000 `MISSING`, and no invalid/expired/provenance/split mismatch rows. The fingerprint supersedes the initial Phase 2D.2 value after canonical core modality metadata was corrected to `optical_sar`; representation bytes and receipts did not change.

## 8. Representation-linking before/after

Both runs covered 101,571 annotations. Missing representation cases changed from 101,571 to 0; verified image-to-representation links changed from 0 to 101,571; provenance failures remained 0; split failures remained 0. Source geometry remains unmapped: 25,303 spatial annotations still report `SPATIAL_LINK_UNAVAILABLE`, with zero bbox/token and point/token links. New link artifact SHA-256: `ee8cf991bc948ad442592a7e3fb2d13477af8b94604053ae66bc52a0ee23e0c7`.

## 9. Scientific regression

Fresh CPU inference over the verified physical + joint GAP representation and unchanged `hybrid.pt` produced 65.0% accuracy, 4.1034286465 pp MAE, and 9.5873311030 pp RMSE. Differences from authoritative stored values are respectively 0, -0.0000000270 pp, and -0.0000001093 pp, within established floating-point reproducibility tolerance. The checkpoint file/state hashes match exactly. No retraining occurred.

## 10. 61_39 isolation verification

The `61_39` artifacts remain under the independent pixel/CROMA probe roots. `61_39` is not an authoritative manifest ID and therefore fails image identity/dataset provenance admission. No 61_39 artifact or receipt appears in the 15,000 verified core receipts.

## 11. Known limitations

BigEarthNet.txt source coordinates remain unresolved and unmapped. The cache is local and its source artifact paths are machine-local provenance. Optional representations are not materialized. The top-level run manifest contains variable performance timing; scientific artifacts and canonical catalog/receipt bytes are deterministic.

## 12. Remaining blockers

There is no Phase 2D.2 core-materialization blocker. Phase 2E may begin only within its separately defined scope; this result does not authorize VQA, captioning, learned grounding, temporal reasoning, CDVQA, or agent planning.
