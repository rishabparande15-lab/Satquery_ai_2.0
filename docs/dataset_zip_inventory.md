# Dataset ZIP Inventory

## `bigearthnet-v2-5000-20260911T162804Z-1-001.zip`

- Path: `E:\SatQuery_ai_2.0\datasets_2.0\bigearthnet-v2-5000-20260911T162804Z-1-001.zip`
- Size: 511,585,298 bytes
- SHA-256: `B3471E5650BD263367BCF4CC650405AB2981A86621579E21A1B1D08AF630C595`
- Integrity: PASS; exhaustive outer and nested CRC tests returned no bad member
- Outer entries: 6; uncompressed member bytes: 511,860,999
- Top level: `bigearthnet-v2-5000/`

It contains `metadata.parquet`, `selection.json`, a 5,000-member reference-map ZIP, and a 10,000-member S1 ZIP.

## `bigearthnet-v2-full-official-20260911T162841Z-1-031.zip`

- Path: `E:\SatQuery_ai_2.0\datasets_2.0\bigearthnet-v2-full-official-20260911T162841Z-1-031.zip`
- Size: 1,878,744,461 bytes
- SHA-256: `599D4AE1F446F8AC61F7693A70A85F1F37326A87EF7A2256F03310A902B91DDE`
- Integrity: PASS for the outer ZIP
- Outer entries: 7; uncompressed member bytes: 1,879,048,192
- Top level: `bigearthnet-v2-full-official/`

All seven members are exactly 268,435,456 bytes. They are non-contiguous `.tar.zst.parts` fragments: S2 parts `00128`, `00131`, `00147`, `00148`, `00158`, `00187`, and S1 part `00058`. Their first bytes are not the zstd frame magic `28 B5 2F FD`, which is expected for interior fragments but prevents standalone decompression. No manifest, metadata, annotation, or complete stream is present. This archive is therefore an intact fragment collection, not a usable full official dataset.

The machine-readable entry record is in [dataset_zip_inventory.json](dataset_zip_inventory.json).