# Phase 6R.2A S2 Data Restoration

Date: 2026-09-16  
Scope: data restoration and independent audit only

## Final verdict

**YELLOW**

The exact missing 4,000 S2 areas were found locally and verified. The combined
dataset now has 5,000 complete multimodal areas, the expected `4,600 / 200 /
200` split, and matching dataset and split fingerprints.

The verdict is not GREEN because the original 1,000-area S2 root contains no
`.satquery-acquisition.json` receipts. The restored 4,000 areas have complete
receipts, but the requested historical state of 5,000 area-level receipts
cannot be demonstrated from the available files.

No model, labels, split, scientific source code, training, inference, or metric
evaluation was run or changed.

## Why restoration was required

Phase 6R.2 found 5,000 metadata rows, 5,000 S1 areas, 5,000 references, but
only 1,000 complete optical S2 areas. The missing population was the exact
4,000 additional train areas. The immutable source list was
`experiments/pipeline3_5000/s2_acquisition/missing_before.json`.

The required exact-ID list was generated at
`experiments/pipeline3_5000/s2_acquisition/missing_s2_area_ids.txt` and
contains 4,000 unique IDs.

Per-area matching evidence is retained in `missing_before.json` (area ID,
S2 identity, frozen split, expected bands and spatial contract) and
`matched_s2.json` (resolved destination, per-band checksum/size and raster
metadata, plus source-record identity). Restored paths follow
`<root>/<tile>/<area_id>/<area_id>_<band>.tif`; each receipt's LMDB record key
is the exact area ID.

## Exact local source

The missing data was already present locally at:

`D:\Satquery_ai datasets\comparison\raw-5000-additional\BigEarthNet-S2`

The 4,000 receipts identify the source as:

- dataset: `hackelle/BigEarthNetV2-LMDB`
- immutable revision: `118d1b6285c080ba8e4078414e1b8a243b18c9bd`
- object: `BENv2.lmdb/data.mdb`
- acquisition: strict HTTP byte-range reads of exact LMDB area keys
- record key: exact area ID
- representation: project-pinned unofficial pre-conversion LMDB mirror

This source and revision are documented in
`scripts/acquire_pipeline3_5000_s2.py` and
`docs/PIPELINE3_5000_DATA_ACQUISITION.md`. No download was performed because
the exact restored files were already available locally.

## Counts before and after

| Item | Before | After |
|---|---:|---:|
| Selected areas | 5,000 | 5,000 |
| Complete S2 areas | 1,000 | 5,000 |
| Missing S2 areas | 4,000 | 0 |
| S2 TIFFs | 12,000 | 60,000 |
| Restored S2 TIFFs | 0 | 48,000 |
| Complete multimodal areas | 1,000 | 5,000 |
| S1 areas | 5,000 | 5,000 |
| Reference maps | 5,000 | 5,000 |
| Train / validation / test | 4,600 / 200 / 200 | 4,600 / 200 / 200 |

The original 1,000-area root was not modified. The restored root contains
exactly 4,000 area directories, 48,000 TIFFs, and 4,000 receipts.

## Restoration verification

Every restored receipt has an area ID in the immutable 4,000-ID list, a
matching S2 identity and record key, `split = train`, all twelve expected
bands, and the pinned source revision.

| Check | Result |
|---|---:|
| Receipts | 4,000 |
| Unique receipt area IDs | 4,000 |
| TIFF paths checked | 48,000 |
| Receipt errors | 0 |
| Receipt ID differences | 0 |
| Source revision differences | 0 |
| Receipt size/checksum differences | 0 |
| Absolute band paths in receipts | 0 |
| Stale or wrong-source restored receipts | 0 |

The final acquisition status is `matched_valid_s2_count = 5000`,
`recovered_s2_count = 4000`, `unresolved_selected_count = 0`, and
`completeness_gate_ready = true`. Three unrelated small-sample candidates were
explicitly rejected because their area IDs are outside the frozen 5,000-area
manifest; none entered the restored population.

The independent full audit reopened both S2 roots and reported:

- `status = ready_for_training`;
- complete optical S2 areas: 5,000;
- complete S1 areas: 5,000;
- complete reference areas: 5,000;
- complete metadata areas: 5,000;
- complete multimodal areas: 5,000;
- missing optical areas: train `0`, validation `0`, test `0`.

It verified exact B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, and
B12 band sets; native dimensions; uint16 dtype; CRS; bounds; north-up
orientation; finite values; nodata/masks; valid pixels; and S1/S2/reference
identity correspondence.

## Duplicate and split checks

| Check | Result |
|---|---:|
| Duplicate area IDs | 0 |
| Duplicate S1 identities | 0 |
| Duplicate S1 content groups | 0 |
| Duplicate reference cross-split groups | 0 |
| Duplicate S2 cross-area groups | 0 |
| Duplicate S2 cross-split groups | 0 |
| Cross-split spatial overlaps | 0 |
| Split intersections | empty |

Seven same-area, cross-band S2 equality groups were recorded. They are not
cross-area or cross-split duplication. The audit also recorded 103 duplicate
reference-content groups, with zero cross-split groups.

## Fingerprints

Dataset fingerprint:

`7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`

Split fingerprint:

`2232ac5bc65d3ed20c6f39deb6037bb8e05410b247fd6c9e538eddf9feb86c`

Both match the authoritative historical values. The selected archive also
matched its frozen SHA-256:

`b3471e5650bd263367bcf4cc650405ab2981a86621579e21a1b1d08af630c595`

The fingerprint artifact reports `status = ready_for_training` and
`training_started = false`.

The split fingerprint above was read from the authoritative split/audit
artifacts and freshly recomputed; it was not reconstructed from the value in
the phase prompt.

## Provenance limitation

The restored population has 4,000 area receipts with per-band checksums,
source record keys, source record SHA-256 values, source revision, split, and
raster metadata. All 48,000 receipt band entries match the current file sizes
and SHA-256 values, and receipt filenames are relative rather than tied to an
absolute local root. The original 1,000-area root contains 12,000 TIFFs but zero
acquisition receipts. Its raster integrity is covered by the independent full
audit, but its source-record provenance cannot be reconstructed from receipts
in the current filesystem.

This is the sole material reason for the YELLOW verdict.

## Discrepancy note

The first acquisition-audit invocation pointed at the restored root while the
frozen pre-restoration manifest still marked those rows `s2_valid = false`; it
therefore reported 1,000 valid rows. That was an audit-state mismatch, not a
data failure. The independent full audit reopened both roots and reported
5,000 valid rows.

## Files and stop condition

Created:

- `experiments/pipeline3_5000/s2_acquisition/missing_s2_area_ids.txt`
- `docs/PIPELINE3_5000_6R2A_DATA_RESTORATION.md`

Updated audit artifacts:

- `experiments/pipeline3_5000/audit_summary.json`
- `experiments/pipeline3_5000/dataset_manifest.json`
- `experiments/pipeline3_5000/fingerprint.json`

No model or scientific execution command was run. Phase 6R.2A stops here.
Training, inference, evaluation, VQA, captioning, and change detection remain
outside this phase.
