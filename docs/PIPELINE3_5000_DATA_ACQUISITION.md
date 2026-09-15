# Pipeline 3 Exact 5,000-Area S2 Acquisition

Date: 2026-09-14

## 1. Original blocker

The canonical selected archive contained 5,000 S1 areas, 5,000 reference maps,
and 5,000 metadata rows, but no embedded S2 TIFFs. The project-designated
`raw-1000` directory supplied complete S2 for only 1,000 selected IDs. The
remaining 4,000 IDs were all members of the frozen training split. Pipeline 3
training remained stopped while that exact gap existed.

## 2. Exact missing IDs

The immutable pre-acquisition list is
`experiments/pipeline3_5000/s2_acquisition/missing_before.json`. It contains all
4,000 exact area IDs, S2 identities, split assignments, required bands, and
expected spatial contracts. All are train IDs. This file is the exact ID list;
it is not regenerated from directory enumeration after acquisition.

## 3. Data source and location

Recovered data is stored outside Git at:

`D:\Satquery_ai datasets\comparison\raw-5000-additional\BigEarthNet-S2`

It supplements, without modifying or duplicating, the validated original root:

`D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2`

The trusted discovery scan covered `D:\Satquery_ai datasets` (including
`comparison-work` and extracted/raw subtrees) and `E:\SatQuery_ai_2.0`
(including `datasets_2.0`). The three concrete S2 candidate roots and every
inspected archive are enumerated in `acquisition_status.json`; inaccessible or
unrelated operating-system locations were not searched.

Native optical arrays were read by exact patch key from the project-pinned
`hackelle/BigEarthNetV2-LMDB` pre-conversion mirror at immutable revision
`118d1b6285c080ba8e4078414e1b8a243b18c9bd`. The source object is
`BENv2.lmdb/data.mdb`, with pinned logical size 155,372,892,160 bytes. The
strict range reader and conversion procedure come from the project's documented
reference repository `raviasha/Sat_Query`, inspected at commit
`ac5541086c637ece2e9fd3c0a79cfa8bea23304d`.

This is the same pinned route used for the established 1,000-area dataset. The
mirror is explicitly an unofficial pre-conversion representation, not a claim
that generated TIFF container bytes equal the publisher's original TIFF bytes.
The official BigEarthNet v2 S2 archive remains available from
`https://bigearth.net/` / Zenodo record 10891137 as a roughly 59 GiB archive.

## 4. Acquisition and recovery process

`scripts/acquire_pipeline3_5000_s2.py` reads no selection parameters. It accepts
only IDs that the authoritative 5,000-area manifest marked as missing, requests
each exact LMDB key with HTTP byte ranges, decodes the safetensors record, and
requires the exact 12-band set. It writes one isolated directory per area and
atomically publishes it only after round-trip validation succeeds.

The recovered TIFF georeferencing is derived from the already verified exact
reference-map spatial contract for the same manifest row. Every area stores a
local `.satquery-acquisition.json` receipt containing the source revision,
record key, source-record SHA-256, per-band file SHA-256, sizes, geometry, and
validation result. The procedure is resumable and refuses to overwrite an
unverified destination.

Recovery result: 4,000/4,000 requested records acquired, zero failed, zero
unresolved. The new root contains 48,000 TIFFs and 4,000 receipts, totaling
688,557,891 bytes. No raw imagery or archive is committed to Git.

## 5. Validation process

The downloader's checks were treated as acquisition checks, not final proof.
After recovery, `scripts/audit_pipeline3_5000.py` independently reopened and
hashed all 60,000 S2 TIFFs across the original and additional roots, alongside
all 10,000 S1 members and 5,000 reference maps. It rebuilt the authoritative
manifest from the frozen archive metadata and split rather than from folders.

The completed discovery evidence is in:

- `matched_s2.json`: 5,000 exact selected matches with per-band hashes, sizes,
  dimensions, resolution, CRS/bounds, masks, and whole-area content hashes;
- `unmatched_candidates.json`: all rejected candidates and archive findings;
- `duplicates.json`: path, area, content, and split duplicate classification;
- `acquisition_status.json`: final counts, roots, and inspected archives.

All four files are under `experiments/pipeline3_5000/s2_acquisition/`.

## 6. Band validation

Every selected S2 area has exactly B01, B02, B03, B04, B05, B06, B07, B08,
B8A, B09, B11, and B12. Native dimensions are 20x20 for B01/B09, 60x60 for
B05/B06/B07/B8A/B11/B12, and 120x120 for B02/B03/B04/B08. All acquired arrays
are single-channel uint16. No missing, extra, ambiguously named, or duplicated
band path remains.

## 7. Raster validation

All selected optical files are readable, single-channel, finite, north-up, and
use the expected native dimensions and 10/20/60 m resolutions. CRS and bounds
are present. Nodata is 0 for the reconstructed optical TIFFs; each mask agrees
with nodata and every band has at least one valid pixel. The auditor records
dtype, nodata, mask flags, valid-pixel count/fraction, orientation, CRS, bounds,
resolution, dimensions, and SHA-256.

## 8. S1/S2/reference correspondence

Area identity must equal the exact `patch_id`; filenames alone are not accepted.
For every area, every S2 band has the same CRS and outer bounds as its canonical
reference map. Native multi-resolution grids are expected, and Pipeline 3's
existing B02 120x120 alignment relationship is unchanged. All S1 VV/VH pairs
remain reference-aligned. No S1 or reference bytes were modified.

## 9. Checksums and provenance

The selected archive remains 511,585,298 bytes with SHA-256
`b3471e5650bd263367bcf4cc650405ab2981a86621579e21a1b1d08af630c595`.
Every recovered source record and generated TIFF has a SHA-256 receipt in the
external data root; every selected S2 band hash is also reproduced in
`matched_s2.json`. The compact whole-area hash is SHA-256 over ordered band name
and band digest pairs, so mutable paths are not dataset identity.

## 10. Duplicate handling

There are zero duplicate area/band paths, zero duplicate whole-area S2 content
groups, zero cross-area S2 content groups, and zero cross-split S2 content
groups. Seven same-area cross-band byte-equality groups are retained and
documented: they occur in four low-reflectance training patches where distinct
bands contain the same valid constant values. They are separate source band
keys with valid masks, not reused files, cross-area copies, or leakage.

The pre-existing 103 identical reference-map groups and 138 positive spatial
overlap pairs remain within train only. No reference or split was changed.

## 11. Split preservation

The authoritative split is unchanged at 4,600 train, 200 validation, and 200
test. Its canonical SHA-256 remains
`2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`.
All split intersections are empty. Acquisition copied no test data into the
additional training root because all 4,000 requested IDs are train members.

## 12. Final counts

| Requirement | Final count |
|---|---:|
| Unique selected manifest rows | 5,000 |
| Valid S1 | 5,000 |
| Valid 12-band S2 | 5,000 |
| Valid reference maps | 5,000 |
| Complete metadata | 5,000 |
| Complete multimodal | 5,000 |
| Train / validation / test | 4,600 / 200 / 200 |
| Unresolved selected IDs | 0 |

## 13. Dataset fingerprint

The established auditor passed twice with the same normalized dataset-gate
fingerprint:

`7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`

This is a dataset-gate fingerprint, not a model fingerprint. No training,
feature fitting, hyperparameter tuning, or test evaluation ran in Phase 6A.9.

## 14. Remaining issues

The scientific data gate is complete. The provenance limitation is explicit:
the acquired 4,000 use exact native arrays from the project-pinned unofficial
LMDB mirror, placed into GeoTIFF containers using canonical reference-map
georeferencing. A future provenance-hardening exercise may compare all arrays
against the publisher's 63,251,710,377-byte official S2 archive, but that is not
required to repeat the project's established pinned acquisition contract.

Final Phase 6A.9 status: **READY FOR 5,000-AREA TRAINING.** Training remains a
separate, not-yet-run phase.
