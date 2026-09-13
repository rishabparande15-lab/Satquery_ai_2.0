# Pass 4 Dataset Quality Audit

## Scope

The outer ZIP directory listings and CRC checks were exhaustive. The selected archive's two nested ZIP listings and CRC checks were also exhaustive: 5,000 reference-map TIFF members and 10,000 S1 TIFF members. Metadata parquet was read completely: 5,000 rows and 8 columns. TIFF decoding was sampled at 15 positions per nested archive, including beginning, quartiles, middle, and end positions, with full pixel decode through Pillow.

The seven members of the official archive were exhaustively listed and CRC-checked, but they are non-contiguous fragments. No claim is made that the missing portions of the official dataset are clean or available.

## Selected 5,000-area archive

| Check | Result |
|---|---|
| Outer ZIP integrity | PASS |
| Nested ZIP integrity | PASS |
| Metadata rows | 5,000 |
| Unique patch IDs | 5,000; no duplicates observed |
| Unique S1 identities | 5,000; no duplicates observed |
| Split counts | train 4,600; validation 200; test 200 |
| Countries | 10, exactly 500 each |
| Labels | 19 unique CORINE-style labels; no null label lists |
| S1 files | 10,000, exactly 5,000 VH and 5,000 VV |
| Reference maps | 5,000 |
| VQA/caption/box/mask/temporal annotations | Not present |
| Sampled S1 TIFF decode | 15/15 passed; 120x120, float32, single band |
| Sampled reference-map decode | 15/15 passed; 120x120, uint16, single band |
| Sampled geospatial tags | WGS 84 / UTM, 10 m pixel scale; zones varied by location |
| NaN/Inf audit | Not exhaustively evaluated because the available NumPy installation is binary-incompatible with Python 3.14; sampled Pillow extrema were finite |

The metadata flags `contains_seasonal_snow` and `contains_cloud_or_shadow` are false for all 5,000 rows. This is metadata evidence, not an independent pixel-level cloud audit.

## Official fragment archive

The outer ZIP is intact, but it contains only seven 256 MiB parts. There is no metadata, manifest, annotation, complete S1/S2 stream, or usable sample ID inventory. Consequently, usable sample count is 0 from this archive alone. The parts cannot be classified as corrupt merely because interior fragments lack a zstd frame header; they are incomplete distribution fragments. Reassembly requires the missing parts and an authoritative ordering/checksum manifest.

## Overlap and leakage

The selected archive contains all 1,000 patch IDs in `experiments/pass3/dataset_manifest.json`. Relationship to the validated Pass 3 data: `PARTIAL OVERLAP`, with complete inclusion of the current 1,000-area baseline. The selected archive must not be used as an independent test set unless the existing 1,000 areas are excluded and split assignment is rebuilt at geographic-area level.

## Smoke-load result

A read-only smoke audit opened the outer archive, read parquet metadata, opened nested ZIP members, decoded representative S1 and reference-map TIFFs, verified dimensions/modes, and constructed the verified conceptual pair `X = (S1 VV, S1 VH)` and `Y = (multi-label metadata labels, optional reference-map target)`. A project artifact/provenance reload was not run because the archive was not extracted and the existing artifact system expects the repository's directory-based dataset layout. No model inference or training was run.