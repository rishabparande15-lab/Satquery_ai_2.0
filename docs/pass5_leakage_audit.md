# Pass 5 Leakage Audit

Status: **PASS**

The exact 1,000-area Pass-3 population was loaded from `experiments/pass3/dataset_manifest.json`. The exclusion artifact records 1,000 unique canonical area IDs, 1,000 unique Sentinel-1 identities, and 1,000 unique metadata/Sentinel-2 identities. All three identity namespaces were joined against the 5,000-row embedded metadata table; filename-only matching was not used.

| Check | Overlap |
|---|---:|
| Pass-3 train vs Pass-5 test | 0 |
| Pass-3 validation vs Pass-5 test | 0 |
| Pass-3 test vs Pass-5 test | 0 |
| Any Pass-3 area vs Pass-5 test | 0 |
| Pass-5 train vs validation | 0 |
| Pass-5 train vs test | 0 |
| Pass-5 validation vs test | 0 |

The source archive has 5,000 areas. Exactly 1,000 were excluded, leaving 4,000 eligible areas. The deterministic, country-balanced seed-53 split is 2,800/600/600. Machine-readable evidence is in `experiments/pass5/pass3_exclusion_set.json`, `dataset_manifest.json`, `split.json`, and `leakage_audit.json`.
