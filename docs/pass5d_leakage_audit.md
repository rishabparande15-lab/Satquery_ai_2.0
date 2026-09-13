# Pass 5D leakage audit

## Verdict: PASS for declared reproduction mode

- Dataset unit is the full geographic area. Split counts are 600/200/200 with empty train/validation/test intersections.
- Primary archive overlap with Pass 3 is 1,000/1,000 and is explicitly declared. No independent-generalization claim is made.
- The 4,000 non-Pass-3 primary areas were not used because matching local S2 was absent.
- S1 and references are identified in the selected 5,000-area primary archive; S2 is matched by the exact full `patch_id`, not coordinate suffix alone.
- Every selected S2 identity has exactly 12 canonical band files. The prepared/feature/target manifests retain identical batch identities.
- Version A and B use the same areas, target rows, seed, training budget, optimizer, validation rule and evaluation function.
- Pixel features use only S1/S2 inputs; no reference map or target enters feature computation.
- Feature selection was frozen from Pass 5B before this test. No test-driven additions or swaps were made.
- Per-image fixed preprocessing and feature formulas do not fit population statistics. Probe standardization uses training only.
- Validation selects epochs; test data are evaluated only after selection.
- Existing Pass 3 identity/hash audits found no repeated areas under alternate S1 or metadata identities. This run did not weaken those checks.

Mode B independent generalization is blocked, not failed: no local S2 exists for the 4,000 eligible non-Pass-3 areas. Downloading or substituting imagery was not authorized and was not attempted.
