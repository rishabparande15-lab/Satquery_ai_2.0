# Phase 3AC.4 — Sanitization and Safe Staging Recovery

## Final Report

1. **Phase status:** `PHASE3AC_COMPLETE`.
2. **Leak root cause:** the paired controller initialized `qwen_revision` from `scripts.phase3o15_contrastive_objective_ablation.SNAPSHOT`, an absolute Hugging Face cache path. `Handler._json` serialized it unchanged, and the browser's JSON export copied that response.
3. **Provenance-fix files:** `src/satquery_v1.py`, `src/api.py`, and new `src/public_serialization.py`; paired provenance is also rendered in `src/static/app.js` and `src/static/index.html`.
4. **Internal Qwen location changed:** no. The controller still passes the existing local `SNAPSHOT` to `Qwen25VLRGBAdapter.load_model`; the mocked controller test confirms it.
5. **Public Qwen model ID:** `Qwen/Qwen2.5-VL-3B-Instruct`.
6. **Public approved Qwen revision:** `66285546d2b821cf421d4f5eb2576359d3770cd3`.
7. **Local cache path exposed after fix:** no. Route/API serialization, saved-report download, and the successful browser export contain no local path indicators.
8. **VQA serialization audit:** pass through the centralized public JSON sanitizer.
9. **Scene-description serialization audit:** pass through the centralized public JSON sanitizer.
10. **Optical-SAR serialization audit:** pass; pinned identity retained and local cache path absent.
11. **Temporal serialization audit:** pass through the centralized public JSON sanitizer.
12. **Pre-existing staging directories:** 3.
13. **Directory 1:** `request-9mtkqrsy` — `UNKNOWN_DO_NOT_DELETE`; age 81,944 seconds at inspection, in root, not a reparse point; content traversal and ACL inspection denied.
14. **Directory 2:** `request-mo7g2r32` — `UNKNOWN_DO_NOT_DELETE`; age 94,621 seconds at inspection, in root, not a reparse point; content traversal and ACL inspection denied.
15. **Directory 3:** `request-_ula3arn` — `UNKNOWN_DO_NOT_DELETE`; age 94,594 seconds at inspection, in root, not a reparse point; content traversal and ACL inspection denied.
16. **Real directories safely deleted:** 0.
17. **Real directories preserved:** 3.
18. **Startup recovery verified:** yes. The final startup receipt logged `deleted=0 preserved=3` and did not require an empty staging root.
19. **Traversal protection:** pass; outside-root and traversal candidates classify unknown and remain untouched.
20. **Symlink/reparse protection:** pass through deterministic reparse-point simulation. Native directory-symlink creation was unavailable to this Windows test account, so that one test was skipped.
21. **Active/recent protection:** pass; active path generators protect every listed owner, and recent recognized folders are preserved.
22. **Optical-SAR browser HTTP:** 200.
23. **Optical-SAR route:** `OPTICAL_SAR_ANALYSIS`.
24. **Optical-SAR output:** `Yes`.
25. **`COREGISTRATION_NOT_VERIFIED` preserved:** yes, in pair evidence and warnings.
26. **Evidence/provenance rendered:** yes; evidence footprint/statistics and the new model/provenance disclosure rendered.
27. **JSON export:** downloaded and parsed successfully at `artifacts/final/runtime/phase3ac4/optical_sar_export.json`; no local path leaked.
28. **Console errors:** 0.
29. **Page errors:** 0.
30. **Failed requests:** 0.
31. **Health:** `/api/v1/health` returned `ok`.
32. **Readiness:** `READY_WITH_LIMITATIONS` (`runtime=READY`).
33. **Grounding:** `BLOCKED`, as expected.
34. **Maximum GPU allocated:** 8,368,817,152 bytes.
35. **Maximum GPU reserved:** 8,531,214,336 bytes.
36. **S2 projector digest unchanged:** yes, `e7f22d74e048cde31a746186b52b9ca5eb18b71532fc248d9a1c15fbd90819db`.
37. **Joint projector digest unchanged:** yes, `bedcb8dc60d3375878b6fe732cb90b568283c7d5565148fb1a52eae0d1b12eaa`.
38. **Chg2Cap digest unchanged:** yes, `d737a92afa3cb76a07f672ee905afb59278211c77ccce517a37ae4637110194c`.
39. **Python tests:** 122 passed, 1 skipped. Command covered provenance serialization, staging recovery, specialist lifecycle, health/API/export, geospatial evidence, single-image, temporal, unified API, agent routing, and scene resolver tests. The skip is native symlink creation unavailable to this Windows account.
40. **Frontend tests:** 8 passed, 0 failed (`node --test tests/frontend_state.test.cjs`).
41. **Held-out TEST images accessed:** 0. The requested browser patch is metadata-classified `validation`.
42. **Held-out TEST labels accessed:** 0.
43. **Held-out TEST inference:** 0. Three diagnostic/verification requests used the specified validation patch while resolving evidence integration; the final passing Chrome run is recorded in the browser receipt.
44. **Held-out TEST metrics:** 0.
45. **Files intentionally modified:** `src/api.py`, `src/satquery_v1.py`, `src/geospatial_evidence.py`, `src/static/app.js`, `src/static/index.html`, `tests/frontend_state.test.cjs`, `tests/test_geospatial_evidence.py`.
46. **Files intentionally created:** `src/public_serialization.py`, `src/upload_staging.py`, `tests/test_public_serialization.py`, `tests/test_upload_staging.py`, `scripts/phase3ac4_optical_sar_browser.py`, `docs/PHASE3AC4_SANITIZATION_AND_STAGING_RECOVERY.md`, plus the receipts in the artifact directory.
47. **Artifact directory:** `artifacts/final/runtime/phase3ac4/`.
48. **Documentation:** `docs/PHASE3AC4_SANITIZATION_AND_STAGING_RECOVERY.md`.
49. **Final Phase 3AC classification:** `PHASE3AC_COMPLETE`.
50. **Remaining runtime gaps:** no Phase 3AC.4 release blocker. Grounding remains intentionally blocked. The three inaccessible old staging folders remain unknown and protected.
51. **Remaining externally blocked items:** the current account cannot enumerate or inspect ACLs for those three directories. Recovery preserves them; no elevated or destructive operation was attempted.
52. **Exact next action:** stop runtime engineering at Phase 3AC; proceed with the project's next planned non-runtime handoff. Do not create Phase 3AC.5.

## Staging Recovery Contract

Startup scans only immediate children of the configured SatQuery upload root. It recognizes only the three SatQuery temporary-directory prefixes and expected image filenames, verifies containment and non-reparse status, honors the existing 900-second TTL, and protects active paths. Only directories classified `STALE_SAFE_TO_DELETE` are removed. Unknown entries, inspection failures, and cleanup permission errors are preserved.

## Verification Artifacts

- `artifacts/final/runtime/phase3ac4/optical_sar_browser.json`
- `artifacts/final/runtime/phase3ac4/optical_sar_browser.png`
- `artifacts/final/runtime/phase3ac4/optical_sar_export.json`
- `artifacts/final/runtime/phase3ac4/staging_recovery.json`
- `artifacts/final/runtime/phase3ac4/phase3ac4_final_receipt.json`
