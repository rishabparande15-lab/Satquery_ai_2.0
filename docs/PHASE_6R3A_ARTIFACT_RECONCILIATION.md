# Phase 6R.3A Artifact Reconciliation

## 1. Objective

Reconcile and protect the validated Pipeline 3 exact 5,000-area scientific records without changing the scientific result, dataset, split, checkpoint, or historical evidence.

## 2. Starting state

The audit started at commit `e5c29bab` on `main`, with `main` synchronized to `origin/main` and a clean working tree. The Phase 6R.3 production architecture repair was already present. The focused Pipeline 3 artifact audit passed before edits. The historical note reported `311 passed, 5 skipped, 1 failed`, but the live checkout did not reproduce that baseline mismatch: `baseline_metrics.json` already contained `complete_validation_and_final_test`.

## 3. Protected artifacts inspected

The following canonical records were inspected:

- `experiments/pipeline3_5000/baseline_metrics.json`: historical baseline plus the recorded final test event; canonical test metrics.
- `experiments/pipeline3_5000/final_metrics.json`: canonical completed experiment summary and selected-model metrics.
- `experiments/pipeline3_5000/experiments.json`: experiment ledger and selection/receipt decisions.
- `experiments/pipeline3_5000/audit_summary.json`: dataset completeness, availability, split, and leakage audit.
- `experiments/pipeline3_5000/fingerprint.json`: dataset-gate fingerprint and canonical dataset fingerprint basis.
- `experiments/pipeline3_5000/reproducibility.json`: dataset audit hashes and environment/receipt metadata.
- `experiments/pipeline3_5000/dataset_manifest.json` and `split_manifest.json`: selected areas and frozen split.
- `experiments/pipeline3_5000/final/config.json`, `metrics.json`, `fingerprint.json`, and `test_receipt.json`: final model configuration, scientific metrics, scientific fingerprint, and single-test receipt.
- `experiments/pipeline3_5000/baseline/config.json`, `fingerprint.json`, and `validation_results.json`: preselection baseline configuration and model-state evidence.
- `docs/PIPELINE3_5000_6R2_REVALIDATION.md`, `docs/PIPELINE3_5000_6R2A_DATA_RESTORATION.md`, `docs/PIPELINE3_5000_6R2B_REVALIDATION.md`, and `docs/PHASE_6R3_ARCHITECTURE_REPAIR.md`: historical validation and production integration reports.

## 4. Authoritative source-of-truth mapping

| Record | Meaning and authority | Required identity |
| --- | --- | --- |
| `dataset_manifest.json` | Canonical selected-area rows and availability | Dataset fingerprint and manifest hash |
| `split_manifest.json` | Canonical train/validation/test membership | Split fingerprint |
| `audit_summary.json` | Dataset-gate and leakage findings | Dataset manifest and split records |
| Root `fingerprint.json` | Dataset-gate fingerprint only; not a model result | Dataset fingerprint `7dfd...7625` |
| Root `reproducibility.json` | Hash receipt for the dataset audit | Manifest, split, and audit hashes |
| `baseline_metrics.json` | Protected baseline/final test metric receipt | 5,000 areas; selected frozen experiment |
| `final_metrics.json` | Canonical completed experiment summary | Same test receipt as baseline metrics |
| `final/config.json` | Final model/config contract | Dataset, split, config, and state hashes |
| `final/fingerprint.json` | Canonical scientific result fingerprint | Dataset, split, config, model state, metrics |
| `final/test_receipt.json` | One consumed held-out test event | Scientific fingerprint and 200 test areas |
| 6R.2B report | Historical fresh revalidation receipt | Reported 6R.2B fingerprint `37a54e...ec01b` |

Canonical JSON artifacts are authoritative for the records they represent. Reports explain provenance and historical comparisons; they do not replace machine-readable receipts.

## 5. Artifact consistency findings

The canonical records are internally consistent:

- Root metrics are complete, use 5,000 areas, select `hybrid`, and agree on the test object.
- The final receipt uses 200 held-out test areas and points to the final scientific fingerprint.
- Dataset and split counts are 4,600 train, 200 validation, and 200 test.
- The root dataset fingerprint status correctly says that it is a dataset-gate fingerprint only; the final model fingerprint is in `final/fingerprint.json`.
- The production architecture report correctly describes the model as `62 physical + 768 joint CROMA GAP = 830D -> 19` scene-level output. No `225 x 19` spatial prediction is claimed.
- Confidence remains unavailable where no calibrated per-scene confidence exists.

## 6. The `baseline_metrics.json` / test discrepancy

The reported historical failure expected `baseline_metrics.json.status == "complete_validation_and_final_test"` while a prior working-tree version reportedly contained `"not_run"`. In the audited checkout, the artifact restored from `HEAD` contains the expected complete status and the unchanged audit test passed. Therefore no scientific artifact was changed and no stale test repair was required for that discrepancy.

The precise canonical final-test values are MAE `4.103428673440559` pp, RMSE `9.587331212294444` pp, and dominant-class accuracy `0.65`. The 6R.2B fresh revalidation report records `4.10342865` pp, `9.58733110` pp, and `130/200 = 65.0%`. The small numerical differences are explicitly retained as separate receipts; they do not justify rewriting either record.

## 7. Decision and justification

The protected canonical artifacts were retained unchanged. The authoritative result is the canonical final-test receipt represented by root `baseline_metrics.json`, root `final_metrics.json`, and the linked final fingerprint. The 6R.2B report remains historical revalidation evidence. The discrepancy is numerical receipt drift at sub-micro precision, not a changed dataset, split, model identity, or metric definition.

The dataset audit script was repaired so its default output is `experiments/pipeline3_5000/audit_runs/current/`, which is ignored, and it raises an error if configured to write directly to the canonical experiment directory. This prevents a dataset audit or runtime revalidation from replacing protected scientific records.

## 8. Hash/fingerprint verification

The repository checks and artifact inspection verified these values and relationships:

- Dataset fingerprint: `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`.
- Split fingerprint: `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`.
- Model-state SHA-256: `85d9390c66a887276db24a7cadb58c398770cde2e17488a1a9c42e16819da634`.
- Final configuration SHA-256: `32c8776f44a070fca265a8ae81d92fe4f22649b190af08f10fdff9eea2c9e80e`.
- Scientific fingerprint: `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`.
- 6R.2B report fingerprint: `37a54e07bf238c7987e223112953c4870610a1e9746166a3cd80419b435ec01b`.

The focused audit recomputes the dataset manifest/split/audit hashes and the root dataset fingerprint from canonical JSON. Final configuration and scientific fingerprint references agree with the final model records.

## 9. Regression protections

`tests/test_pipeline3_5000_audit.py` now protects the exact canonical metrics, dataset/split/model/config/scientific fingerprints, receipt status, and the distinction between dataset-gate and final scientific fingerprints. It also verifies that the dataset audit output is non-canonical and that canonical output is rejected. Existing architecture/API tests continue to protect the production capability route and scene-level output contract.

## 10. Test results

- Focused artifact tests after the repair: `7 passed, 1 warning`.
- The earlier global-Python full-suite attempt could not collect NumPy-dependent tests because the global Python 3.14 NumPy installation has an incompatible binary extension.
- The first supported-environment full-suite attempt was blocked by a Windows permission error scanning the user pytest temp directory; it produced setup errors, not test failures.
- The supported-environment full suite rerun with isolated repository-local `--basetemp .pytest-temp/full` passed: `314 passed, 5 skipped, 0 failed`.
- No frontend package or frontend test project exists in this workspace, so no frontend command was applicable.

## 11. Remaining limitations

The canonical model is a scene-level classifier/probe, not a spatial head. No calibrated per-scene confidence, segmentation, grounding, VQA, captioning, or temporal-change model is introduced by this phase. The machine-local dataset and checkpoint remain ignored and are not added to Git.

## 12. Exact completion criteria

Phase 6R.3A is complete only when protected artifacts and their source mapping are documented; the baseline discrepancy is resolved without corrupting records; hashes and fingerprints reconcile; runtime outputs cannot overwrite canonical records; focused and full applicable tests are evaluated; documentation and diff are reviewed; and a clean checkpoint is committed and pushed without history rewriting or force push.

## 13. Final 6R.3A status

**Ready for final diff review and checkpoint.** All applicable Phase 6R.3A tests pass; frontend validation is not applicable because no frontend project exists in the workspace.
