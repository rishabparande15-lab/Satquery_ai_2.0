# SatQuery AI Final Pipeline Audit

## Pass 5E constrained interpretation update — 2026-09-13

Pass 5E adds `constrained_interpretation_v1` after the validated spatial-evidence layer. It uses exact allowlisted claims and deterministic templates; no LLM, VLM, VQA, prediction-head, or CROMA change was introduced. Simple and technical outputs preserve sensor, strength, epistemic status, feature, token/region, artifact, and evidence provenance. Unsupported objects, counts, locations, identities, confidence, altered claim text, and fake joint claims fail closed.

Real sample `61_39` generated its answer automatically from the canonical evidence artifact (SHA-256 `5906f4de2dbcb3d541590b57c802eba14e8a5d8585f591d88bc438eb3e6e3d8c`). Repeated structured outputs matched exactly. The 200-run in-process text-adapter benchmark measured a median of 0.390 ms and p95 of 0.926 ms on this machine. A final-code real loopback API run measured 0.1385 s for evidence construction, 0.00171 s for interpretation, and 0.1402 s total added latency, with projected region extents preserved. The API and website expose the additive interpretation result while preserving existing prediction behavior. Scope remains constrained evidence presentation, not arbitrary semantic answering.

## Pass 5 Update — 2026-09-12

Status: **VALIDATED (GREEN)** for the scoped SAR scene-level generalization experiment.

Pass 5 excluded all 1,000 Pass-3 areas by canonical, Sentinel-1, and metadata identity and built a new country-balanced 2,800/600/600 split from 4,000 eligible areas. Every one of the 4,000 SAR/reference triplets passed exhaustive geospatial and finite-value validation. Official SAR CROMA produced `[4000,225,768]` token representations and `[4000,768]` scene representations.

On the common 600-area held-out test set, constant/SAR physical/SAR CROMA MAE was 7.8588/7.2288/4.9280 pp; dominant accuracy was 21.33%/35.83%/54.00%. Independent metric recomputation differed by `0.0`; artifact hash/reload and saved-checkpoint reproducibility passed with maximum prediction difference `0.0`. The full report is `docs/pass5_sar_generalization.md` and the leakage evidence is `docs/pass5_leakage_audit.md`.

This additive result does not alter or downgrade Pass 3. Its claim is limited to independent-area SAR generalization for the existing scene-level 19-class labelled-pixel coverage target on the selected European population.

## Pass 3 Update — 2026-09-11

Pass 3 completed a fresh 1,000-area scene-level held-out comparison using the canonical 600/200/200 geographic split. The validated manifest is `experiments/pass3/dataset_manifest.json`; the comparison artifacts are under `experiments/pass3/comparison/`.

| Model | MAE pp | RMSE pp | Bias pp | Dominant accuracy |
|---|---:|---:|---:|---:|
| Constant | 7.6984 | 15.1128 | 0.0000 | 22.0% |
| Physical | 7.7384 | 14.7769 | 0.0000 | 33.0% |
| Optical CROMA | 5.5708 | 12.1651 | 0.0000 | 51.0% |
| SAR CROMA | 5.6489 | 12.2389 | 0.0000 | 50.5% |
| Joint CROMA | 4.9815 | 10.9457 | 0.0000 | 60.5% |
| Hybrid `[62 physical + 768 joint]` | 4.8846 | 11.0251 | 0.0000 | 59.5% |

The hybrid has the best MAE, while joint CROMA has the best RMSE and dominant accuracy. Physical-only is 0.52% worse than constant on MAE. Independent metric recomputation differed by `0.0`; a complete repeat produced bitwise-identical predictions and byte-identical checkpoints. The fresh leakage audit is in `docs/pass3_leakage_audit.md`, and the full scientific report is in `docs/pass3_scientific_validation.md`.

The result validates the current **scene-level coverage comparison** on this controlled subset. It does not validate token-level hybrid grounding, VQA, captioning, temporal prediction, or calibrated confidence.

## Overall Status

YELLOW

## Pass 5C structured spatial evidence update — 2026-09-13

Pass 5C adds a read-only `spatial_evidence_v1` representation over the existing validated artifacts. Real sample `61_39` now has 225 row-major token objects, exact pixel/projected extents, selected optical/SAR/local features, latent references to all three actual CROMA token arrays, deterministic evidence regions, traceable claims, scene summaries, sensor-specific views and complete provenance under `artifacts/spatial_evidence/61_39`.

No production architecture, Pass 3 result, CROMA representation, or task model was changed. Because no trained task-head prediction exists for `61_39`, prediction fields are explicitly unavailable; reference labels and untrained hybrid representations are not misreported as predictions. Exact two-run canonical hashes match. This validates a structured evidence foundation, not VQA, captioning, segmentation, confidence calibration or temporal prediction.

## Pass 5D controlled A/B update — 2026-09-13

Pass 5D ran an exact 600/200/200 Pass 3 reproduction comparison. Independent-generalization mode was impossible because the selected 5,000-area archive contains S1/reference/metadata but no S2, while the only matching local S2 contains exactly the 1,000 Pass 3 areas. This overlap is explicit and is not described as independent.

Version A (`62 physical + 768 joint CROMA GAP`) exactly reproduced Hybrid MAE/RMSE/accuracy of 4.8846/11.0251 pp/59.5%. Version B added 24 summaries from the frozen selected token features and obtained 4.8463/10.9362 pp/58.5%. Paired MAE change was −0.0383 pp with 95% interval `[−0.1211,+0.0435]`; accuracy fell one percentage point. Prediction superiority is inconclusive. The pixel layer is retained as the stronger spatial/explainability sidecar, not promoted as the default predictor. Metric recomputation and repeated inference differed by `0.0`.

## Scientific Components

| Component | Status | Evidence |
|-----------|--------|----------|
| BigEarthNet loader | VALIDATED | Strict full-identity discovery found 1,000 areas; fresh raw tensors `[12,120,120]`, `[2,120,120]`, `[120,120]`; malformed inputs rejected |
| S1 preprocessing | VALIDATED | VV/VH and deterministic profile executed fresh smoke |
| S2 preprocessing | VALIDATED | Canonical 12 bands, B02 grid, deterministic profile executed |
| CROMA | VALIDATED | Six fresh areas produced three `[1,225,768]` and three `[1,768]` outputs |
| Token alignment | VALIDATED FOUNDATION | Row-major mapping and spatial edge/fraction tests |
| 19-class targets | VALIDATED | Raw references generated token counts and scene coverage |
| Physical baseline | SMOKE VALIDATED | Fresh probe: 5.4574 pp MAE |
| CROMA baseline | VALIDATED WITH SCOPE | Fresh scene smoke plus historical 1,000-area token result |
| Hybrid baseline | SMOKE VALIDATED | Fresh aligned scene target: 10.5263 pp MAE |
| Train/validation/test | VALIDATED | Whole-area 2/2/2 smoke; historical 600/200/200 |
| Held-out evaluation | PARTIALLY VALIDATED | Fresh test has two areas; large result is historical |
| Leakage audit | VALIDATED | Disjoint identities, train-only scaling, validation selection |
| BigEarthNet.txt | VALIDATED LINKAGE | Fresh: 955 matched, 45 unmatched, 20,453 records |
| Region grounding | FOUNDATION ONLY | Deterministic overlaps; no learned model |
| VQA | NOT VALIDATED | No evaluated model |
| Captioning | NOT VALIDATED | Linkage only |
| Temporal/change | VALIDATION ONLY | Pair rejection works; no prediction model |
| Confidence | NOT VALIDATED | No calibration experiment |
| Controller | TRUTHFUL / PARTIAL | Tests prevent unavailable specialist selection |
| Website | VALIDATED EXISTING RUNTIME | Optical/SAR/joint paths retained |

## Actual Metrics

Fresh six-area scene smoke: constant 7.5965 MAE / 18.2982 RMSE / 50% accuracy; physical 5.4574 / 12.3221 / 0%; optical 10.5263 / 26.2817 / 0%; SAR 9.0199 / 20.6863 / 0%; joint 10.5263 / 26.0735 / 0%; hybrid 10.5263 / 26.7673 / 0%. Biases are recorded in `docs/fresh_scientific_evaluation.md`.

Historical 1,000-area token joint CROMA: 6.1553 pp MAE and 53.8743% accuracy versus 9.2137 pp constant MAE.

## Reproducibility

The smoke run stores identities, config, seed, logs, checkpoints, truth, predictions, metrics, and provenance under `experiments/smoke`. Independent metric recomputation differed by `0.0`. Clean-machine reproduction remains partial because data/checkpoints are external and historical prepared directories were inaccessible.

## Blockers

- No validated VQA, captioning, learned grounding, temporal/change, or calibration resources.
- Historical prepared 1,000-area features/targets could not be freshly read.
- A meaningful fresh held-out evaluation requires more accessible areas per split and run budget.

## Remaining Gaps

- Fresh large same-split physical/CROMA/hybrid evaluation.
- Learned grounding, VQA, captioning, change prediction, and calibration.
- Browser visual verification and clean-environment asset provisioning.

## Final Verdict

YELLOW — PARTIALLY VALIDATED

The raw-to-prediction pipeline completed on six real areas with optimization, validation selection, held-out inference, checkpointing, failure tests, and exact metric recomputation. It is not GREEN because the fresh held-out set is too small, the large run was inaccessible, and several product capabilities have no evaluated models.
