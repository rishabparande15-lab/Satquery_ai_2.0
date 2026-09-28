# Phase 3A - Benchmark Re-run and Expected-Output Audit

## Scope and safety

This phase audited the current image-language and benchmark foundation without
selecting or training a VLM. No scientific predictor, preprocessing, dataset
split, CROMA checkpoint, representation shard, annotation, spatial mapping
policy, or existing staged work was modified. No commit or push was performed.

The locked scientific reference remains:

- 5,000 areas; 4,600 train / 200 validation / 200 test.
- Test accuracy: `65.0%`.
- Test MAE: `4.1034286734 pp`.
- Test RMSE: `9.5873312123 pp`.
- Bootstrap 95% CI: `[3.8375601352, 4.3741238082] pp`.

## 1. Repository inventory

The current executable image-language foundation is in
`src/image_language_foundation.py` and `src/image_language_dataset.py`. It
provides canonical sample assembly, typed representation inputs, prompt/output
contracts, reserved benchmark adapters, and no learned inference.

Relevant current paths inspected:

- `src/image_language_foundation.py`: `ImageLanguageSample`,
  `BigEarthNetTextAdapter`, reserved RSVQA/VRSBench/CDVQA adapters,
  `VLMAdapter`, and future evaluator boundaries.
- `src/image_language_dataset.py`: deterministic source-preserving pools and
  eligibility/leakage audits.
- `src/representation_catalog.py`, `src/representation_linking.py`, and
  `src/representation_artifacts.py`: verified representation identity and
  provenance.
- `src/region_grounding.py`: deterministic box-to-token geometry helper, not a
  learned grounding model.
- `src/interpretation_adapter.py`: allowlisted deterministic
  evidence-to-text adapter, explicitly no LLM/VLM generation.
- `src/temporal.py`: validation of ordered, spatially corresponding pairs and
  numerical representation differences; no learned temporal model/evaluator.
- `tests/test_image_language_foundation.py`, `tests/test_image_language_dataset.py`,
  `tests/test_spatial_contract.py`, `tests/test_representation_linking.py`,
  `tests/test_region_grounding.py`, and `tests/test_phase2_integration.py`:
  current focused coverage.

No local RSVQA, VRSBench, or CDVQA dataset, benchmark evaluator, model adapter,
or benchmark notebook was found. No concrete EO-VLM is installed or selected.

## 2. Capability matrix

| Task | Dataset | Current code path | Required input / representation | Model required | Evaluation metric | Can run now? | Classification and reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Binary QA | BigEarthNet.txt | `BigEarthNetTextAdapter`, canonical samples | Verified optical+SAR scene references: physical, joint GAP, hybrid | Learned VLM or task model | Exact answer accuracy, if a benchmark evaluator/model exists | Foundation assembly only | CONTRACT/FOUNDATION ONLY; no learned inference or score |
| Multiple-choice QA | BigEarthNet.txt | `BigEarthNetTextAdapter`, deterministic MCQ parsing | Same verified core references plus source choices | Learned VLM or task model | Exact option accuracy, if protocol exists | Foundation assembly only | CONTRACT/FOUNDATION ONLY; no learned inference or score |
| Captioning | BigEarthNet.txt | `BigEarthNetTextAdapter`, canonical caption samples | Verified scene representations | Learned captioning VLM | Caption metric not implemented in current repository | No | MODEL DEPENDENCY and EVALUATOR MISSING |
| Text-box grounding | BigEarthNet.txt | `BigEarthNetTextAdapter`, source-only spatial reference | Source geometry plus a validated image/token frame | Learned grounding model and verified coordinates | IoU/mIoU or benchmark-specific detection metric | No | BLOCKED BY SPATIAL SEMANTICS and MODEL DEPENDENCY |
| Point-guided grounding | BigEarthNet.txt | Same foundation path | Point and enclosing box with authoritative coordinate semantics | Learned grounding model | IoU/mIoU or benchmark-specific metric | No | BLOCKED BY SPATIAL SEMANTICS and MODEL DEPENDENCY |
| Deterministic evidence explanation | Local scientific evidence | `interpretation_adapter.py` | Validated `spatial_evidence_v1` | No learned model | No VLM benchmark score | Yes, for evidence questions | ACTUALLY RUNNABLE AND MEASURED as controlled deterministic output only |
| RSVQA | No local dataset found | Reserved adapter only | Benchmark images/annotations and a learned VLM | Required | Repository evaluator absent | No | BLOCKED BY DATASET and MODEL/EVALUATOR DEPENDENCY |
| VRSBench captioning | No local dataset found | No executable adapter/evaluator | Benchmark images/captions and learned VLM | Required | Benchmark caption metrics unavailable | No | BLOCKED BY DATASET and MODEL/EVALUATOR DEPENDENCY |
| VRSBench grounding | No local dataset found | No executable adapter/evaluator | Benchmark grounding annotations and learned VLM | Required | Benchmark grounding protocol unavailable | No | BLOCKED BY DATASET and MODEL/EVALUATOR DEPENDENCY |
| CDVQA | No local dataset found | `CDVQAAdapter` reserved only | Authoritative T1/T2 data, temporal identities, learned model | Required | Benchmark change-QA evaluator unavailable | No | BLOCKED BY DATASET, TEMPORAL SUPPORT, and MODEL DEPENDENCY |
| Scientific optical+SAR coverage | Frozen BigEarthNet v2 5,000-area set | Existing Pipeline 3 hybrid path | `physical_62d` + `joint_croma_gap_768d` -> `hybrid_830d` | Existing frozen probe | Accuracy, MAE, RMSE, bootstrap CI | Yes | ACTUALLY RUNNABLE AND MEASURED; this is not language reasoning |
| Learned optical+SAR semantic reasoning | No VLM dataset/evaluator | No concrete VLM adapter | Model-specific optical+SAR input contract | Required | No evaluator | No | MODEL DEPENDENCY; scientific joint representation does not imply VLM reasoning |
| Temporal representation difference | Synthetic/unit contract only | `validate_temporal_pair`, `compare_representations` | Explicit dates, geospatial metadata, corresponding arrays | No learned model | Numerical L2/mean absolute difference only | Contract/unit behavior only | RUNNABLE BUT NO VALID BENCHMARK SCORE; not CDVQA |

## 3. BigEarthNet.txt foundation re-run

The current canonical artifact was read without rebuilding or changing it.

| Identity or count | Observed |
| --- | ---: |
| Validated samples | 101,542 |
| Quarantined annotations | 29 |
| Included samples | 101,542 |
| Unique represented images | 4,770 |
| Train / validation / test | 93,208 / 4,175 / 4,159 |
| Binary QA | 37,503 |
| Multiple-choice QA | 33,969 |
| Captions | 4,770 |
| Text boxes | 12,178 |
| Points | 13,122 |
| Missing required representations | 0 |
| Spatial samples deliberately unmapped | 25,300 |

The canonical identity matched:

- Image-language fingerprint:
  `2e2ccb5c85147519ee4b19d639a689408ba6ea01dd2f2ed98a70b847b3eb6ae0`.
- Samples SHA-256:
  `7db25d8b6f40034fd15795edb500ee3265ce11551ad1af540622782b99a39e31`.
- Dataset fingerprint:
  `7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625`.
- Split fingerprint:
  `2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c`.
- Representation catalog fingerprint:
  `0f5ce8feceff2fe8f2a75e1570b36edabf8a4716c57159243ffc6063eafcfd30`.

Representative canonical samples were audited:

| Task | Sample ID | Split | Actual adapter/model action | Preserved output | Scientific result |
| --- | --- | --- | --- | --- | --- |
| Binary QA | `ils:05e6e05dae4c440196cbee9428dd6790d54e7b961db0c3776fbc869de2568287` | train | Foundation record only; no learned model | Source answer `no` | No benchmark score |
| Multiple choice | `ils:f20ecb8ae58b6312dc4eab220a2a8c97398413d3e4f2467a5ac009cb98c6c8a7` | train | Foundation record and deterministic choice structure; no learned model | Source answer `b` | No benchmark score |
| Captioning | `ils:6a5fbbbbbd44a5beb0d146f1391d0f6e02177a759e931f03c5465145ea4a7c78` | test | Foundation record only; no caption model | Source caption preserved | No benchmark score |
| Text-box grounding | `ils:3bfa21801cda649196fcc5daa1870539ae7691a3d8463ac6c988f2116749d869` | train | Source-only spatial record; no coordinate conversion | `UNMAPPED` geometry preserved | No grounding score |
| Point grounding | `ils:43513ac84791d2d0968013221fbc65247f0de487cdb595815a9ec99542301aa1` | train | Source-only spatial record; no coordinate conversion | `UNMAPPED` geometry preserved | No grounding score |

The constrained interpretation adapter was invoked with no scientific evidence
for these representative language samples. It returned its explicit
unavailable response and was validated as deterministic. It is a controlled
allowlisted evidence-to-text adapter, not a learned VLM and not a benchmark
baseline. No learned model was invoked and no language metric was computed.

## 4. Spatial and grounding audit

The Phase 2F decision remains **NO / NOT YET**. BigEarthNet.txt source boxes
and points remain preserved with provenance, but authoritative axis order,
origin, pixel semantics, raster relationship, and normalization semantics are
not established.

Current audit state:

- Spatial annotations preserved: yes.
- Geometry provenance preserved: yes.
- Verified source-to-analysis mapping: `0`.
- Source-to-CROMA-token mapping: `0`.
- Learned grounding predictions: `0`.
- Foundation spatial samples deliberately unmapped: `25,300`.
- Fail-closed status: active.

No coordinate transform was applied. The deterministic
`region_grounding.py` helper remains a generic geometry utility for already
canonical coordinates; it is not evidence that BigEarthNet.txt coordinates can
be mapped and is not a learned grounding system.

## 5. RSVQA audit

No RSVQA dataset, local split, loader, preprocessing path, inference adapter,
or evaluator is available in the repository. The reserved `RSVQAAdapter`
raises `NotImplementedError` by design.

- Samples attempted: `0`.
- Samples evaluated: `0`.
- Metrics: none.
- Exact-answer accuracy: not computed.
- Status: **BLOCKED / MODEL DEPENDENCY**, with the local dataset and evaluator
  also unavailable.

No full or partial RSVQA dataset was downloaded.

## 6. VRSBench audit

No VRSBench images, annotations, caption evaluator, grounding evaluator, or
executable adapter is available locally.

- Caption samples attempted/evaluated: `0 / 0`.
- Grounding samples attempted/evaluated: `0 / 0`.
- Metrics: none.
- Status: **FOUNDATION READY, MODEL DEPENDENCY BLOCKED** and dataset/evaluator
  unavailable.

No generic LLM or non-EO model was substituted.

## 7. CDVQA and temporal audit

No CDVQA dataset, temporal annotation set, benchmark loader, or evaluator is
available. BigEarthNet v2's paired Sentinel-1/Sentinel-2 modalities are not a
T1/T2 temporal pair. BigEarthNet.txt temporal fields are reserved and empty.

The repository does provide `validate_temporal_pair` and
`compare_representations` for explicit dates, geospatial metadata, spatial
correspondence, and numerical array differences. These are contract/unit
utilities only.

**TEMPORAL REPRESENTATION CONTRACT EXISTS BUT LEARNED TEMPORAL
MODEL/EVALUATION IS NOT IMPLEMENTED.**

No temporal score was produced.

## 8. Optical-SAR audit

The scientific Pipeline 3 joint representation is executable and locked:

```text
physical_62d + joint_croma_gap_768d -> hybrid_830d -> scientific coverage probe
```

The fresh Phase 0 rerun verified its 65.0% accuracy, 4.1034286734 pp MAE,
9.5873312123 pp RMSE, and bootstrap CI. Focused Phase 3A tests also passed
joint integration paths.

This is scientific joint representation/prediction, not learned image-language
reasoning. There is no current VLM that consumes optical+SAR and emits semantic
answers, captions, or grounding predictions. No language benchmark score is
claimed.

## 9. Expected-output scorecard

| Benchmark | Task | Dataset available | Current pipeline | Learned VLM | Can score | Fresh result | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BigEarthNet.txt | Binary QA | Yes, 37,503 samples | Canonical foundation assembly | No | No | Source answer preservation only | CONTRACT/FOUNDATION ONLY |
| BigEarthNet.txt | Multiple choice QA | Yes, 33,969 samples | Canonical foundation assembly and deterministic options | No | No | Source option/answer preservation only | CONTRACT/FOUNDATION ONLY |
| BigEarthNet.txt | Captioning | Yes, 4,770 samples | Canonical foundation assembly | No | No | Source captions preserved | CONTRACT/FOUNDATION ONLY |
| BigEarthNet.txt | Text/point grounding | Yes, 25,300 validated samples | Source-only geometry and fail-closed provenance | No | No | 0 mappings; 0 token links | BLOCKED BY DATA / SEMANTICS / MODEL DEPENDENCY |
| RSVQA | VQA | No | Reserved adapter only | No | No | None | BLOCKED BY DATA / MODEL DEPENDENCY |
| VRSBench | Captioning | No | No executable adapter/evaluator | No | No | None | BLOCKED BY DATA / MODEL DEPENDENCY |
| VRSBench | Grounding | No | No executable adapter/evaluator | No | No | None | BLOCKED BY DATA / MODEL DEPENDENCY |
| CDVQA | Change QA | No | Reserved adapter; temporal contracts only | No | No | None | BLOCKED BY DATA / TEMPORAL / MODEL DEPENDENCY |
| Pipeline 3 | Optical-SAR coverage prediction | Yes, exact 5,000 areas | Frozen scientific predictor | Existing frozen probe, not a VLM | Yes | 65.0% / 4.1034286734 pp / 9.5873312123 pp | ACTUALLY RUNNABLE AND MEASURED |
| Pipeline 3 | Evidence-to-text interpretation | Local evidence artifacts | `interpretation_adapter.py` | No; deterministic adapter | Not a benchmark score | Deterministic controlled output or explicit unavailable response | RUNNABLE BUT NO VALID SCORE |

## 10. Current foundation versus actual capability

### Actually runnable and measured

- Deterministic Pipeline 3 optical/SAR scientific analysis.
- Frozen `hybrid_830d` scene-level coverage prediction.
- Deterministic evidence generation and constrained evidence-to-text output.
- Verified representation, provenance, split, and receipt handling.
- Canonical BigEarthNet.txt foundation assembly and source-preserving task
  records.

### Runnable foundation with no valid benchmark score

- BigEarthNet.txt binary QA, MCQ, caption, and grounding sample assembly.
- Deterministic prompt/template and adapter contract behavior.
- Temporal pair validation and numerical representation comparison for explicitly
  provided, spatially corresponding inputs.

### Contract/foundation only

- RSVQA, VRSBench, and CDVQA benchmark adapters.
- Learned VQA, captioning, grounding, temporal QA, and change description.
- Model-specific optical+SAR language reasoning.

### Blocked

- All learned EO-VLM tasks: no concrete model or model interface exists.
- BigEarthNet.txt grounding: source coordinate semantics are unresolved.
- RSVQA/VRSBench/CDVQA: datasets and benchmark evaluators are unavailable.
- Temporal/change benchmark scoring: no authoritative temporal dataset or model.

## 11. Validation

Focused Phase 3A-relevant tests:

- `59 passed, 1 skipped` across annotation, image-language, spatial, linking,
  region, and joint-representation tests.

Full regression:

- `359 passed, 5 skipped`.
- Existing warnings only: rasterio deprecation warnings and the known Windows
  pytest cache permission warning.
- `git diff --check`: passed.

No new tests, pipeline code, model files, datasets, or generated scientific
artifacts were added. The representative audit outputs remain isolated under
`.tmp/`.

## 12. Recommended next research gate

Do not select a model based on this audit alone. The next gate should define a
benchmark-compliant EO-VLM interface that can consume the required optical,
SAR, joint, and possibly temporal inputs; preserve dataset/split provenance;
produce task-appropriate structured outputs; support exact-answer, caption, and
grounding evaluators; and expose resource behavior on the target machine.

Model selection and training belong after that contract and benchmark protocol
are approved.

## FINAL GATE

### BENCHMARK AUDIT INCOMPLETE — BLOCKERS REMAIN
