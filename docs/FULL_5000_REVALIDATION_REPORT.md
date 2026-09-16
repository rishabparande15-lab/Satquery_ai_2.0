# Full 5,000-Area Forensic Revalidation Report

## A. Executive summary

This audit did not reach a fresh scientific re-run because the local Python runtime is currently broken before the repository can import its scientific stack. The environment is locked to Python 3.14.4, and both NumPy and PyTorch fail at import time. Because of that, no fresh 5,000-area processing, feature extraction, model evaluation, or architecture/API execution could be produced from local evidence in this session.

The repository still contains extensive documentation stating that the exact 5,000-area Pipeline 3 experiment is complete and validated, with a documented final result of 65.0% dominant-class accuracy, 4.1034 pp MAE, and 9.5873 pp RMSE. However, those values remain documented claims, not fresh runtime measurements from the current environment.

The forensic verdict is therefore:

- Fresh runtime evidence: BLOCKED by environment failure.
- Documented scientific result: retained as a previously reported claim, not freshly reproduced.
- Current status: NOT VERIFIED AS A FRESH 5,000-AREA END-TO-END RUN.

This is the correct scientific stop condition. The pipeline must not be treated as freshly validated while the interpreter and dependency stack are broken.

## B. Repository identity

- Repository: E:\SatQuery_ai_2.0\Satquery_ai_2.0
- Git branch: main
- Git status: working tree contains uncommitted files, including annotation-related additions and generated artifacts
- HEAD: 2f7b241e4f4687d2aae263faf4b1d5c08eae9e5e

Relevant documentation reviewed:

- [README.md](README.md)
- [docs/CURRENT_PROJECT_STATUS.md](CURRENT_PROJECT_STATUS.md)
- [docs/PIPELINE3_5000_TRAINING_REPORT.md](PIPELINE3_5000_TRAINING_REPORT.md)
- [docs/ANNOTATION_FOUNDATION.md](ANNOTATION_FOUNDATION.md)
- [docs/ANNOTATION_SOURCE_AUDIT.md](ANNOTATION_SOURCE_AUDIT.md)
- [docs/SPATIAL_ANNOTATION_PROTOCOL.md](SPATIAL_ANNOTATION_PROTOCOL.md)
- [docs/IMAGE_LANGUAGE_DATASET.md](IMAGE_LANGUAGE_DATASET.md)

## C. Environment

Freshly recorded environment state:

- Python: 3.14.4, free-threading build
- NumPy: 2.5.1
- PyTorch: import fails in this environment
- CUDA: not measurable because torch import fails before any CUDA detection is possible
- GPU: not measurable because torch import fails before any GPU enumeration is possible

Fresh verification command:

```powershell
git status --short --branch; echo "HEAD=$(git rev-parse HEAD)"; py -V; py -c "import sys; print(sys.executable); print(sys.version)"; py -c "import torch; print('TORCH=' + torch.__version__); print('CUDA=' + str(torch.version.cuda)); print('GPU=' + str(torch.cuda.device_count()))"
```

Fresh result:

- The repo is on main at HEAD 2f7b241e4f4687d2aae263faf4b1d5c08eae9e5e.
- Python is 3.14.4.
- `import torch` fails with: "Failed to load PyTorch C extensions... It appears that PyTorch has loaded the torch/_C folder of the PyTorch repository rather than the C extensions..."

Fresh test-suite command:

```powershell
rmdir /s /q .pytest_cache .pytest-temp 2>$null; mkdir .pytest-temp 2>$null; py -m pytest -q --basetemp=.pytest-temp -p no:cacheprovider
```

Fresh result:

- Pytest does not reach the repository test logic.
- It crashes during collection because NumPy is broken: `ImportError: No module named 'numpy._core._multiarray_umath'`.

Scientific implication:

- The environment is not capable of fresh end-to-end validation in this session.
- Any claim of a fresh 5,000-area scientific result is unsupported by current runtime evidence.

## D. Dataset integrity

The repository documentation asserts the following dataset state:

- 5,000 BigEarthNet v2 areas total
- 4,600 train
- 200 validation
- 200 test
- exact selected multimodal dataset with optical, SAR, and reference targets

This is documented in [README.md](README.md) and [docs/PIPELINE3_5000_TRAINING_REPORT.md](docs/PIPELINE3_5000_TRAINING_REPORT.md).

Status:

- Documented condition: PASS (documentation only)
- Fresh runtime verification: BLOCKED by broken Python/NumPy/PyTorch environment

No fresh dataset-validation claim is warranted from this session.

## E. Split integrity

The repository documents a split of 4,600 / 200 / 200 and states that the 200 test areas were sealed before final evaluation.

Status:

- Documented condition: PASS in project documentation
- Fresh runtime verification: BLOCKED

No fresh split-proof claim is possible until the environment is repaired.

## F. Duplicate audit

The documentation proposes a duplicate and leakage audit as part of the Pipeline 3 validation story. It reports zero cross-split identity leakage and no repeated S1 plus area contamination in the documented checks.

Status:

- Documented condition: PASS in reports
- Fresh runtime verification: NOT PERFORMED in this session because the environment fails before scientific code imports

## G. Label/image alignment

The reported scientific pipeline depends on strict matching between optical, SAR, and reference maps for the same area. The project documents that raster IMG and reference maps are north-up, CRS-checked, and aligned to the same 120x120 common grid.

Status:

- Documented condition: PASS in project docs
- Fresh runtime verification: BLOCKED

## H. Feature leakage

The project documents that the current scientific predictor uses:

- 62-D physical features
- 768-D joint CROMA GAP
- total 830-D hybrid vector

This is described in [README.md](README.md) and [docs/CURRENT_PROJECT_STATUS.md](docs/CURRENT_PROJECT_STATUS.md).

Status:

- Documented condition: described as a controlled scientific representation
- Fresh runtime verification: BLOCKED

The audit cannot validate the actual feature path without a working Python stack.

## I. CROMA audit

The repo documents a canonical CROMA path and official checkpoint usage. It also documents the 225x768 token layout, 768-D GAP, and the hybrid 830-D representation.

Status:

- Documented condition: PASS in architecture and training reports
- Fresh runtime verification: BLOCKED

No fresh assertion about actual CROMA execution is possible under the current broken environment.

## J. Physical feature audit

The documented pipeline says physical features are deterministic and not learned. It also distinguishes the final 830-D predictor from the separate 192-D `HybridFusion` infrastructure.

Status:

- Documented condition: described and documented
- Fresh runtime verification: BLOCKED

## K. Model/checkpoint audit

The project documentation identifies a frozen scientific checkpoint and a documented result. It mentions seed 17 and epoch 54 selection.

Status:

- Documented condition: described in reports
- Fresh runtime verification: BLOCKED

No fresh checkpoint verification or model execution was possible in this session.

## L. Fresh training result

Not recomputed.

Reason: the environment fails before the repo can import NumPy or PyTorch. A fresh training run cannot be trusted without a repaired Python stack.

## M. Fresh 5,000-area inference result

Not recomputed.

Reason: the interpreter is not operational, and the scientific runtime cannot be imported.

## N. Fresh test metrics

Not recomputed.

Status: NOT RECOMPUTED.

The values 65.0% accuracy, 4.1034 pp MAE, and 9.5873 pp RMSE remain documentation-only results, not fresh measurements from this environment.

## O. Ablation comparison

The training report documents the following comparisons:

- Constant
- Physical
- Optical CROMA
- SAR CROMA
- Joint CROMA
- Hybrid

Status:

- Documentation: present
- Fresh recomputation: NOT RECOMPUTED in this session

## P. Evidence connectivity

The documentation states that representation generation, evidence generation, and constrained interpretation are wired into the runtime. However, the code path could not be executed in this session.

Status:

- Documented condition: likely intended/declared
- Fresh runtime evidence: unavailable because the environment cannot import scientific libraries

## Q. Architecture connectivity

The repository documents a typed architecture with runtime orchestration, evidence, and capability routing. The user request specifically requires an actual runtime trace, but no fresh end-to-end path could be executed here.

Status:

- Documented architecture: present
- Fresh runtime proof: UNVERIFIED

## R. API connectivity

The project says `/api/analyze` is the supported API path. No fresh API execution was possible because the Python environment is broken.

Status:

- Documented API flow: described
- Fresh execution: BLOCKED

## S. Frontend connectivity

The frontend is documented as connected to the backend API, but no browser/API check was possible here because the backend could not even import the scientific stack.

Status:

- Documented connectivity: described
- Fresh verification: BLOCKED

## T. Cache/receipt audit

The project documents receipt-based representation persistence and checksum validation, but no runtime cache or receipt audit could be performed under the broken environment.

Status:

- Documentation: present
- Fresh verification: BLOCKED

## U. Performance profile

No runtime profiling was possible in this session because the environment fails before importing the scientific stack.

Status:

- Fresh measurement: not available
- Bottleneck claim: not justified

## V. Failure-mode results

No failure-mode audit could be executed because the Python stack is non-functional.

## W. Reproducibility

The project documents reproducibility checks and scientific fingerprint stability, but the current environment does not allow rerunning or validating them.

Current evidence:

- Scientific fingerprint reported in docs: `ac8bbefc8918b2ee16f47653e6fd91eba0d875e4ce340e27254248712a173fae`
- Fresh independent reproduction: not possible in this session

## X. Scientific fingerprint

The repo documents a scientific fingerprint, but without a fresh runtime run, it cannot be validated against the current environment.

Status:

- Documented fingerprint: present
- Fresh verification: NOT POSSIBLE

## Y. Annotation integration safety

The annotation foundation and protocol documents explicitly state that the spatial annotation layer remains separate and does not implement raster/CROMA mapping. This is a documented scientific boundary, not a runtime validation.

Status:

- Documented safety policy: present and conservative
- Fresh integration validation: blocked by broken environment

## Z. Final verdict

### Layer classification

- GREEN: not justified by fresh evidence in this session
- YELLOW: some documentation exists, but no runtime proof
- RED: environment failure blocks all fresh validation

The current technical state is best classified as:

- Overall state: NEEDS REPAIR (runtime environment)
- Scientific state: FRESHLY UNVERIFIED

The real issue is not a suspected scientific bug in the logic; it is a broken execution environment. The repository cannot currently run the scientific code because the Python stack is invalid. This must be repaired before any meaningful 5,000-area end-to-end revalidation can be trusted.

## Final metrics table

| Metric | Previous | Fresh | Difference | Status |
|---|---:|---:|---:|---|
| Test accuracy | 65.0% | NOT RECOMPUTED | — | BLOCKED |
| Test MAE | 4.1034 pp | NOT RECOMPUTED | — | BLOCKED |
| Test RMSE | 9.5873 pp | NOT RECOMPUTED | — | BLOCKED |
| Train MAE | documented only | NOT RECOMPUTED | — | BLOCKED |
| Validation MAE | documented only | NOT RECOMPUTED | — | BLOCKED |
| Runtime | undocumented | NOT RECOMPUTED | — | BLOCKED |
| Peak GPU memory | undocumented | NOT RECOMPUTED | — | BLOCKED |
| Peak RAM | undocumented | NOT RECOMPUTED | — | BLOCKED |

## Required final answer

1. Did all 5,000 areas process successfully?  
   No fresh evidence. The environment is currently broken and cannot import NumPy/PyTorch.

2. Were any silently skipped?  
   Not determinable because the runtime cannot start; the current evidence does not support any fresh successful processing.

3. Any dataset leakage?  
   No fresh evidence; not verifiable under the current broken environment.

4. Any split leakage?  
   No fresh evidence; not verifiable under the current broken environment.

5. Any label/image misalignment?  
   No fresh evidence; not verifiable under the current broken environment.

6. Any feature leakage?  
   No fresh evidence; not verifiable under the current broken environment.

7. Any CROMA issue?  
   No fresh evidence; not verifiable under the current broken environment.

8. Any physical-feature issue?  
   No fresh evidence; not verifiable under the current broken environment.

9. Any checkpoint mismatch?  
   No fresh evidence; not verifiable under the current broken environment.

10. Is the 830-D model actually being used?  
   Not freshly verified; the runtime cannot import and execute.

11. Is the architecture actually connected?  
   Not freshly verified; the runtime cannot import and execute.

12. Is evidence actually connected?  
   Not freshly verified; the runtime cannot import and execute.

13. Is the API actually connected?  
   Not freshly verified; the runtime cannot import and execute.

14. Is frontend/backend connectivity verified?  
   No. Not verified in this session.

15. What is the fresh test accuracy?  
   NOT RECOMPUTED. Current environment blocks all fresh evaluation.

16. What is the fresh MAE?  
   NOT RECOMPUTED.

17. What is the fresh RMSE?  
   NOT RECOMPUTED.

18. What is the biggest measured bottleneck?  
   The current bottleneck is the broken Python/NumPy/PyTorch runtime itself; it prevents all scientific execution.

19. What components are only partially connected?  
   All scientific runtime components are effectively untested until the environment is repaired.

20. What needs fixing before VQA?  
   The immediate requirement is the Python dependency environment: repair the Python 3.14 + NumPy + PyTorch stack so the repo can import and execute. VQA should not start until then.

21. What can safely remain as-is?  
   The documentation and the conservative annotation protocol may remain as-is, but they are not a substitute for runtime validation. The scientific pipeline itself is not fresh-validated.

## Stop condition

This session stopped after the forensic audit. No VQA, no training, no model changes, and no architecture refactor were performed. The correct finding is that the runtime cannot currently support a valid fresh 5,000-area end-to-end scientific revalidation.
