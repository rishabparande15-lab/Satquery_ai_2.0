# Pass 5D reproducibility

Version A exactly reproduced the historical Hybrid result: MAE 4.8846 pp, RMSE 11.0251 pp, effectively zero bias and 59.5% dominant accuracy. This independently confirms the reproduction split, target and training protocol.

For both versions, held-out inference was executed twice from the selected in-memory checkpoint. Maximum absolute prediction difference was `0.0`. Stored metrics were independently recomputed from saved predictions and truth; maximum difference was `0.0` for both versions.

Configuration, training history, selected epoch, checkpoint, predictions and metrics are stored separately under `experiments/pass5d/version_a` and `version_b`. Checkpoint SHA-256 values are:

- A: `dbc7115d3c7957942aba6991d63fd3eea3bc908834cb5071c28750bfceab2229`
- B: `0eaa677be7ab0a6e396a077358f4f463264991b759ebb5ac581900178cda6fd3`

Deterministic tolerance is exact equality on the recorded Windows/PyTorch/CUDA environment. Cross-device retraining is not claimed bitwise deterministic. The primary and fragmented archive SHA-256 values were recomputed, and current outer/nested CRC checks found no bad members.

Robustness perturbations were not run: the frozen CROMA artifacts cannot be recomputed consistently under new masks, so perturbing only the pixel branch would violate the controlled A/B design. This limitation is recorded rather than presenting an unfair supplementary result.
