# SatQuery AI Data Leakage Audit

Date: 2026-09-11

The fresh smoke used six full areas: two train, two validation, and two test. No block-level splitting occurred. Test areas were `S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN_49_49` and `S2B_MSIL2A_20180225T105019_N9999_R051_T31UER_68_32`; all identities are in `experiments/smoke/split.json`.

- Test used for training or epoch selection: no.
- Validation used for gradients: no.
- Standardization: train mean/std only, folded into each selected head.
- Features: frozen CROMA and label-independent physical statistics.
- Targets never enter feature extraction.
- Spatial leakage: absent because whole areas are assigned once.

Independent reload reproduced every saved MAE, RMSE, and bias with maximum absolute difference `0.0`. The historical 600/200/200 experiment separately reports disjoint patch/S1 identities but was not freshly recomputed.
