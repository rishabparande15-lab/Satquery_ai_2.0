# Small pixel-feature experiment

## Design

This is an 18-area token-level ridge feasibility probe, not final scientific performance. It uses the existing Pass 3 area split: first eligible 10 train, 4 validation and 4 test areas encountered in the pinned prepared manifest. Fully labelled tokens only are retained, matching the established target mask. No area occurs in multiple splits. The target is the existing 19-class 8×8-pixel fraction per token.

Model A broadcasts the existing 62 scene physical features to each token. Model B adds 12 token features: mean/std for NDVI, NDWI, NDBI, BSI, VV−VH and 5×5 NDVI local std. Features are standardized from training data only. Ridge alpha is selected from 0.1/1/10/100 on validation only; the test set is evaluated once after selection. Both selected alpha 100.

| Model | Dimensions | Validation MAE | Test MAE |
|---|---:|---:|---:|
| A: physical | 62 | 10.5343 pp | 10.1534 pp |
| B: physical + selected token | 74 | 10.7071 pp | 9.9625 pp |

The addition improves test MAE by 0.1910 pp but worsens validation by 0.1728 pp. With only four validation and four test areas, this is unstable and must not be described as a validated gain. It does show that local/token values permit spatially varying predictions whereas scene-only physical inputs are identical across all tokens in an area.

Across the 18 scenes, candidate scene means were deliberately redundant: NDVI, NDWI and NDBI correlated essentially 1.0 with their existing physical entries; BSI mean reached 0.994 with an existing physical feature and VV−VH mean 0.876. Token-distribution standard deviations were less redundant (maximum absolute correlations with any physical dimension: NDVI 0.739, NDWI 0.791, NDBI 0.644, BSI 0.564, VV−VH 0.542). This supports retaining spatial distribution features, not adding duplicate scene means. These correlations are exploratory and unstable at n=18.

No CROMA features were retrained or replaced, no main experiment was rerun, and no result file under `experiments/pass3` was modified. Exact area identities, token counts, selected features and results are recorded in `artifacts/pixel_feature_probe/61_39/experiment.json`.
