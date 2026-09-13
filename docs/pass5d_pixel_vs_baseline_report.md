# Pass 5D pixel-enhanced versus current pipeline

## Result

This is **Mode A: reproduction comparison** on the exact Pass 3 600/200/200 area split. It is not independent generalization: all 1,000 locally S2-complete areas occur in Pass 3. Version A exactly reproduced the validated Hybrid metrics, confirming the protocol. Version B produced slightly lower MAE/RMSE but lower dominant accuracy; the paired MAE interval crosses zero. The predictive winner is therefore **inconclusive**.

| Metric | Version A | Version B | Absolute change (B−A) | Relative change |
|---|---:|---:|---:|---:|
| MAE ↓ | 4.884571 pp | 4.846286 pp | −0.038286 pp | −0.784% |
| RMSE ↓ | 11.025107 pp | 10.936235 pp | −0.088872 pp | −0.806% |
| Bias | −0.000000016 pp | +0.000000010 pp | +0.000000026 pp | practically zero for both |
| Dominant accuracy ↑ | 59.5% | 58.5% | −1.0 percentage point | −1.681% |

The paired, area-level bootstrap estimate for MAE change was −0.038286 pp with 95% interval `[−0.121147, +0.043515]`; 104/200 test areas improved and 96 worsened. This does not establish a dependable prediction improvement.

## Dataset and protocol

The primary archive was `E:\SatQuery_ai_2.0\datasets_2.0\bigearthnet-v2-5000-20260911T162804Z-1-001.zip`, SHA-256 `B3471E5650BD263367BCF4CC650405AB2981A86621579E21A1B1D08AF630C595`. Current exhaustive CRC checks passed for the outer ZIP, 5,000-member reference ZIP and 10,000-member S1 ZIP. The fragmented archive outer CRC passed but remains unusable.

Matching S2 is at `D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2`: exactly 1,000 primary IDs, all Pass 3 IDs, each with 12 bands. The remaining 4,000 primary areas have no matching local S2. No archive was merged and no data were downloaded.

Version A uses 62 established physical features plus 768 joint CROMA GAP values (830 dimensions). Version B adds only Pass 5B-selected token summaries: mean/std per 8×8 block for NDVI, NDWI, NDBI, BSI, VV−VH and 5×5 NDVI local std, then mean/std across 225 tokens (24 dimensions; total 854). Both used identical scene-level 19-class labelled-pixel coverage targets, linear softmax head, seed 17, AdamW 0.001, batch size 1,024, 60-epoch cap, patience 10, training-only standardization and validation-only model selection. Both selected epoch 60.

The task remains whole-area coverage prediction. Token-level predictive metrics are not reported because these heads emit one scene distribution; broadcasting that output to tokens would be misleading. Spatial evidence is evaluated separately without claiming segmentation.

## Cost and decision

Frozen-artifact loading took 5.723 s. Existing physical extraction took 0.893 s total (0.893 ms/area); the new pixel/local/token calculation took 71.529 s total (71.529 ms/area). The in-memory feature matrices were 3,320,000 versus 3,416,000 bytes (+2.89%). Saved Version A/B artifact bundles were 264,750/266,619 bytes. Recorded training times (1.508/0.165 s) are not directly comparable because Version A incurred CUDA cold-start; inference was below 0.2 ms for all 200 areas in both runs. Frozen CROMA extraction was not rerun, so no new CROMA timing is claimed.

Decision-framework outcome: **B — pixel features improve spatial/evidence quality, but do not establish an overall predictive improvement**. Keep the current predictor as default and retain the pixel layer as a spatial-evidence sidecar pending an independent S2-complete generalization experiment.
