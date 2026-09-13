# Fresh Scientific Evaluation

Date: 2026-09-11

Six raw BigEarthNet v2 areas were selected deterministically: two train, two validation, and two test. Each was read from S2, S1, and reference GeoTIFFs using full metadata identities. The run performed strict validation, preprocessing, official frozen CROMA extraction, scene-target aggregation, five trainable linear probes, validation selection, held-out inference, checkpointing, and artifact persistence.

Seed 17; eight epochs; AdamW; learning rate 0.001; weight decay 0.01; soft-target cross-entropy; batch 16. Runtime was 11.6966 seconds on CUDA. Artifacts are under `experiments/smoke/`.

| Model | MAE (pp) | RMSE (pp) | Bias (pp) | Dominant Accuracy |
|---|---:|---:|---:|---:|
| Constant | 7.5965 | 18.2982 | 0.00000010 | 50% |
| Physical | 5.4574 | 12.3221 | 0.00000024 | 0% |
| Optical CROMA GAP | 10.5263 | 26.2817 | -0.00000022 | 0% |
| SAR CROMA GAP | 9.0199 | 20.6863 | -0.00000010 | 0% |
| Joint CROMA GAP | 10.5263 | 26.0735 | 0.00000004 | 0% |
| Physical + joint CROMA | 10.5263 | 26.7673 | 0.00000016 | 0% |

Physical selected epoch 1, optical 7, SAR 1, joint 4, and hybrid 8. Losses changed and five checkpoints were written. Independent metrics exactly match saved predictions. With only two examples per split this is an end-to-end smoke, not a generalization benchmark.

A fresh scan of the 9,553,962-row BigEarthNet.txt Parquet matched 955/1,000 identities: 20,453 records comprising 7,529 binary Q&A, 6,818 MCQ, 955 captions, and 5,151 boxes. Strings were preserved/nonempty. This is linkage, not model validation.

The historical 600/200/200 token result remains 6.1553 pp MAE and 53.8743% accuracy versus 9.2137 pp constant MAE. Its prepared directories remained inaccessible, so it was not freshly reproduced.
