# Pipeline 1 Archive: So2Sat-LCZ42 FusionCNN

**Status: archived and reproduction-only. This is not a production entry point.**

These scripts preserve the completed So2Sat-LCZ42 S1/S2 FusionCNN experiment
identified as Pipeline 1 in
`experiments/reports/autonomous_execution_report.json`. They use an external
HDF5 dataset and checkpoint directory, import optional historical dependencies
(`h5py` and scikit-learn), and are intentionally not imported by `src/`.

The fixed paths and original scientific behavior are retained for auditability.
Do not use this classifier as the SatQuery API model: it has a different dataset,
band contract, class ontology, preprocessing, model, and output contract from
the canonical Pipeline 3 architecture.
