from src.dataset_loader import discover_samples, load_sample
from src.gee_experiment_runtime import gee_status, select_mode


def load_local_samples(dataset_root, sample_ids):
    samples = {sample.patch_id: sample for sample in discover_samples(dataset_root)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in samples]
    if missing:
        raise FileNotFoundError(f"Missing local reference samples: {missing}")
    return [(sample, load_sample(sample)) for sample_id in sample_ids for sample in [samples[sample_id]]]
