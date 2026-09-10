from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, precision_recall_fscore_support


def compute_multilabel_metrics(targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    targets = np.asarray(targets, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if targets.shape != probabilities.shape or targets.ndim != 2:
        raise ValueError(f"Target/probability shapes must match [N,C]: {targets.shape} vs {probabilities.shape}")
    if not np.isfinite(probabilities).all() or not np.isfinite(targets).all():
        raise ValueError("Metrics require finite targets and probabilities")
    predictions = (probabilities >= threshold).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(targets, predictions, average=None, zero_division=0)
    return {
        "threshold": threshold,
        "sample_count": int(targets.shape[0]),
        "class_count": int(targets.shape[1]),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "exact_match_accuracy": float(np.all(targets == predictions, axis=1).mean()),
        "mean_average_precision": float(average_precision_score(targets, probabilities, average="macro")),
        "per_class": [{"precision": float(p), "recall": float(r), "f1": float(score), "support": int(s)} for p, r, score, s in zip(precision, recall, f1, support)],
    }
