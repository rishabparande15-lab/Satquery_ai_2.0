import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(r"<machine-local-data-root>\github_datasets\So2Sat-LCZ42")
EXP = ROOT / "experiment"
DATA = ROOT / "data"
WORKSPACE = Path(r"<repo-root>")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiments.github_sen12ms_training.train import CLASS_NAMES, FusionCNN, H5Dataset


def compute_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "class_support": {cls: int(report[cls]["support"]) for cls in CLASS_NAMES},
        "per_class_metrics": {
            cls: {
                "precision": float(report[cls].get("precision", 0.0)),
                "recall": float(report[cls].get("recall", 0.0)),
                "f1": float(report[cls].get("f1-score", report[cls].get("f1", 0.0))),
                "support": int(report[cls].get("support", 0)),
            }
            for cls in CLASS_NAMES
        },
    }
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = EXP / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get("model_state")
    if model_state is None:
        raise KeyError(f"Checkpoint does not contain 'model_state': {checkpoint_path}")

    model = FusionCNN().to(device)
    model.load_state_dict(model_state)
    model.eval()

    history_path = EXP / "reports" / "training_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {"history": []}
    best_row = min(history.get("history", []), key=lambda row: row["validation_loss"], default={})

    ds = H5Dataset(DATA / "testing.h5")
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    all_true = []
    all_pred = []
    all_probs = []
    confidences = []
    total_loss = 0.0
    total_count = 0
    csv_rows = []
    backend_sample = None

    started = time.perf_counter()
    with torch.no_grad():
        for sen1, sen2, labels, indices in loader:
            sen1 = sen1.to(device)
            sen2 = sen2.to(device)
            labels = labels.to(device)
            logits = model(sen1, sen2)
            loss = nn.functional.cross_entropy(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_count += labels.size(0)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().tolist()
            labels_list = labels.cpu().tolist()
            all_true.extend(labels_list)
            all_pred.extend(preds)
            all_probs.append(probs)
            batch_conf = probs[np.arange(len(preds)), preds]
            confidences.extend(batch_conf.tolist())

            for row_idx, sample_idx in enumerate(indices.tolist()):
                true_idx = int(labels_list[row_idx])
                pred_idx = int(preds[row_idx])
                conf = float(batch_conf[row_idx])
                entry = {
                    "dataset_index": int(sample_idx),
                    "true_label": true_idx,
                    "true_label_name": CLASS_NAMES[true_idx],
                    "predicted_label": pred_idx,
                    "predicted_label_name": CLASS_NAMES[pred_idx],
                    "confidence": conf,
                    "correct": true_idx == pred_idx,
                    "probabilities": {CLASS_NAMES[i]: float(probs[row_idx, i]) for i in range(len(CLASS_NAMES))},
                }
                csv_rows.append(
                    [
                        int(sample_idx),
                        true_idx,
                        CLASS_NAMES[true_idx],
                        pred_idx,
                        CLASS_NAMES[pred_idx],
                        conf,
                        bool(true_idx == pred_idx),
                    ]
                )
                if backend_sample is None:
                    backend_sample = entry

    metrics = compute_metrics(all_true, all_pred)
    metrics.update(
        {
            "dataset": "So2Sat-LCZ42",
            "split": "testing",
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "runtime_seconds": float(time.perf_counter() - started),
            "test_loss": float(total_loss / total_count),
            "epochs_ran": int(len(history.get("history", []))),
            "best_validation_epoch": int(best_row.get("epoch", 0)) if best_row else 0,
            "best_validation_loss": float(best_row.get("validation_loss", float("nan"))) if best_row else float("nan"),
            "best_validation_accuracy": float(best_row.get("validation_accuracy", float("nan"))) if best_row else float("nan"),
            "best_validation_macro_f1": float(best_row.get("validation_macro_f1", float("nan"))) if best_row else float("nan"),
            "checkpoint_loads_for_inference": True,
            "prediction_confidence_stats": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "median": float(np.median(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences)),
                "p10": float(np.percentile(confidences, 10)),
                "p25": float(np.percentile(confidences, 25)),
                "p75": float(np.percentile(confidences, 75)),
                "p90": float(np.percentile(confidences, 90)),
            },
        }
    )

    if backend_sample is not None:
        backend_sample["model"] = "FusionCNN_S1_S2"
        backend_sample["modalities"] = ["Sentinel-1", "Sentinel-2"]
        backend_sample["input_shape"] = {
            "sen1": [8, 32, 32],
            "sen2": [10, 32, 32],
        }
        backend_sample["top1_probability"] = backend_sample["confidence"]
        backend_sample["softmax_probabilities"] = backend_sample["probabilities"]

    np.save(EXP / "predictions" / "test_probabilities.npy", np.concatenate(all_probs, axis=0))

    csv_path = EXP / "predictions" / "test_predictions.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "dataset_index",
            "true_label",
            "true_label_name",
            "predicted_label",
            "predicted_label_name",
            "confidence",
            "correct",
        ])
        writer.writerows(csv_rows)

    prediction_json = [
        {
            "dataset_index": int(row[0]),
            "true_label": int(row[1]),
            "true_label_name": row[2],
            "predicted_label": int(row[3]),
            "predicted_label_name": row[4],
            "confidence": float(row[5]),
            "correct": bool(row[6]),
        }
        for row in csv_rows
    ]
    (EXP / "predictions" / "test_predictions.json").write_text(json.dumps(prediction_json, indent=2))

    if backend_sample is not None:
        (EXP / "predictions" / "backend_inference_sample.json").write_text(json.dumps(backend_sample, indent=2))

    (EXP / "metrics" / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    (EXP / "metrics" / "confusion_matrix.csv").write_text(
        "\n".join(",".join(map(str, row)) for row in metrics["confusion_matrix"]) + "\n"
    )
    (EXP / "metrics" / "confusion_matrix.json").write_text(json.dumps(metrics["confusion_matrix"], indent=2))

    model_summary = {
        "model_name": "FusionCNN",
        "model_type": "Multimodal Sentinel-1 + Sentinel-2 LCZ classifier",
        "modalities": ["Sentinel-1", "Sentinel-2"],
        "input_channels": {"sen1": 8, "sen2": 10},
        "classes": CLASS_NAMES,
        "number_of_epochs": int(len(history.get("history", []))),
        "best_validation_epoch": int(best_row.get("epoch", 0)) if best_row else 0,
        "best_validation_result": {
            "validation_loss": float(best_row.get("validation_loss", float("nan"))) if best_row else float("nan"),
            "validation_accuracy": float(best_row.get("validation_accuracy", float("nan"))) if best_row else float("nan"),
            "validation_macro_f1": float(best_row.get("validation_macro_f1", float("nan"))) if best_row else float("nan"),
        },
        "test_results": {
            "test_loss": float(metrics["test_loss"]),
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
        },
        "class_imbalance_effects": {
            "summary": "The LCZ dataset is class-imbalanced across built-environment and land-cover categories, so macro metrics are more informative than raw accuracy. Minority classes can be underrepresented in training and therefore show lower recall and F1 despite good overall accuracy.",
            "support": metrics["class_support"],
        },
        "limitations": [
            "Only one epoch was executed in the saved training run, so the model is a lightweight baseline rather than a fully converged classifier.",
            "The input patches are 32x32, so the model is limited to local spatial context and does not explicitly model broader scene context.",
            "The dataset contains known class imbalance across LCZ categories, which influences per-class performance and macro averages.",
            "Model quality is partially constrained by the single-epoch baseline training configuration used in this isolated experiment."
        ],
        "checkpoint_usage": {
            "best_checkpoint": str(checkpoint_path),
            "inference_verified": True,
        },
    }

    (EXP / "reports" / "final_model_quality_report.json").write_text(json.dumps(model_summary, indent=2))
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "test_loss": float(metrics["test_loss"]),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "epochs_ran": int(len(history.get("history", []))),
        "best_validation_epoch": int(best_row.get("epoch", 0)) if best_row else 0,
        "checkpoints_exist": [str(p.name) for p in sorted((EXP / "checkpoints").glob("*.pt"))],
        "prediction_csv": str(EXP / "predictions" / "test_predictions.csv"),
        "prediction_json": str(EXP / "predictions" / "test_predictions.json"),
        "inference_json": str(EXP / "predictions" / "backend_inference_sample.json"),
        "metrics_json": str(EXP / "metrics" / "evaluation_metrics.json"),
        "final_report": str(EXP / "reports" / "final_model_quality_report.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
