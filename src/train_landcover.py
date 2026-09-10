from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .hybrid_fusion import FeatureModeFusion, LandCoverClassificationHead
from .multilabel_metrics import compute_multilabel_metrics
from .training_data import load_manifest


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_split(cache_root: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gee, croma, labels = [], [], []
    for sample_id in ids:
        root = cache_root / sample_id
        gee.append(np.load(root / "gee_features.npy"))
        croma.append(np.load(root / "croma_pooled.npy"))
        labels.append(np.load(root / "labels.npy"))
    if not gee:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    return np.stack(gee).astype(np.float32), np.stack(croma).astype(np.float32), np.stack(labels).astype(np.float32)


def validate_training_population(manifest) -> None:
    ids = set().union(*manifest.splits.values())
    if len(ids) <= 3 or ids.issubset({"61_39", "61_40", "61_41"}):
        raise RuntimeError("The three-sample development dataset is for inference and pipeline validation only; supervised training is prohibited.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the scalable BigEarthNet hybrid multi-label head.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument("--feature-mode", choices=("gee_only", "croma_only", "hybrid"), default="hybrid")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = load_manifest(args.split_manifest)
    validate_training_population(manifest)
    total = sum(len(ids) for ids in manifest.splits.values())
    if total < args.min_samples and not args.allow_small_dataset:
        raise RuntimeError(f"Refusing supervised training on only {total} samples; provide a larger dataset or pass --allow-small-dataset explicitly.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    set_seed(manifest.seed)
    train_gee, train_croma, train_y = load_split(args.cache_root, manifest.splits["train"])
    val_gee, val_croma, val_y = load_split(args.cache_root, manifest.splits["validation"])
    if not len(train_y) or not len(val_y):
        raise RuntimeError("Training and validation splits must both contain samples")
    model = FeatureModeFusion(args.feature_mode, train_gee.shape[1], train_croma.shape[1]).to(device)
    head = LandCoverClassificationHead(model.output_dim, len(manifest.class_names)).to(device)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-3, weight_decay=1e-4)
    pos = train_y.sum(axis=0); neg = len(train_y) - pos
    pos_weight = torch.tensor(np.maximum(neg, 1) / np.maximum(pos, 1), dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_gee), torch.from_numpy(train_croma), torch.from_numpy(train_y)), batch_size=args.batch_size, shuffle=True)
    best = -1.0; stale = 0; history = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); head.train(); total_loss = 0.0
        for gee, croma, labels in train_loader:
            logits = head(model(gee.to(device), croma.to(device)))
            loss = loss_fn(logits, labels.to(device)); optimizer.zero_grad(); loss.backward(); optimizer.step(); total_loss += float(loss.item()) * len(labels)
        model.eval(); head.eval()
        with torch.no_grad():
            val_prob = head.probabilities(model(torch.from_numpy(val_gee).to(device), torch.from_numpy(val_croma).to(device))).cpu().numpy()
        metrics = compute_multilabel_metrics(val_y, val_prob, threshold=args.threshold)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_y), **{f"validation_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}}
        history.append(row)
        checkpoint = {"epoch": epoch, "model_state": model.state_dict(), "head_state": head.state_dict(), "optimizer_state": optimizer.state_dict(), "config": {"feature_mode": args.feature_mode, "gee_dim": train_gee.shape[1], "croma_dim": train_croma.shape[1], "output_dim": model.output_dim, "class_names": manifest.class_names, "seed": manifest.seed, "threshold": args.threshold}}
        torch.save(checkpoint, args.output_root / "latest.pt")
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]; stale = 0; torch.save(checkpoint, args.output_root / "best.pt")
        else:
            stale += 1
        if stale >= 5: break
    report = {"status": "success", "feature_mode": args.feature_mode, "sample_counts": {key: len(value) for key, value in manifest.splits.items()}, "feature_dimensions": {"gee": int(train_gee.shape[1]), "croma": int(train_croma.shape[1]), "model_output": int(model.output_dim)}, "label_distribution": {name: int(train_y[:, index].sum()) for index, name in enumerate(manifest.class_names)}, "device": str(device), "seed": manifest.seed, "threshold": args.threshold, "best_validation_macro_f1": best, "history": history, "split_manifest": str(args.split_manifest), "checkpoint": str(args.output_root / "best.pt"), "runtime_seconds": time.perf_counter() - started, "confidence": "uncalibrated sigmoid probabilities"}
    (args.output_root / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
