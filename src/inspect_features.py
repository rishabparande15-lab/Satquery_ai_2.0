import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import PROJECT_ROOT, get_settings
from .dataset_loader import OPTICAL_BANDS, SAR_BANDS, discover_samples, load_sample


LOGGER = logging.getLogger(__name__)
FEATURE_NAMES = {
    "SAR_GAP": "SAR pooled",
    "optical_GAP": "Optical pooled",
    "joint_GAP": "Joint pooled",
}
ENCODING_NAMES = {
    "SAR_encodings": "SAR tokens",
    "optical_encodings": "Optical tokens",
    "joint_encodings": "Joint tokens",
}


def _scale_for_display(array: np.ndarray) -> np.ndarray:
    values = array[np.isfinite(array)]
    if values.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0, 1).astype(np.float32)


def _rgb(optical: np.ndarray) -> np.ndarray:
    return np.stack([_scale_for_display(optical[index]) for index in (3, 2, 1)], axis=-1)


def _save_figure(figure: plt.Figure, path: Path) -> str:
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(path)


def _plot_rgb(optical: np.ndarray, path: Path, title: str) -> str:
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(_rgb(optical))
    axis.set_title(title)
    axis.set_axis_off()
    return _save_figure(figure, path)


def _plot_optical_bands(optical: np.ndarray, path: Path, title: str) -> str:
    figure, axes = plt.subplots(3, 4, figsize=(12, 9))
    for index, (axis, band) in enumerate(zip(axes.flat, OPTICAL_BANDS)):
        image = axis.imshow(_scale_for_display(optical[index]), cmap="viridis")
        axis.set_title(band)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(title)
    return _save_figure(figure, path)


def _plot_sar(sar: np.ndarray, path: Path, title: str) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, index, band in zip(axes, range(2), SAR_BANDS):
        image = axis.imshow(_scale_for_display(sar[index]), cmap="magma")
        axis.set_title(band)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(title)
    return _save_figure(figure, path)


def _plot_sar_difference_ratio(sar: np.ndarray, path: Path, title: str) -> str:
    vv, vh = sar
    difference = vv - vh
    ratio = vv / (np.abs(vh) + 1e-6)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, image, label in zip(axes, (difference, ratio), ("VV - VH", "VV / abs(VH)")):
        plotted = axis.imshow(_scale_for_display(image), cmap="coolwarm")
        axis.set_title(label)
        axis.set_axis_off()
        figure.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(title)
    return _save_figure(figure, path)


def _plot_pooled_vectors(features: dict[str, np.ndarray], path: Path, title: str) -> str:
    figure, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for axis, name in zip(axes, ("SAR_GAP", "optical_GAP", "joint_GAP")):
        vector = features[name].reshape(-1)
        axis.plot(vector, linewidth=0.7)
        axis.set_ylabel(FEATURE_NAMES[name])
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Embedding dimension")
    figure.suptitle(title)
    return _save_figure(figure, path)


def _plot_token_maps(features: dict[str, np.ndarray], path: Path, title: str) -> str:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, name in zip(axes, ("SAR_encodings", "optical_encodings", "joint_encodings")):
        tokens = features[name]
        if tokens.shape[1] != 225:
            raise ValueError(f"{name} must contain 225 tokens, got {tokens.shape}")
        token_norms = np.linalg.norm(tokens[0], axis=-1).reshape(15, 15)
        plotted = axis.imshow(token_norms, cmap="plasma")
        axis.set_title(ENCODING_NAMES[name])
        axis.set_xlabel("token column")
        axis.set_ylabel("token row")
        figure.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"{title} (15 x 15 token grid)")
    return _save_figure(figure, path)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    value = float(np.dot(left, right) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _similarity_matrix(vectors: list[np.ndarray]) -> list[list[float]]:
    return [[_cosine(left, right) for right in vectors] for left in vectors]


def _plot_similarity_matrices(matrices: dict[str, list[list[float]]], sample_ids: list[str], path: Path) -> str:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (name, matrix) in zip(axes, matrices.items()):
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_title(FEATURE_NAMES[name])
        axis.set_xticks(range(len(sample_ids)), sample_ids)
        axis.set_yticks(range(len(sample_ids)), sample_ids)
        for row in range(len(sample_ids)):
            for column in range(len(sample_ids)):
                axis.text(column, row, f"{matrix[row][column]:.3f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("CROMA pooled-vector cosine similarity")
    return _save_figure(figure, path)


def _plot_pca(pooled: dict[str, dict[str, np.ndarray]], sample_ids: list[str], path: Path) -> str:
    figure, axis = plt.subplots(figsize=(7, 6))
    colors = {"SAR_GAP": "tab:orange", "optical_GAP": "tab:green", "joint_GAP": "tab:blue"}
    for name, sample_vectors in pooled.items():
        matrix = np.stack([sample_vectors[sample_id].reshape(-1) for sample_id in sample_ids])
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        if matrix.shape[0] > 1:
            coordinates = centered @ np.linalg.svd(centered, full_matrices=False)[2][:2].T
        else:
            coordinates = np.zeros((1, 2))
        for index, sample_id in enumerate(sample_ids):
            axis.scatter(coordinates[index, 0], coordinates[index, 1], color=colors[name], label=FEATURE_NAMES[name] if index == 0 else None)
            axis.annotate(sample_id, (coordinates[index, 0], coordinates[index, 1]), xytext=(4, 4), textcoords="offset points")
    axis.set_title("PCA of pooled CROMA embeddings")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(alpha=0.25)
    axis.legend()
    return _save_figure(figure, path)


def _feature_paths_from_report(report: dict, features_root: Path) -> dict[str, dict[str, Path]]:
    result = {}
    for sample in report.get("samples", []):
        result[sample["patch_id"]] = {
            name: Path(info["path"]) for name, info in sample.get("outputs", {}).items()
        }
    if not result:
        raise ValueError(f"No samples found in {features_root / 'verification_report.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect BigEarthNet inputs and saved CROMA features.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override the D: inspection output directory.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    output_root = args.output_root or settings.features_root / "inspection"
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = settings.features_root / "verification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    feature_paths = _feature_paths_from_report(report, settings.features_root)
    result = {
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(settings.dataset_root),
        "features_root": str(settings.features_root),
        "output_root": str(output_root),
        "samples_processed": [],
        "similarity": {},
        "visualization_paths": [],
        "warnings": [],
        "errors": [],
        "command": "cmd /c python -m src.inspect_features",
    }
    prepared_by_id = {}
    try:
        samples = discover_samples(settings.dataset_root)
    except Exception as error:
        result["errors"].append(str(error))
        (settings.features_root / "inspection_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise
    for sample in samples:
        sample_result = {"sample_id": sample.patch_id, "input_file_paths": {}, "feature_file_paths": {}, "feature_shapes": {}, "visualizations": [], "warnings": [], "errors": []}
        try:
            item = load_sample(sample)
            prepared_by_id[sample.patch_id] = item
            sample_result["input_file_paths"] = {
                "optical": {band: str(path) for band, path in sample.optical_paths.items()},
                "sar": {band: str(path) for band, path in sample.sar_paths.items()},
                "reference_map": str(sample.reference_map),
            }
            if item.optical.shape != (12, 120, 120) or item.sar.shape != (2, 120, 120):
                raise ValueError(f"Unexpected loader shapes for {sample.patch_id}")
            if not np.isfinite(item.optical).all() or not np.isfinite(item.sar).all():
                raise ValueError(f"Non-finite preprocessed values for {sample.patch_id}")
            paths = feature_paths.get(sample.patch_id, {})
            features = {name: np.load(path) for name, path in paths.items()}
            required = set(FEATURE_NAMES) | set(ENCODING_NAMES)
            missing = required - set(features)
            if missing:
                raise FileNotFoundError(f"Missing feature files for {sample.patch_id}: {sorted(missing)}")
            for name, array in features.items():
                sample_result["feature_file_paths"][name] = str(paths[name])
                sample_result["feature_shapes"][name] = list(array.shape)
            if any(features[name].shape != (1, 768) for name in FEATURE_NAMES):
                raise ValueError(f"Unexpected pooled shape for {sample.patch_id}")
            if any(features[name].shape != (1, 225, 768) for name in ENCODING_NAMES):
                raise ValueError(f"Unexpected token shape for {sample.patch_id}")
            sample_dir = output_root / sample.patch_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            plots = [
                _plot_rgb(item.raw_optical, sample_dir / "sentinel2_rgb.png", f"{sample.patch_id} Sentinel-2 RGB (B04/B03/B02)"),
                _plot_optical_bands(item.raw_optical, sample_dir / "sentinel2_bands.png", f"{sample.patch_id} Sentinel-2 bands"),
                _plot_sar(item.raw_sar, sample_dir / "sentinel1_vv_vh.png", f"{sample.patch_id} Sentinel-1 backscatter"),
                _plot_sar_difference_ratio(item.raw_sar, sample_dir / "sentinel1_difference_ratio.png", f"{sample.patch_id} Sentinel-1 comparison"),
                _plot_rgb(item.optical, sample_dir / "preprocessed_optical_rgb.png", f"{sample.patch_id} preprocessed optical"),
                _plot_sar(item.sar, sample_dir / "preprocessed_sar.png", f"{sample.patch_id} preprocessed SAR"),
                _plot_pooled_vectors(features, sample_dir / "pooled_embeddings.png", f"{sample.patch_id} pooled CROMA embeddings"),
                _plot_token_maps(features, sample_dir / "token_maps_15x15.png", sample.patch_id),
            ]
            sample_result["visualizations"] = plots
            result["visualization_paths"].extend(plots)
        except Exception as error:
            sample_result["errors"].append(str(error))
            result["errors"].append(f"{sample.patch_id}: {error}")
        result["samples_processed"].append(sample_result)
    successful_ids = [sample_id for sample_id, item in prepared_by_id.items() if not next(entry for entry in result["samples_processed"] if entry["sample_id"] == sample_id)["errors"]]
    pooled = {name: {} for name in FEATURE_NAMES}
    for sample_id in successful_ids:
        for name in FEATURE_NAMES:
            pooled[name][sample_id] = np.load(feature_paths[sample_id][name])
    if len(successful_ids) == len(samples):
        matrices = {name: _similarity_matrix([pooled[name][sample_id] for sample_id in successful_ids]) for name in FEATURE_NAMES}
        result["similarity"] = {name: {"sample_ids": successful_ids, "matrix": matrix} for name, matrix in matrices.items()}
        similarity_plot = _plot_similarity_matrices(matrices, successful_ids, output_root / "similarity_matrices.png")
        pca_plot = _plot_pca(pooled, successful_ids, output_root / "pca_pooled_embeddings.png")
        result["visualization_paths"].extend([similarity_plot, pca_plot])
        if not all(-1 <= value <= 1 and np.isfinite(value) for matrix in matrices.values() for row in matrix for value in row):
            result["errors"].append("Cosine similarity contained a non-finite or out-of-range value")
    else:
        result["warnings"].append("Cross-sample similarity skipped because one or more samples failed")
    result["successful_visualization_count"] = len(result["visualization_paths"])
    result["samples_processed_count"] = len(successful_ids)
    output_report = settings.features_root / "inspection_report.json"
    output_report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("inspection report=%s visualizations=%d samples=%d", output_report, result["successful_visualization_count"], len(successful_ids))
    if result["errors"]:
        raise RuntimeError(f"Inspection completed with errors: {result['errors']}")


if __name__ == "__main__":
    main()