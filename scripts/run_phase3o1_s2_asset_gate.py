"""Recover existing raw S2 assets and run the Phase 3O.1 input gate; never train."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import psutil
import torch
import rasterio
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.dataset_loader import _read_to_grid
from src.preprocessing import _robust_channel_scale
from src.eo_vlm.multispectral_projector import S2MultispectralProjector
from src.eo_vlm.s2_asset_gate import S2_BANDS, deterministic_manifest, inspect_area, training_gate


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def path_index(roots: list[Path]) -> dict[str, Path]:
    index = {}
    for root in roots:
        for path in root.rglob("*.tif"):
            if "_B" in path.stem:
                index.setdefault(path.name, path)
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2-root", type=Path, action="append", required=True)
    parser.add_argument("--annotations", type=Path, default=ROOT / "artifacts/annotations/image_language/visual_vqa_candidates.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/training/phase3o")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if not 5 <= args.limit <= 20: raise ValueError("limit must be between 5 and 20")
    index = path_index([root.resolve() for root in args.s2_root if root.is_dir()])
    accepted, selected = [], {}
    rejected = []
    seen = set()
    with args.annotations.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if len(accepted) == args.limit: break
            area = record.get("image_id")
            if area in seen or record.get("split") != "train" or record.get("source_category") == "relative pos": continue
            paths = {band: index.get(f"{area}_{band}.tif") for band in S2_BANDS}
            if not all(paths.values()): continue
            try:
                asset = inspect_area(record, paths)
            except ValueError as exc:
                rejected.append({"image_id": area, "record_id": record.get("record_id"), "reason": str(exc)}); continue
            seen.add(area); accepted.append(asset); selected[asset.sample_id] = record
    asset_payload = {"phase": "3O.1", "status": "VERIFIED" if accepted else "BLOCKED", "asset_count": len(accepted),
                     "rejected_count": len(rejected), "records": [item.to_dict() for item in accepted], "rejections": rejected}
    write_json(args.output / "verified_s2_training_assets.json", asset_payload)
    manifest = deterministic_manifest(accepted, selected) if accepted else {"manifest_version": "phase3o_s2_training_manifest_v1", "sample_count": 0, "records": [], "manifest_sha256": None}
    write_json(args.output / "phase3o_s2_training_manifest.json", manifest)
    smoke = "NOT_RUN"
    measurements = {"cpu_preprocessing_seconds": None, "projector_forward_seconds": None, "peak_vram_bytes": None, "peak_ram_bytes": None, "output_shape": None, "output_finite": None}
    if accepted:
        process = psutil.Process(); started = time.perf_counter(); cubes = []
        for asset in accepted:
            base = Path(asset.local_path); paths = {band: next(base.glob(f"*_{band}.tif")) for band in S2_BANDS}
            with rasterio.open(paths["B02"]) as grid:
                cube = np.stack([_read_to_grid(paths[band], grid, Resampling.bilinear) for band in S2_BANDS])
            cubes.append(_robust_channel_scale(cube).astype(np.float32))
        measurements["cpu_preprocessing_seconds"] = time.perf_counter() - started
        device = "cuda" if torch.cuda.is_available() else "cpu"; projector = S2MultispectralProjector().to(device).eval()
        if device == "cuda": torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode(): output = projector(torch.from_numpy(np.stack(cubes)).to(device))
        measurements.update({"projector_forward_seconds": time.perf_counter() - started, "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
                             "peak_ram_bytes": process.memory_info().rss, "output_shape": list(output.shape), "output_finite": bool(torch.isfinite(output).all().item()), "device": device})
        smoke = "PASS" if measurements["output_shape"] == [len(accepted), 16, 2048] and measurements["output_finite"] else "FAIL"
    gate = training_gate(assets=accepted, projector_smoke_test=smoke)
    receipt = {"phase": "3O.1", "status": "READY_FOR_SUBSEQUENT_CONTROLLED_PHASE" if gate["training_input_ready"] else "TRAINING_BLOCKED",
               **gate, "dataset_revision": accepted[0].source_revision if accepted else None, "provenance_complete": bool(accepted),
               "split_verified": bool(accepted), "linkage_verified": bool(accepted), "test_contamination": False,
               "resource_measurements": measurements, "execution_permitted": False,
               "training_performed": False, "benchmark_inference_performed": False}
    write_json(args.output / "phase3o1_s2_asset_gate_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))

if __name__ == "__main__": main()
