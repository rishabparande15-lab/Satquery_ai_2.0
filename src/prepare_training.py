from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import get_settings
from .dataset_loader import discover_samples
from .feature_cache import extract_feature_cache
from .training_data import BIGEARTHNET_CLASSES, make_splits, save_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reusable splits and compact hybrid features.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    samples = discover_samples(args.dataset_root)
    manifest = make_splits([sample.patch_id for sample in samples], seed=args.seed)
    save_manifest(manifest, args.split_manifest)
    summary = extract_feature_cache(args.dataset_root, args.cache_root, settings.croma_source, settings.croma_checkpoint, args.device, resume=not args.no_resume)
    result = {"split_manifest": str(args.split_manifest), "split_counts": {key: len(value) for key, value in manifest.splits.items()}, "class_count": len(BIGEARTHNET_CLASSES), "feature_cache": summary}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
