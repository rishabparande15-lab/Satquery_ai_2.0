from __future__ import annotations

import argparse
from pathlib import Path

from .config import get_settings
from .dataset_loader import discover_samples, load_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the supplied BigEarthNet samples.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()
    settings = get_settings()
    root = args.dataset_root or settings.dataset_root
    for sample in discover_samples(root):
        item = load_sample(sample)
        print(f"Sample: {sample.patch_id}")
        print(f"Optical bands: {item.metadata['optical_band_order']}")
        print(f"Optical tensor shape: {item.raw_optical.shape}")
        print(f"SAR channels: {item.metadata['sar_band_order']}")
        print(f"SAR tensor shape: {item.raw_sar.shape}")
        print(f"Metadata: {item.metadata}")
        print(f"Reference labels: {sample.reference_map}")
        print(f"CRS: {item.metadata['crs']}")
        print(f"Resolution: {item.metadata['resolution']}")
        print(f"Spatial extent: {item.metadata['bounds']}")
        print()


if __name__ == "__main__":
    main()
