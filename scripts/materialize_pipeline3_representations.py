"""Verify the authoritative Pipeline 3 cache and materialize its core representation set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.representation_materialization import materialize_core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path,
                        default=Path("experiments/pipeline3_5000/dataset_manifest.json"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("artifacts/pipeline3_5000/representations"))
    args = parser.parse_args()
    result = materialize_core(args.source_cache_manifest, args.dataset_manifest, args.output_root)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
