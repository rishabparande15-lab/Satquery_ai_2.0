"""Build the deterministic Phase 2D representation catalog."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.representation_catalog import DEFAULT_TYPES, build_catalog, write_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/pipeline3_5000/dataset_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/representation_catalog/pipeline3_5000"))
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    parser.add_argument("--shard-manifest", type=Path, action="append", default=[])
    parser.add_argument("--materialization-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    catalog = build_catalog(args.manifest, receipt_paths=args.receipt,
                            shard_manifest_paths=args.shard_manifest,
                            materialization_manifest_paths=args.materialization_manifest,
                            representation_types=DEFAULT_TYPES)
    report = write_catalog(catalog, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
