"""Generate Phase 6B.1 artifacts from a separately obtained source table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from src.annotation_foundation import (OFFICIAL_REVISION, OFFICIAL_SHA256, OFFICIAL_URL,
                                       write_artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/raw/bigearthnet_txt/source/BigEarthNet.txt.parquet"),
                        help="Pinned BigEarthNet.txt Parquet or JSONL")
    parser.add_argument("--manifest", type=Path, default=Path("experiments/pipeline3_5000/dataset_manifest.json"))
    parser.add_argument("--split", type=Path, default=Path("experiments/pipeline3_5000/split_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/annotations/bigearthnet_txt"))
    parser.add_argument("--source-revision", default=OFFICIAL_REVISION)
    parser.add_argument("--expected-source-sha256", default=OFFICIAL_SHA256)
    parser.add_argument("--source-url", default=OFFICIAL_URL)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"Missing source file: {args.source}. Obtain {OFFICIAL_URL} and verify SHA-256 {OFFICIAL_SHA256}.")
    try:
        code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        code_revision = None
    result = write_artifact(args.source, args.manifest, args.split, args.output,
                            source_revision=args.source_revision, code_revision=code_revision,
                            expected_source_sha256=args.expected_source_sha256,
                            source_url=args.source_url)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
