"""Build Phase 6B.4 deterministic image-language dataset pools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from src.image_language_dataset import write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        default=Path("artifacts/annotations/bigearthnet_txt/annotations.jsonl"))
    parser.add_argument("--source-manifest", type=Path,
                        default=Path("artifacts/annotations/bigearthnet_txt/manifest.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/annotations/image_language"))
    args = parser.parse_args()
    if not args.annotations.is_file() or not args.source_manifest.is_file():
        parser.error("Verified annotation JSONL and its manifest are required")
    try:
        code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        code_revision = None
    print(json.dumps(write_dataset(args.annotations, args.source_manifest, args.output,
                                   code_revision=code_revision), sort_keys=True))


if __name__ == "__main__":
    main()
