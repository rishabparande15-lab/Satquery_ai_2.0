"""Build the canonical Phase 2E BigEarthNet.txt image-language view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.image_language_foundation import build_bigearthnet_text_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        default=Path("artifacts/annotations/bigearthnet_txt/annotations.jsonl"))
    parser.add_argument("--annotation-manifest", type=Path,
                        default=Path("artifacts/annotations/bigearthnet_txt/manifest.json"))
    parser.add_argument("--dataset-manifest", type=Path,
                        default=Path("experiments/pipeline3_5000/dataset_manifest.json"))
    parser.add_argument("--representation-catalog", type=Path,
                        default=Path("artifacts/representation_catalog/pipeline3_5000/records.jsonl"))
    parser.add_argument("--representation-catalog-report", type=Path,
                        default=Path("artifacts/representation_catalog/pipeline3_5000/validation_report.json"))
    parser.add_argument("--representation-link-manifest", type=Path,
                        default=Path("artifacts/representation_links/bigearthnet_txt_phase2d2/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/image_language/bigearthnet_txt"))
    args = parser.parse_args()
    result = build_bigearthnet_text_view(
        args.annotations, args.annotation_manifest, args.dataset_manifest,
        args.representation_catalog, args.representation_catalog_report,
        args.representation_link_manifest, args.output,
    )
    print(json.dumps({"manifest": result["manifest"], "report": result["report"]}, sort_keys=True))


if __name__ == "__main__":
    main()
