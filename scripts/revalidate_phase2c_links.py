"""Rebuild Phase 2C links from the authoritative Phase 2D.2 catalog."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.representation_catalog import load_manifest
from src.representation_linking import write_link_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        default=Path("artifacts/annotations/bigearthnet_txt/annotations.jsonl"))
    parser.add_argument("--dataset-manifest", type=Path,
                        default=Path("experiments/pipeline3_5000/dataset_manifest.json"))
    parser.add_argument("--catalog-records", type=Path,
                        default=Path("artifacts/representation_catalog/pipeline3_5000/records.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/representation_links/bigearthnet_txt_phase2d2"))
    args = parser.parse_args()

    annotations = [json.loads(line) for line in args.annotations.read_text(encoding="utf-8").splitlines() if line]
    images = load_manifest(args.dataset_manifest)
    representations: dict[str, list[dict]] = {}
    for line in args.catalog_records.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("availability") != "VERIFIED":
            continue
        representations.setdefault(row["image_id"], []).append({
            "reference_id": row["representation_id"], "representation_type": row["representation_type"],
            "scene_id": row["image_id"], "split": row["split"], "producer": row["producer"],
            "producer_version": row["model_version"], "preprocessing_version": row["preprocessing_version"],
            "checksum_sha256": row["checksum"], "artifact_uri": row["logical_uri"],
            "spatial_reference": row["spatial_semantics"],
        })
    report = write_link_artifact(
        annotations, images, representations, args.output,
        requested_representation_types=("physical_62d", "joint_croma_gap_768d", "hybrid_830d"),
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
