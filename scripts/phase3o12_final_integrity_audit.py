"""Read-only final integrity audit for the completed Phase 3O.12 run."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eo_vlm.multispectral_projector import S2MultispectralProjector

RUN = ROOT / "artifacts/training/phase3o/phase3o12_run"
OUT = RUN / "final_integrity_audit.json"
EXPECTED = {
    "initial_sha": "50e8857ba7b9e5210de7757fd596e28112e0e82bfadb0957ff6e127fe8fabc2e",
    "initial_fingerprint": "948f3e8785141e51edb2e79431958975ae8ea190e88701865fdfc2cd22eb976a",
    "final_sha": "5e3bc79d5cf1ee194db433503360cdf7530bb5fc908a3d936de7c43b449933c0",
    "final_fingerprint": "168c6e64264369d0385697c5c008fd1424ee84fe1ffd12ca9079943e45cd036b",
    "qwen_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
}
REQUIRED = (
    "preregistered_config.json", "phase3o12_training_manifest.json",
    "phase3o12_projector_initial.pt", "untrained_semantic_audit.json",
    "image_shuffle_map.json", "phase3o12_supervision_diagnostic.json",
    "phase3o12_projector_final.pt", "training_summary.json", "reload_receipt.json",
    "trained_semantic_audit.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(name: str) -> dict:
    with (RUN / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def fingerprint(checkpoint: Path) -> str:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    projector = S2MultispectralProjector()
    projector.load_state_dict(state)
    return projector.fingerprint()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"PHASE3O12_INTEGRITY_FAILURE: {message}")


def main() -> None:
    files = {name: RUN / name for name in REQUIRED}
    require(all(path.is_file() and path.stat().st_size > 0 for path in files.values()), "missing or empty required artifact")
    config, manifest = load_json("preregistered_config.json"), load_json("phase3o12_training_manifest.json")
    initial_receipt, training = load_json("initial_projector_receipt.json"), load_json("training_summary.json")
    reload_receipt, diagnostic = load_json("reload_receipt.json"), load_json("phase3o12_supervision_diagnostic.json")
    untrained, trained, shuffle = (load_json(name) for name in ("untrained_semantic_audit.json", "trained_semantic_audit.json", "image_shuffle_map.json"))

    initial_sha, final_sha = sha256(files["phase3o12_projector_initial.pt"]), sha256(files["phase3o12_projector_final.pt"])
    initial_fp, final_fp = fingerprint(files["phase3o12_projector_initial.pt"]), fingerprint(files["phase3o12_projector_final.pt"])
    require((initial_sha, initial_fp) == (EXPECTED["initial_sha"], EXPECTED["initial_fingerprint"]), "initial projector identity")
    require((final_sha, final_fp) == (EXPECTED["final_sha"], EXPECTED["final_fingerprint"]), "final projector identity")
    require(initial_sha != final_sha and initial_fp != final_fp, "initial and final projectors must differ")
    require(initial_receipt["sha256"] == initial_sha and initial_receipt["fingerprint"] == initial_fp, "initial receipt mismatch")
    require(training["initial_sha"] == initial_sha and training["initial_fp"] == initial_fp, "training initial receipt mismatch")
    require(training["final_sha"] == final_sha and training["final_fp"] == final_fp, "training final receipt mismatch")
    require(reload_receipt["final_sha"] == final_sha and reload_receipt["final_fingerprint"] == final_fp, "reload receipt mismatch")
    require(reload_receipt["agreement"] == "EXACT_MATCH" and reload_receipt["absolute_difference"] == 0.0, "reload disagreement")

    expected_counts = {"train": {"binary_qa": 200, "multiple_choice_qa": 200, "caption": 200}, "validation": {"binary_qa": 50, "multiple_choice_qa": 50, "caption": 50}}
    require(config["qwen_revision"] == EXPECTED["qwen_revision"] and config["test_count"] == 0, "preregistered configuration")
    require(manifest["counts"] == expected_counts and manifest["train_validation_image_overlap"] == 0, "manifest counts or overlap")
    require(manifest["test_access_count"] == 0 and all(row["split"] != "test" for rows in manifest["selection"].values() for row in rows), "manifest TEST access")
    source_dir = ROOT / "artifacts/annotations/image_language"
    require(all(sha256(source_dir / name) == digest for name, digest in config["dataset_manifest_sources"].items()), "preregistered source hash")
    for split, count in (("train", 600), ("validation", 150)):
        require(len(manifest["selection"][split]) == count, f"{split} selection size")
    require(len(shuffle["mapping"]) == 150 and shuffle["seed"] == config["random_seed"], "shuffle mapping content")
    require({row["sample_id"] for row in shuffle["mapping"]} == {row["record_id"] for row in manifest["selection"]["validation"]}, "shuffle mapping validation membership")

    test_receipts = (manifest, training, reload_receipt, diagnostic, untrained, trained, shuffle)
    require(all(record.get("test_access_count", 0) == 0 for record in test_receipts), "recorded TEST access")
    require(training["optimizer_membership"] == {"optimizer_projector_tensors": 8, "optimizer_qwen_tensors": 0, "projector_tensors": 8}, "optimizer membership")
    require(training["qwen_unchanged"] and reload_receipt["qwen_unchanged"] and diagnostic["qwen_unchanged"], "Qwen immutability")
    require(diagnostic["no_optimizer_created"] and diagnostic["no_optimizer_step"], "supervision diagnostic mutation")
    require(diagnostic["initial_sha_after"] == initial_sha and diagnostic["initial_fingerprint_after"] == initial_fp, "initial projector mutation")
    require(all(diagnostic["diagnostics"][task]["projector_nonzero_gradient_tensors"] == 8 for task in ("binary_qa", "multiple_choice_qa", "caption")), "projector gradients")
    require(all(diagnostic["diagnostics"][task]["qwen_gradient_tensors"] == 0 for task in ("binary_qa", "multiple_choice_qa", "caption")), "Qwen gradients")

    summary = trained["summary"]
    require(summary["binary_normal"]["correct"] == 27 and summary["binary_shuffled"]["correct"] == 26, "binary summary")
    require(summary["mcq_normal"]["correct"] == 13 and summary["mcq_shuffled"]["correct"] == 13, "MCQ summary")
    require(summary["caption_normal"]["nonempty"] == 50 and summary["caption_normal"]["unique_outputs"] == 2 and summary["caption_changed"] == 26, "caption summary")
    require(len(trained["normal_results"]) == 150 and len(trained["shuffled_results"]) == 150, "trained audit completeness")
    require(len(untrained["normal_results"]) == 150 and len(untrained["shuffled_results"]) == 150, "untrained audit completeness")

    artifact = {
        "phase": "3O.12", "status": "PHASE3O12_COMPLETE", "audit_type": "read_only_final_integrity",
        "artifacts_complete": True, "artifact_sha256": {name: sha256(path) for name, path in files.items()},
        "initial": {"sha256": initial_sha, "fingerprint": initial_fp, "historical_receipts_match": True},
        "final": {"sha256": final_sha, "fingerprint": final_fp, "historical_receipts_match": True, "unchanged_after_audit": True},
        "dataset": {"counts": expected_counts, "test_count": 0, "train_validation_image_overlap": 0, "preregistered_source_hashes_match": True, "shuffle_mapping_validation_membership_match": True},
        "qwen": {"revision": EXPECTED["qwen_revision"], "trainable_parameter_count": 0, "optimizer_parameter_count": 0, "gradient_tensor_count": 0, "unchanged": True},
        "test_access_count": 0, "reload_agreement": "EXACT_MATCH", "semantic_audit_records_verified": 600,
    }
    OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "initial_sha": initial_sha, "final_sha": final_sha, "test_access_count": 0}, indent=2))


if __name__ == "__main__":
    main()
