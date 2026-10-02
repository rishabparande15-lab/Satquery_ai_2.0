"""Fast, non-loading runtime readiness checks for the frozen SatQuery release."""
from __future__ import annotations
from pathlib import Path
import importlib.util, platform, sys
from typing import Any
from .config import PROJECT_ROOT, get_settings

def _asset(path: Path | None, *, required: bool, kind: str) -> dict[str, Any]:
    exists = bool(path and path.exists())
    return {"kind": kind, "configured": bool(path), "exists": exists, "required": required,
            "status": "AVAILABLE" if exists else ("BLOCKED" if required else "OPTIONAL_UNAVAILABLE")}

def run_preflight() -> dict[str, Any]:
    settings = get_settings(); warnings=[]; errors=[]
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        gpu = {"cuda_available": cuda, "torch": torch.__version__, "device": settings.device_preference,
               "name": torch.cuda.get_device_name(0) if cuda else None,
               "vram_total_bytes": int(torch.cuda.get_device_properties(0).total_memory) if cuda else None}
    except Exception:
        gpu = {"cuda_available": False, "torch": None, "device": settings.device_preference, "name": None, "vram_total_bytes": None}
        errors.append("DEPENDENCY_MISSING: torch")
    dataset = _asset(settings.dataset_root, required=True, kind="BigEarthNet development root")
    metadata = _asset(settings.dataset_root / "metadata.parquet", required=True, kind="metadata.parquet")
    levir = _asset(settings.levir_root / "images" / "val" / "A" / "val_000001.png", required=True, kind="LEVIR validation PRE")
    models = {"croma_source": _asset(settings.croma_source / "use_croma.py", required=True, kind="CROMA source"),
              "croma_checkpoint": _asset(settings.croma_checkpoint, required=True, kind="CROMA checkpoint"),
              "s2_projector": _asset(settings.s2_projector_checkpoint, required=True, kind="S2 projector"),
              "joint_projector": _asset(settings.joint_projector_checkpoint, required=True, kind="joint projector"),
              "chg2cap": _asset(settings.chg2cap_checkpoint, required=True, kind="Chg2Cap checkpoint"),
              "scene_model": _asset(settings.scene_description_model, required=False, kind="scene-description model")}
    filesystem = {"project_root": PROJECT_ROOT.is_dir(), "uploads_writable": False, "frontend_assets": all((PROJECT_ROOT / "src" / "static" / item).is_file() for item in ("index.html","app.js","style.css"))}
    try:
        upload = PROJECT_ROOT / "experiments" / "outputs" / "web_uploads"; upload.mkdir(parents=True, exist_ok=True); probe=upload / ".preflight"; probe.write_text("ok"); probe.unlink(); filesystem["uploads_writable"] = True
    except OSError: errors.append("FILESYSTEM_NOT_WRITABLE: upload staging")
    for item in (dataset, metadata, levir, *models.values()):
        if item["status"] == "BLOCKED": errors.append(f"MISSING_REQUIRED_ASSET: {item['kind']}")
        elif item["status"] == "OPTIONAL_UNAVAILABLE": warnings.append(f"OPTIONAL_UNAVAILABLE: {item['kind']}")
    if not gpu["cuda_available"]: warnings.append("CUDA_UNAVAILABLE: GPU demonstration routes may be slow or unavailable")
    return {"status": "BLOCKED" if errors else ("READY_WITH_WARNINGS" if warnings else "READY"),
            "python": {"version": platform.python_version(), "executable": Path(sys.executable).name, "rasterio_available": importlib.util.find_spec("rasterio") is not None},
            "gpu": gpu, "datasets": {"bigearthnet": dataset, "metadata": metadata, "levir": levir}, "models": models,
            "filesystem": filesystem, "warnings": warnings, "errors": errors}
