"""Validated, modality-aware analysis with a bounded shared CROMA runtime."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import time
from uuid import uuid4, UUID

import numpy as np
import torch
from .config import PROJECT_ROOT, get_settings
from .croma_adapter import CROMAAdapter
from .data_orchestrator import DataOrchestrator
from .hybrid_fusion import HybridFusion, pooled_croma_features
from .input_validation import validate_inputs, check_grid, TEMPORAL_MESSAGE
from .modality_features import extract
from .multimodal_cube import MultimodalCube
from .query_interpreter import interpret_query
from .raster_inputs import assemble
from .tool_selector import select_tools

LOGGER = logging.getLogger(__name__)
DEMO_NOTICE = "Development demonstration using local sample data. Supervised accuracy evaluation is pending a larger labeled dataset."
MODES = {"optical_analysis", "sar_analysis", "joint_optical_sar_analysis", "water_detection",
         "water_change_analysis", "change_detection", "temporal_comparison", "land_cover_classification",
         "scene_description", "single_image_vqa", "segmentation", "object_detection"}
_runtime_lock = threading.Lock()
_adapter = None
_adapter_key = None

class RequestError(ValueError):
    pass

def validate_request(request):
    if not isinstance(request, dict):
        raise RequestError("Request must be a JSON object.")
    query = request.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 4000:
        raise RequestError("Query must contain 1 to 4000 characters.")
    for field in ("start_date", "end_date", "sample_id", "analysis_type"):
        if request.get(field) is not None and not isinstance(request[field], str):
            raise RequestError(f"{field} must be a string.")
    if request.get("analysis_type") and request["analysis_type"] not in MODES:
        raise RequestError("Unknown analysis mode.")
    if not isinstance(request.get("files", {}), dict) or any(k not in {"optical", "sar", "before", "after"} or not isinstance(v, str) for k, v in request.get("files", {}).items()):
        raise RequestError("Files must map optical, sar, before or after to uploaded file IDs.")
    if "band_order_confirmed" in request and type(request["band_order_confirmed"]) is not bool:
        raise RequestError("Band-order confirmation must be boolean.")
    if request.get("cloud_cover") is not None:
        raise RequestError("Cloud filtering requires a provider with quality masks; it is unavailable for local inputs.")
    return request

def croma_available():
    settings = get_settings()
    return (settings.croma_source / "use_croma.py").is_file() and settings.croma_checkpoint.is_file()

def _deep_features(arrays, physical):
    global _adapter, _adapter_key
    settings = get_settings()
    with _runtime_lock:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        key = (str(settings.croma_source), str(settings.croma_checkpoint), settings.croma_checkpoint.stat().st_mtime_ns, device)
        cached = _adapter is not None and key == _adapter_key
        if not cached:
            _adapter = None
            _adapter_key = None
            _adapter = CROMAAdapter(settings.croma_source, settings.croma_checkpoint, device=device)
            _adapter_key = key
        try:
            outputs = _adapter.infer_modality(arrays.get("optical"), arrays.get("sar"))
            if not all(torch.isfinite(value).all() for value in outputs.values()):
                raise ValueError("Non-finite CROMA representation.")
            pooled = pooled_croma_features(outputs) if len(arrays) == 2 else next(value for name, value in outputs.items() if name.endswith("_GAP")).detach().cpu().numpy().reshape(-1)
            deep = {"status": "computed", "source": "official pretrained CROMA base", "device": str(_adapter.device),
                    "pooled_dimension": int(pooled.size), "representations": {k: list(v.shape) for k, v in outputs.items()},
                    "model_reused": cached, "validity": "finite", "values": pooled.tolist()}
            hybrid = {"status": "not applicable", "reason": "The established hybrid schema requires both optical and SAR."}
            if len(arrays) == 2:
                # Preserve the validated fusion seed without changing application-global RNG state.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(42)
                    fusion = HybridFusion(physical.size, pooled.size).to(_adapter.device).eval()
                with torch.no_grad():
                    vector = fusion(torch.from_numpy(physical).to(_adapter.device)[None], torch.from_numpy(pooled).to(_adapter.device)[None]).cpu().numpy()[0]
                if not np.isfinite(vector).all():
                    raise ValueError("Non-finite fusion representation.")
                hybrid = {"status": "computed", "dimension": int(vector.size), "values": vector.tolist(),
                          "source": "deterministic seeded untrained fusion", "validity": "representation only; no task prediction"}
            return deep, hybrid
        except Exception:
            _adapter = None
            _adapter_key = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

def run_analysis(request):
    validate_request(request)
    started = time.perf_counter()
    query = request["query"].strip()
    trace = []
    def step(name, status="completed"):
        trace.append({"step": len(trace) + 1, "timestamp": datetime.now(timezone.utc).isoformat(), "message": name, "status": status})
    step("User query received")
    plan = interpret_query(query, aoi=request.get("aoi"), start_date=request.get("start_date"), end_date=request.get("end_date"), analysis_type=request.get("analysis_type")).to_dict()
    step("Query interpreted by deterministic local parser")
    step("AOI and date range extracted")
    retrieval = DataOrchestrator().retrieve(plan, request).to_dict()
    step("Required datasets selected")
    step("Data availability checked", retrieval["status"])
    validation = validate_inputs(plan, retrieval, request)
    # Public reports contain source identifiers, never server filesystem paths.
    public_retrieval = {**retrieval, "assets": {k: (v if k == "sample_id" else f"upload:{k}") for k, v in retrieval["assets"].items() if k != "dataset_root"}}
    selection = select_tools(plan, {"croma": croma_available()}).to_dict()
    result = {"schema_version": "2.0", "analysis_id": str(uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(),
              "query": query, "status": "rejected", "development_notice": DEMO_NOTICE if retrieval["source"] == "local development provider" else "User-provided imagery. No supervised accuracy claim.",
              "query_interpretation": plan, "data_retrieval": public_retrieval, "validation": validation,
              "tool_selection": selection, "execution_trace": trace, "preprocessing": [], "data_cube": None,
              "features": {"status": "unavailable", "spectral": {"status": "unavailable"}, "deep": {"status": "unavailable"}, "hybrid": {"status": "unavailable"}, "temporal": {"status": "unavailable"}},
              "model_results": {"status": "pending training", "prediction": None, "training_performed": False},
              "evidence": [], "warnings": list(retrieval["warnings"]),
              "confidence": {"prediction_status": "no trained task head available", "calibration_status": "uncalibrated",
                             "accuracy_status": "no accuracy claim", "model_confidence": None, "query_level_confidence": None},
              "temporal": {"status": "requires valid input" if plan["requires_temporal_pair"] else "not requested", "message": TEMPORAL_MESSAGE if plan["requires_temporal_pair"] else None},
              "device": "not used", "error": None}
    arrays = {}
    if validation["valid"] and not plan["requires_temporal_pair"]:
        try:
            arrays, metadata, references = assemble(plan, retrieval, request)
            validation["checks"]["files"] = metadata
            validation["checks"]["optical_sar_compatibility"] = check_grid(metadata["optical"], metadata["sar"]) if len(arrays) == 2 else "not applicable"
            if len(arrays) == 2:
                dates = [m.get("acquisition_date") for m in metadata.values()]
                validation["checks"]["acquisition_matching"] = {"dates": dates, "status": "same-date" if len(set(dates)) == 1 and all(dates) else "different or unknown dates"}
                if len(set(dates)) != 1 or not all(dates):
                    validation["warnings"].append("Optical/SAR acquisition dates differ or are unknown; joint representation is not simultaneous ground truth.")
            result["evidence"] = references
            result["preprocessing"] = [
                {"operation": "canonical band ordering and float32 tensor conversion", "status": "completed"},
                {"operation": "alignment", "status": "completed", "detail": "Local bands reprojected to reference grid" if retrieval["source"] == "local development provider" else "Uploaded grids verified equal; no resampling"},
                {"operation": "cloud masking", "status": "skipped", "reason": "No validated quality mask"},
                {"operation": "SAR denoising", "status": "skipped", "reason": "No validated denoising configuration"},
                {"operation": "AOI cropping", "status": "skipped", "reason": "AOI intersection is checked; statistics cover the entire input patch"}]
            cube = MultimodalCube(optical=arrays.get("optical"), sar=arrays.get("sar"), metadata=metadata, aoi=plan["aoi"])
            result["data_cube"] = cube.summary()
        except (ValueError, OSError, StopIteration):
            LOGGER.warning("Raster validation or assembly failed")
            validation["valid"] = False
            # ValueErrors are authored input errors; raster IO details stay in logs.
            import sys
            exc = sys.exception()
            validation["errors"].append(str(exc) if type(exc) is ValueError else "Selected raster could not be assembled. Check bands and source availability.")
            arrays = {}
    step("Inputs validated", "completed" if validation["valid"] else "rejected")
    step("Available model/tool selected")
    runnable = bool(arrays) and validation["valid"]
    step("Preprocessing completed" if runnable else "Preprocessing not run", "completed" if runnable else "skipped")
    step("Multimodal cube created" if runnable else "Multimodal cube not created", "completed" if runnable else "skipped")
    if runnable:
        try:
            physical, report = extract(arrays)
            report["unavailable_indices"] = [key for key, count in report["valid_pixel_counts"].items() if key.startswith("index_") and count == 0]
            if report["unavailable_indices"]:
                report["limitations"].append("Indices with zero valid denominators have legacy zero placeholders; they are unavailable observations.")
            result["features"]["spectral"] = {"status": "computed", "source": "local GEE-style physical features", **report, "statistics": report["feature_values"]}
            result["features"]["status"] = "partial"
            result["warnings"].extend(report["limitations"])
            result["status"] = "partial"
            result["device"] = "cpu"
            if croma_available() and all(array.shape[1:] == (120, 120) for array in arrays.values()):
                deep_started = time.perf_counter()
                try:
                    deep, hybrid = _deep_features(arrays, physical)
                    result["features"].update(status="computed", deep=deep, hybrid=hybrid)
                    result["device"] = deep["device"]
                    result["status"] = "completed"
                    result["preprocessing"].append({"operation": "CROMA normalization", "status": "completed", "detail": "Official per-channel mean +/- 2 std clipping to [0,1]"})
                except Exception:
                    LOGGER.exception("CROMA inference failed")
                    result["features"]["deep"] = {"status": "unavailable", "reason": "CROMA execution failed; physical features were retained. Retry or inspect server logs."}
                    result["warnings"].append(result["features"]["deep"]["reason"])
                result["features"]["deep"]["runtime_seconds"] = time.perf_counter() - deep_started
            else:
                reason = "Official CROMA source/checkpoint is unavailable." if not croma_available() else "CROMA requires 120x120 pixels. Physical features cover the native raster; no implicit resize."
                result["features"]["deep"] = {"status": "unavailable", "reason": reason}
                result["warnings"].append(reason)
        except ValueError:
            result["status"] = "rejected"
            validation["valid"] = False
            validation["errors"].append("Physical feature calculation failed; check numerical range and valid pixels.")
    step("Features extracted", result["features"]["status"])
    step("Representation analysis executed", result["status"])
    if plan["requires_temporal_pair"] and validation["valid"]:
        result["status"] = "unavailable"
        result["temporal"] = {"status": "unavailable", "message": "Pair metadata passed validation. The web temporal algorithm is not implemented; no change or area is reported."}
    result["evidence"] += [{"kind": "feature statistics", "reference": "features.spectral.statistics", "status": result["features"]["spectral"]["status"]},
                           {"kind": "prediction mask/change map", "status": "unavailable"}]
    step("Evidence generated")
    step("Confidence assessed")
    result["warnings"].extend(validation["warnings"])
    if result["features"]["spectral"]["status"] == "computed":
        physical_dim = result["features"]["spectral"]["dimension"]
        deep = result["features"]["deep"]
        representation = f" Extracted {deep['pooled_dimension']} CROMA representation values." if deep["status"] == "computed" else " " + deep.get("reason", "Deep representations unavailable.")
        fusion = result["features"]["hybrid"]
        representation += f" Computed a {fusion['dimension']}-dimensional untrained hybrid representation." if fusion["status"] == "computed" else ""
        answer = f"Processed {' and '.join(plan['modalities'])} inputs and computed {physical_dim} physical features over the full input patch." + representation
    else:
        answer = "; ".join(validation["errors"]) or result["temporal"]["message"] or "Analysis unavailable."
    result["llm_explanation"] = answer + " No task prediction, supervised accuracy, calibrated confidence, or change map was generated."
    step("Grounded deterministic explanation generated")
    step("Final report payload created")
    result["runtime_seconds"] = time.perf_counter() - started
    result["confidence"]["validation_status"] = "passed" if validation["valid"] else "rejected"
    json.dumps(result, allow_nan=False)
    return result

def save_report(result, output_root=None):
    output_root = Path(output_root or PROJECT_ROOT / "experiments" / "outputs" / "web_reports")
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_id = str(UUID(result["analysis_id"]))
    path = output_root / f"{analysis_id}.json"
    payload = json.dumps(result, indent=2, allow_nan=False)
    # A report ID is immutable; reject accidental overwrites.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return path
