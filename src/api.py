from .dataset_loader import (OPTICAL_BANDS, SAR_BANDS, discover_samples, discover_s2_samples,
                             load_optical_sample, load_sample)
"""Loopback HTTP application with bounded uploads and serialized inference."""
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
import rasterio
import numpy as np
from urllib.parse import urlparse
from uuid import uuid4

from .analysis_engine import RequestError, save_report, validate_request
from .architecture_contracts import TaskRequest
from .config import PROJECT_ROOT, get_settings
from .dataset_loader import discover_samples, discover_s2_samples, load_optical_sample, load_sample
from .scene_description import (SceneDescriptionController, SceneDescriptionInputError,
                                load_rgb_image, render_s2_rgb)
from .deterministic_scene_analysis import TASK_TYPE, run_deterministic_scene_analysis_with_context
from .satquery_v1 import SatQueryV1Controller
from .single_image_vqa import SingleImageVQAController, classify_single_image_route
from .single_image_sar_vqa import SingleImageSARVQAController
from .temporal_change import (TemporalChangeDescriptionController, TemporalChangeInputError,
                              validate_temporal_change_input, validate_temporal_image_path)
from .input_validation import inspect_raster
from .generic_raster import GenericRasterInspectionError, inspect_generic_raster
from .imagery_availability import provider_status, registry_response, search_imagery
from .pipeline3_scene_probe import IdentityResolutionError, ModelContractError, list_pipeline3_area_ids
from .query_interpreter import interpret_query
from .agent.api_integration import analyze_controller_request
from .satquery_agent import SatQueryAgent, capability_registry
from .sensor_adapters import SensorDeclarationError, inspect_and_gate, validate_external_pair
from .geospatial_evidence import raster_evidence, pair_evidence, response_evidence
from .runtime_preflight import run_preflight
from .specialist_runtime import RUNTIME
from .public_serialization import sanitize_public_value
from .upload_staging import recover_staging

# Compatibility seam for existing API fault-injection tests.  This name now
# points only to the registry dispatcher; it is not the legacy Pipeline 3
# function and therefore cannot bypass CapabilityRegistry.
run_analysis = run_deterministic_scene_analysis_with_context

LOGGER = logging.getLogger(__name__)
STATIC_ROOT = PROJECT_ROOT / "src" / "static"
REPORT_ROOT = PROJECT_ROOT / "experiments" / "outputs" / "web_reports"
UPLOAD_ROOT = PROJECT_ROOT / "experiments" / "outputs" / "web_uploads"
MAX_UPLOAD = 32 * 1024 * 1024
MAX_JSON = 64 * 1024
UPLOAD_TTL = 15 * 60
ANALYSIS_LOCK = threading.Lock()
UPLOAD_LOCK = threading.Lock()
UPLOADS = {}
SINGLE_IMAGE_LOCK = threading.Lock()
SINGLE_IMAGE_CONTROLLER = None
TEMPORAL_LOCK = threading.Lock()
TEMPORAL_CONTROLLER = None
SCENE_DESCRIPTION_LOCK = threading.Lock()
SCENE_DESCRIPTION_CONTROLLER = None
OPTICAL_SAR_CONTROLLER = None


def _claim_upload(token, role):
    """Resolve a short-lived upload token without exposing it to the agent."""
    if not isinstance(token, str):
        raise ValueError(f"A valid {role} upload token is required.")
    with UPLOAD_LOCK:
        item = UPLOADS.get(token)
        if not item or item["role"] != role:
            raise ValueError(f"{role} upload is invalid or expired.")
        item["created"] = float("inf")
        return item["path"]


def run_unified_query(request):
    """Invoke the central agent while delegating every model call to its existing controller."""
    tokens = []
    def single_executor(payload):
        global SINGLE_IMAGE_CONTROLLER
        token = request.get("upload_token")
        uploaded_path = _claim_upload(token, "optical") if token else None
        if token: tokens.append(token)
        source = {**request, "question": payload["query"], "task": "SINGLE_IMAGE_VQA"}
        resolved = build_single_image_request(source, uploaded_path=uploaded_path)
        if resolved["route"] != "SINGLE_IMAGE_VQA":
            raise ValueError("The declared single-image input is not compatible with S2 VQA.")
        if not SINGLE_IMAGE_LOCK.acquire(blocking=False):
            raise RuntimeError("A single-image inference is already running.")
        try:
            with RUNTIME.use("SINGLE_IMAGE_VQA", SingleImageVQAController) as controller:
                response = controller.run(image_id=resolved["image_id"], split=resolved["split"], optical=resolved["optical"],
                                               metadata=resolved["spatial_metadata"], question=resolved["question"],
                                               task_type=resolved["task_type"], choices=resolved["choices"])
            response.setdefault("details", {})["evidence_inputs"] = [{"input_id": resolved["image_id"],
                "inspection": (resolved.get("external_input_inspection") or {}).get("inspection"),
                "declaration": (resolved.get("external_input_inspection") or {}).get("sensor_declaration"),
                "adapter": (resolved.get("external_input_inspection") or {}).get("adapter"),
                "compatibility": (resolved.get("external_input_inspection") or {}).get("compatibility_gate")}]
            return response
        finally:
            SINGLE_IMAGE_LOCK.release()

    def optical_sar_executor(payload):
        global OPTICAL_SAR_CONTROLLER
        source = {**request, "question": payload["query"]}
        resolved = build_v1_route_request(source)
        kwargs = {key: value for key, value in resolved.items() if key in {"task_type", "question", "s1", "s2", "patch_id", "s1_patch_id", "s2_patch_id", "spatial_metadata"}}
        with RUNTIME.use("OPTICAL_SAR_ANALYSIS", SatQueryV1Controller) as controller:
            result = controller.run_satquery(**kwargs)
        if result.get("status") != "OK": raise RuntimeError(result.get("error_code") or "optical-SAR specialist failed")
        spatial = resolved["spatial_metadata"]
        shape = spatial["shape"]
        evidence_inputs = []
        for role, patch_key, bands in (("S1", "s1_patch_id", spatial["sar_band_order"]),
                                       ("S2", "s2_patch_id", spatial["optical_band_order"])):
            inspection = {"file_name": spatial["source_filenames"][role], "crs": spatial["crs"],
                          "bounds": spatial["bounds"], "resolution": spatial["resolution"],
                          "width": shape[1], "height": shape[0], "band_count": len(bands),
                          "band_descriptions": bands}
            evidence_inputs.append({"input_id": resolved[patch_key], "inspection": inspection})
        result["evidence_inputs"] = evidence_inputs
        result["warnings"] = list(dict.fromkeys([*result.get("warnings", []), "COREGISTRATION_NOT_VERIFIED"]))
        return result

    def single_sar_executor(payload):
        token = request.get("sar_upload_token") or request.get("upload_token")
        uploaded_path = _claim_upload(token, "sar") if token else None
        if token: tokens.append(token)
        resolved = build_single_sar_request({**request, "question": payload["query"]}, uploaded_path=uploaded_path)
        with RUNTIME.use("SINGLE_IMAGE_SAR_VQA", SingleImageSARVQAController) as controller:
            response = controller.run(image_id=resolved["image_id"], split=resolved["split"], sar=resolved["sar"], metadata=resolved["spatial_metadata"], question=resolved["question"], task_type=resolved["task_type"], choices=resolved["choices"])
        response["evidence_inputs"] = [{"input_id": resolved["image_id"], "inspection": resolved["external_input_inspection"]["inspection"], "declaration": resolved["external_input_inspection"]["sensor_declaration"], "adapter": resolved["external_input_inspection"]["adapter"], "compatibility": resolved["external_input_inspection"]["compatibility_gate"]}]
        return response

    def temporal_executor(payload):
        global TEMPORAL_CONTROLLER
        t1_token, t2_token = request.get("t1_token"), request.get("t2_token")
        t1_path = _claim_upload(t1_token, "t1"); t2_path = _claim_upload(t2_token, "t2")
        tokens.extend((t1_token, t2_token))
        source = {**request, "query": payload["query"], "temporal_order": "PRE_POST"}
        resolved = build_temporal_change_request(source, t1_path=t1_path, t2_path=t2_path)
        if not TEMPORAL_LOCK.acquire(blocking=False): raise RuntimeError("A temporal change description is already running.")
        try:
            with RUNTIME.use("TEMPORAL_CHANGE_DESCRIPTION", TemporalChangeDescriptionController) as controller:
                response = controller.run(t1_path=resolved["t1_path"], t2_path=resolved["t2_path"],
                                           query=resolved["query"], metadata=resolved["metadata"])
            response.setdefault("details", {})["evidence_inputs"] = [{"input_id": resolved["t1_identity"]["filename"], "inspection": inspect_generic_raster(t1_path)}, {"input_id": resolved["t2_identity"]["filename"], "inspection": inspect_generic_raster(t2_path)}]
            return response
        finally:
            TEMPORAL_LOCK.release()

    def scene_description_executor(payload):
        global SCENE_DESCRIPTION_CONTROLLER
        token = request.get("scene_upload_token")
        uploaded_path = _claim_upload(token, "scene") if token else None
        if token: tokens.append(token)
        resolved = build_scene_description_request({**request, "query": payload["query"]}, uploaded_path=uploaded_path)
        if not SCENE_DESCRIPTION_LOCK.acquire(blocking=False):
            raise RuntimeError("A scene-description inference is already running.")
        try:
            with RUNTIME.use("SINGLE_IMAGE_SCENE_DESCRIPTION", SceneDescriptionController) as controller:
                response = controller.run(**resolved)
            external = resolved.get("source_metadata", {}).get("external_input", {})
            response.setdefault("details", {})["evidence_inputs"] = [{"input_id": resolved["image_identity"], "inspection": external.get("inspection"), "declaration": external.get("sensor_declaration"), "adapter": external.get("adapter"), "compatibility": external.get("compatibility")}]
            return response
        finally:
            SCENE_DESCRIPTION_LOCK.release()

    agent = SatQueryAgent({"SINGLE_IMAGE_VQA": single_executor, "SINGLE_IMAGE_SAR_VQA": single_sar_executor, "OPTICAL_SAR_ANALYSIS": optical_sar_executor,
                           "TEMPORAL_CHANGE_DESCRIPTION": temporal_executor,
                           "SINGLE_IMAGE_SCENE_DESCRIPTION": scene_description_executor})
    try:
        result = agent.run(inputs=request.get("inputs", []), query=request.get("query", ""),
                           context=request.get("context"), requested_task=request.get("requested_task"))
        result["evidence"] = response_evidence(result)
    finally:
        cleanup_uploads(tokens)
    return result


def build_task_request(request):
    """Translate validated HTTP input into the sole capability boundary."""
    plan = interpret_query(
        request["query"], aoi=request.get("aoi"), start_date=request.get("start_date"),
        end_date=request.get("end_date"), analysis_type=request.get("analysis_type"),
    )
    sample_id = request.get("sample_id")
    full_manifest_identity = isinstance(sample_id, str) and sample_id.startswith(("S1A_", "S1B_", "S2A_", "S2B_"))
    modalities = ("optical", "sar") if full_manifest_identity else tuple(plan.modalities)
    return TaskRequest(
        task_type=TASK_TYPE,
        scene_id=sample_id,
        scene_reference={"input_type": "satellite_scene", "request": request},
        query=request["query"], requested_modalities=modalities,
        parameters={"requested_analysis_type": plan.task},
        execution_metadata={"entrypoint": "POST /api/analyze", "dispatch": "CapabilityRegistry"},
    )

def build_v1_route_request(request):
    """Resolve a live BigEarthNet patch into the strict paired S1+S2 route."""
    task_type = str(request.get("task_type") or "caption")
    patch_id = request.get("patch_id") or request.get("sample_id") or request.get("bigearthnet_patch_id")
    if not isinstance(patch_id, str) or not patch_id.strip():
        raise ValueError("A valid patch_id is required for the MULTIMODAL_S1_S2 route.")
    patch_id = patch_id.strip()
    dataset_root = Path(request.get("dataset_root") or get_settings().dataset_root)
    samples = discover_samples(dataset_root, strict=True)
    sample = next((item for item in samples if item.patch_id == patch_id), None)
    if sample is None:
        raise FileNotFoundError(f"Patch not found in the local dataset: {patch_id}")
    prepared = load_sample(sample)
    s1 = prepared.sar.astype("float32")
    s2 = prepared.optical.astype("float32")
    if request.get("s1") is not None:
        s1 = request["s1"]
    if request.get("s2") is not None:
        s2 = request["s2"]
    spatial = {"crs": prepared.metadata.get("crs"), "resolution": prepared.metadata.get("resolution"), "bounds": prepared.metadata.get("bounds"), "shape": prepared.metadata.get("shape"),
               "optical_band_order": prepared.metadata.get("optical_band_order"), "sar_band_order": prepared.metadata.get("sar_band_order"),
               "source_filenames": {"S1": sample.sar_paths["VV"].name, "S2": sample.optical_paths["B02"].name}}
    return {
        "route": "MULTIMODAL_S1_S2",
        "task_type": task_type,
        "question": str(request.get("question") or "Describe the scene."),
        "patch_id": patch_id,
        "s1_patch_id": str(request.get("s1_patch_id") or patch_id),
        "s2_patch_id": str(request.get("s2_patch_id") or patch_id),
        "s1": s1,
        "s2": s2,
        "spatial_metadata": spatial,
    }


def build_scene_description_request(request, *, uploaded_path=None):
    """Resolve either a declared RGB upload or a verified non-test local S2 patch."""
    if not isinstance(request, dict):
        raise ValueError("A scene-description request object is required.")
    query = str(request.get("query") or "Describe this remote-sensing image in one concise sentence.").strip()
    if not query:
        raise ValueError("A non-empty scene-description request is required.")
    if uploaded_path is not None:
        preinspection = inspect_generic_raster(str(uploaded_path))
        declaration = request.get("sensor_declaration") or {"sensor": "generic-rgb", "modality": "rgb", "role": "SINGLE", "band_order": ["R", "G", "B", "A"][:preinspection["band_count"]], "band_order_confirmed": True}
        try:
            external_input = inspect_and_gate(str(uploaded_path), declaration, "SINGLE_IMAGE_SCENE_DESCRIPTION", inspector=inspect_generic_raster)
        except (GenericRasterInspectionError, SensorDeclarationError) as exc:
            raise ValueError(str(exc)) from None
        gate = external_input["compatibility_gate"]
        if not gate["eligible_for_agent"]:
            raise ValueError(f"{gate['code']}: {gate['message']}")
        image = load_rgb_image(uploaded_path)
        return {"image": image, "image_identity": str(request.get("image_id") or f"upload:{Path(uploaded_path).name}"),
                "source_kind": "user_declared_rgb_upload", "source_metadata": {"sensor": declaration.get("sensor"), "modality": "RGB",
                "external_input": {"inspection": external_input["inspection"], "sensor_declaration": external_input["sensor_declaration"],
                                   "adapter": external_input["adapter"], "compatibility": gate}}, "query": query}
    patch_id = request.get("patch_id")
    if not isinstance(patch_id, str) or not patch_id.strip():
        raise ValueError("Select a non-test local S2 patch or supply an explicit RGB scene upload.")
    samples = {sample.patch_id: sample for sample in discover_s2_samples(Path(request.get("dataset_root") or get_settings().dataset_root), allowed_splits=("train", "validation"))}
    sample = samples.get(patch_id.strip())
    if sample is None:
        raise FileNotFoundError("The requested local S2 patch is absent or belongs to a forbidden split.")
    prepared = load_optical_sample(sample)
    return {"image": render_s2_rgb(prepared.raw_optical), "image_identity": sample.patch_id,
            "source_kind": "verified_local_s2_b04_b03_b02_rgb_rendering",
            "source_metadata": {"split": sample.split, **prepared.metadata, "rendering": "B04/B03/B02 contrast-stretched RGB"}, "query": query}


def build_single_image_request(request, *, uploaded_path=None):
    """Resolve one non-test optical patch without opening or requiring S1 files."""
    if not isinstance(request, dict):
        raise ValueError("A single-image request object is required")
    image_count = request.get("image_count", 1)
    if image_count != 1:
        raise ValueError("SINGLE_IMAGE routes require exactly one image")
    query = str(request.get("question") or request.get("query") or "").strip()
    task_type = str(request.get("task_type") or "binary_qa")
    task = request.get("task")
    modality = str(request.get("modality") or "s2")
    route = classify_single_image_route(query, image_count=image_count, modalities=(modality,), task=task)
    if route not in {"SINGLE_IMAGE_VQA", "SINGLE_IMAGE_GROUNDING"}:
        return {"route": route, "question": query, "task": route, "task_type": task_type}
    if uploaded_path is not None:
        path = Path(uploaded_path)
        if not path.is_file():
            raise FileNotFoundError("Uploaded optical GeoTIFF is no longer available")
        if route == "SINGLE_IMAGE_VQA":
            declaration = request.get("sensor_declaration")
            if declaration is None:
                raise ValueError("External 12-band VQA uploads require an explicit Sentinel-2 sensor declaration. Use the generic raster inspector for Cartosat-2S, RISAT, or other sensors.")
            try:
                external_input = inspect_and_gate(str(path), declaration, "SINGLE_IMAGE_VQA", inspector=inspect_generic_raster)
            except (GenericRasterInspectionError, SensorDeclarationError) as exc:
                raise ValueError(str(exc)) from None
            gate = external_input["compatibility_gate"]
            if not gate["eligible_for_agent"]:
                raise ValueError(f"{gate['code']}: {gate['message']}")
        with rasterio.open(path) as source:
            if source.count != len(OPTICAL_BANDS) or (source.height, source.width) != (120, 120):
                raise ValueError("Single-image upload requires a 12-band 120x120 GeoTIFF")
            if source.crs is None or source.res[0] <= 0 or source.res[1] <= 0:
                raise ValueError("Single-image upload requires a CRS and positive resolution")
            descriptions = tuple((value or "").strip() for value in source.descriptions)
            if set(descriptions) == set(OPTICAL_BANDS) and len(descriptions) == len(OPTICAL_BANDS):
                indexes = [descriptions.index(band) + 1 for band in OPTICAL_BANDS]
            elif request.get("band_order_confirmed") is True:
                indexes = list(range(1, len(OPTICAL_BANDS) + 1))
            else:
                raise ValueError("Upload band descriptions must match canonical B01..B12 order, or confirm canonical order")
            # Convert before using NaN as the explicit masked-value sentinel;
            # uint Sentinel products cannot represent NaN in their native dtype.
            optical = source.read(indexes=indexes, masked=True).astype("float32").filled(float("nan"))
            metadata = {"shape": [source.height, source.width], "crs": str(source.crs),
                        "resolution": list(source.res), "bounds": list(source.bounds),
                        "transform": list(source.transform), "optical_band_order": list(OPTICAL_BANDS),
                        "source": "user-provided 12-band GeoTIFF"}
        if route == "SINGLE_IMAGE_VQA":
            metadata["external_input_adapter"] = external_input["adapter"]
            metadata["external_input_compatibility"] = gate
        if not np.isfinite(optical).all():
            raise ValueError("Single-image upload contains masked, nodata, or non-finite pixels")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        image_id = str(request.get("image_id") or f"upload:{digest}")
        return {"route": route, "task": route, "task_type": task_type, "question": query,
                "patch_id": image_id, "image_id": image_id, "split": "external_inference",
                "modality": "S2", "optical": optical, "spatial_metadata": metadata,
                "choices": request.get("choices"), "source_asset_sha256": digest,
                "external_input_inspection": external_input if route == "SINGLE_IMAGE_VQA" else None}
    patch_id = request.get("patch_id") or request.get("image_id")
    if not isinstance(patch_id, str) or not patch_id.strip():
        raise ValueError("A patch_id is required for the single-image route")
    dataset_root = Path(request.get("dataset_root") or get_settings().dataset_root)
    samples = discover_s2_samples(dataset_root)
    sample = next((item for item in samples if item.patch_id == patch_id.strip()), None)
    if sample is None:
        raise FileNotFoundError(f"Non-test S2 patch not found: {patch_id}")
    prepared = load_optical_sample(sample)
    return {
        "route": route,
        "task": route,
        "task_type": task_type,
        "question": query,
        "patch_id": prepared.patch_id,
        "image_id": prepared.patch_id,
        "split": sample.split,
        "modality": "S2",
        "optical": prepared.raw_optical.astype("float32"),
        "spatial_metadata": prepared.metadata,
        "choices": request.get("choices"),
    }


def build_single_sar_request(request, *, uploaded_path=None):
    """Resolve exactly one explicitly declared external Sentinel-1 VV/VH raster."""
    if uploaded_path is None: raise ValueError("SINGLE_IMAGE_SAR_VQA requires one uploaded Sentinel-1 SAR raster.")
    path = Path(uploaded_path)
    if not path.is_file(): raise FileNotFoundError("Uploaded SAR GeoTIFF is no longer available")
    declaration = request.get("sensor_declaration")
    try:
        external = inspect_and_gate(str(path), declaration, "SINGLE_IMAGE_SAR_VQA", inspector=inspect_generic_raster)
    except (GenericRasterInspectionError, SensorDeclarationError) as exc:
        raise ValueError(str(exc)) from None
    gate = external["compatibility_gate"]
    if not gate["eligible_for_agent"]: raise ValueError(f"{gate['code']}: {gate['message']}")
    with rasterio.open(path) as source:
        sar = source.read((1, 2), masked=True).astype("float32").filled(float("nan"))
        metadata = {"shape": [source.height, source.width], "crs": str(source.crs), "resolution": list(source.res), "bounds": list(source.bounds), "transform": list(source.transform), "sar_band_order": ["VV", "VH"], "source": "user-provided declared Sentinel-1 GeoTIFF"}
    if not np.isfinite(sar).all(): raise ValueError("S1 SAR upload contains masked, nodata, or non-finite pixels")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    image_id = str(request.get("image_id") or f"upload:{digest}")
    return {"route": "SINGLE_IMAGE_SAR_VQA", "task_type": str(request.get("task_type") or "binary_qa"), "question": str(request.get("question") or request.get("query") or "").strip(), "image_id": image_id, "split": "external_inference", "sar": sar, "spatial_metadata": metadata, "choices": request.get("choices"), "external_input_inspection": external}


def build_temporal_change_request(request, *, t1_path=None, t2_path=None):
    """Resolve an explicitly ordered RGB PRE/POST pair without reading any dataset split."""
    if not isinstance(request, dict):
        raise TemporalChangeInputError("INVALID_TEMPORAL_REQUEST", "A temporal request object is required")
    query = str(request.get("query") or request.get("question") or "").strip()
    metadata = dict(request.get("metadata") or {})
    metadata["temporal_order"] = request.get("temporal_order", metadata.get("temporal_order"))
    metadata["pair_id"] = request.get("pair_id", metadata.get("pair_id"))
    metadata["split"] = request.get("split", metadata.get("split", "external_inference"))
    t1, t2 = validate_temporal_change_input(
        t1_path=t1_path, t2_path=t2_path, temporal_order=str(metadata.get("temporal_order", "")),
        pair_id=metadata.get("pair_id"), split=str(metadata.get("split", "external_inference")),
    )
    return {"route": "TEMPORAL_CHANGE_DESCRIPTION", "query": query, "metadata": metadata,
            "t1_path": t1.path, "t2_path": t2.path,
            "t1_identity": {"filename": t1.identity, "sha256": t1.sha256, "dimensions": [t1.width, t1.height]},
            "t2_identity": {"filename": t2.identity, "sha256": t2.sha256, "dimensions": [t2.width, t2.height]}}


def cleanup_uploads(tokens=None):
    with UPLOAD_LOCK:
        targets = list(tokens) if tokens is not None else [key for key, item in UPLOADS.items() if time.monotonic() - item["created"] > UPLOAD_TTL]
        for token in targets:
            item = UPLOADS.pop(token, None)
            if item:
                item["temporary"].cleanup()

class HTTPProblem(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message

class Handler(BaseHTTPRequestHandler):
    reports = {}
    server_version = "SatQuery"
    sys_version = ""
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, fmt, *args):
        LOGGER.info("%s", fmt % args)

    def _send(self, data, status=200, content_type="application/json", attachment=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; object-src 'none'; frame-ancestors 'none'")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            LOGGER.info("Client disconnected; request resources released")

    def _json(self, payload, status=200, attachment=None):
        public_payload = sanitize_public_value(payload, local_roots=(PROJECT_ROOT, UPLOAD_ROOT))
        self._send(json.dumps(public_payload, allow_nan=False).encode("utf-8"), status, attachment=attachment)

    def _error(self, status, code, message):
        self._json({"status": "error", "error": {"code": code, "message": message}}, status)

    def _origin(self):
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed:
            raise HTTPProblem(403, "host_rejected", "Only the local application host is accepted.")
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{item}" for item in allowed}:
            raise HTTPProblem(403, "origin_rejected", "Cross-origin requests are not allowed.")

    def _body(self, maximum):
        if self.headers.get("Transfer-Encoding"):
            raise HTTPProblem(400, "invalid_body", "Chunked request bodies are not supported.")
        value = self.headers.get("Content-Length")
        if value is None:
            raise HTTPProblem(411, "length_required", "Content-Length is required.")
        try:
            size = int(value)
        except ValueError:
            raise HTTPProblem(400, "invalid_length", "Invalid Content-Length.") from None
        if size < 0:
            raise HTTPProblem(400, "invalid_length", "Invalid Content-Length.")
        if size > maximum:
            self.close_connection = True
            raise HTTPProblem(413, "body_too_large", f"Request exceeds the {maximum} byte limit.")
        body = self.rfile.read(size)
        if len(body) != size:
            raise HTTPProblem(400, "incomplete_body", "Request body was incomplete.")
        return body

    def do_GET(self):
        try:
            self._origin()
            cleanup_uploads()
            route = urlparse(self.path).path
            static = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/static/style.css": ("style.css", "text/css; charset=utf-8")}
            if route in static:
                filename, mime = static[route]
                self._send((STATIC_ROOT / filename).read_bytes(), content_type=mime)
            elif route == "/favicon.ico":
                self._send(b"", 204)
            elif route == "/api/health":
                self._json({"status": "ok", "service": "SatQuery AI", "busy": ANALYSIS_LOCK.locked()})
            elif route == "/api/v1/health":
                self._json({"status": "ok", "service": "SatQuery AI", "version": "frozen-v1", "busy": ANALYSIS_LOCK.locked(), "runtime": RUNTIME.status()})
            elif route == "/api/v1/runtime/memory":
                try:
                    import torch
                    cuda = bool(torch.cuda.is_available())
                    try:
                        import psutil; rss = int(psutil.Process().memory_info().rss)
                    except Exception: rss = None
                    self._json({"status": "ok", "cuda": cuda, "allocated": int(torch.cuda.memory_allocated()) if cuda else 0, "reserved": int(torch.cuda.memory_reserved()) if cuda else 0, "peak_allocated": int(torch.cuda.max_memory_allocated()) if cuda else 0, "peak_reserved": int(torch.cuda.max_memory_reserved()) if cuda else 0, "rss": rss})
                except Exception:
                    self._json({"status": "ok", "cuda": False, "allocated": 0, "reserved": 0, "peak_allocated": 0, "peak_reserved": 0})
            elif route == "/api/v1/ready":
                preflight = run_preflight()
                routes = {"SINGLE_IMAGE_VQA": "AVAILABLE" if not preflight["errors"] else "BLOCKED", "SINGLE_IMAGE_SAR_VQA": "AVAILABLE_WITH_LIMITATIONS" if not preflight["errors"] else "BLOCKED", "SINGLE_IMAGE_SCENE_DESCRIPTION": "AVAILABLE" if preflight["models"]["scene_model"]["exists"] else "LIMITED", "OPTICAL_SAR_ANALYSIS": "AVAILABLE" if not preflight["errors"] else "BLOCKED", "TEMPORAL_CHANGE_DESCRIPTION": "AVAILABLE" if preflight["models"]["chg2cap"]["exists"] else "BLOCKED", "SINGLE_IMAGE_GROUNDING": "BLOCKED"}
                self._json({"status": "READY_WITH_LIMITATIONS" if preflight["status"] != "BLOCKED" else "BLOCKED", "runtime": preflight["status"], "routes": routes, "warnings": preflight["warnings"], "errors": preflight["errors"]})
            elif route == "/api/samples":
                try:
                    samples = list_pipeline3_area_ids()
                    self._json({"samples": samples, "source": "exact Pipeline 3 manifest"})
                except (ValueError, OSError):
                    try:
                        samples = [item.patch_id for item in discover_samples(get_settings().dataset_root, strict=True)]
                        self._json({"samples": samples, "source": "strict local dataset metadata"})
                    except (ValueError, OSError):
                        self._json({"samples": [], "warning": "Local dataset unavailable. Check DATASET_ROOT or upload imagery."})
            elif route == "/api/v1/samples":
                try:
                    samples = [sample.patch_id for sample in discover_samples(get_settings().dataset_root, strict=True)]
                    self._json({"samples": samples[:100], "route": "MULTIMODAL_S1_S2", "source": "strict local dataset metadata"})
                except (ValueError, OSError, FileNotFoundError):
                    self._json({"samples": [], "route": "MULTIMODAL_S1_S2", "warning": "Local dataset unavailable. Check DATASET_ROOT."})
            elif route == "/api/v1/single-image/samples":
                try:
                    samples = discover_s2_samples(get_settings().dataset_root)
                    self._json({"samples": [sample.patch_id for sample in samples[:100]],
                                "route": "SINGLE_IMAGE_VQA", "modalities": ["S2"],
                                "splits": ["train", "validation"], "source": "S2-only strict metadata"})
                except (ValueError, OSError, FileNotFoundError):
                    self._json({"samples": [], "route": "SINGLE_IMAGE_VQA",
                                "warning": "Non-test S2 dataset unavailable. Check DATASET_ROOT."})
            elif route == "/api/v1/capabilities":
                self._json({"capabilities": capability_registry(), "confidence_policy": {"value": None, "type": "NOT_AVAILABLE"}})
            elif route == "/api/sources":
                status = provider_status()
                self._json({"sources": registry_response(), **status})
            elif re.fullmatch(r"/api/report/[0-9a-f-]{36}", route):
                analysis_id = route.rsplit("/", 1)[1]
                path = REPORT_ROOT / f"{analysis_id}.json"
                if not path.is_file():
                    raise HTTPProblem(404, "not_found", "Report not found.")
                report = json.loads(path.read_text(encoding="utf-8"))
                self._json(report, attachment=f"satquery-{analysis_id}.json")
            else:
                raise HTTPProblem(404, "not_found", "Route not found.")
        except HTTPProblem as exc:
            self._error(exc.status, exc.code, exc.message)
        except Exception:
            LOGGER.exception("GET failed")
            self._error(500, "internal_error", "Request failed. Check server logs.")

    def do_POST(self):
        tokens = []
        acquired = False
        payload, response_status = None, 200
        try:
            self._origin()
            cleanup_uploads()
            route = urlparse(self.path).path
            if route == "/api/upload":
                self._upload()
                return
            if route == "/api/v1/raster/inspect":
                self._inspect_external_raster()
                return
            if route == "/api/v1/raster/pair/inspect":
                self._inspect_external_pair()
                return
            if route == "/api/v1/query":
                if self.headers.get_content_type() != "application/json":
                    raise HTTPProblem(415, "content_type", "Unified SatQuery requests require application/json.")
                try:
                    request = json.loads(self._body(MAX_JSON), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
                except (ValueError, UnicodeError):
                    raise HTTPProblem(400, "invalid_json", "Request body must contain valid finite JSON.") from None
                try:
                    result = run_unified_query(request)
                except (ValueError, TemporalChangeInputError) as exc:
                    raise HTTPProblem(422, "invalid_unified_query", str(exc)) from None
                status = 200 if result["status"] in {"COMPLETED", "BLOCKED"} else 422
                self._json(result, status)
                return
            if route == "/api/v1/temporal":
                if self.headers.get_content_type() != "application/json":
                    raise HTTPProblem(415, "content_type", "Temporal change description requires application/json.")
                try:
                    request = json.loads(self._body(MAX_JSON), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
                except (ValueError, UnicodeError):
                    raise HTTPProblem(400, "invalid_json", "Request body must contain valid finite JSON.") from None
                t1_token, t2_token = request.get("t1_token"), request.get("t2_token")
                if not isinstance(t1_token, str):
                    raise HTTPProblem(422, "TEMPORAL_T1_MISSING", "A valid T1 upload token is required.")
                if not isinstance(t2_token, str):
                    raise HTTPProblem(422, "TEMPORAL_T2_MISSING", "A valid T2 upload token is required.")
                with UPLOAD_LOCK:
                    first, second = UPLOADS.get(t1_token), UPLOADS.get(t2_token)
                    if not first or first["role"] != "t1":
                        raise HTTPProblem(422, "TEMPORAL_T1_MISSING", "T1 upload is invalid or expired.")
                    if not second or second["role"] != "t2":
                        raise HTTPProblem(422, "TEMPORAL_T2_MISSING", "T2 upload is invalid or expired.")
                    first["created"], second["created"] = float("inf"), float("inf")
                    t1_path, t2_path = first["path"], second["path"]
                tokens.extend((t1_token, t2_token))
                try:
                    resolved = build_temporal_change_request(request, t1_path=t1_path, t2_path=t2_path)
                except TemporalChangeInputError as exc:
                    raise HTTPProblem(422, exc.code, str(exc)) from None
                if not TEMPORAL_LOCK.acquire(blocking=False):
                    raise HTTPProblem(409, "temporal_change_busy", "A temporal change description is already running.")
                try:
                    global TEMPORAL_CONTROLLER
                    if TEMPORAL_CONTROLLER is None:
                        TEMPORAL_CONTROLLER = TemporalChangeDescriptionController()
                    result = TEMPORAL_CONTROLLER.run(t1_path=resolved["t1_path"], t2_path=resolved["t2_path"],
                                                     query=resolved["query"], metadata=resolved["metadata"])
                    first_evidence = raster_evidence(resolved["t1_identity"]["filename"], inspect_generic_raster(t1_path))
                    second_evidence = raster_evidence(resolved["t2_identity"]["filename"], inspect_generic_raster(t2_path))
                    result["evidence"] = {"schema": "SATQUERY_GEOSPATIAL_EVIDENCE_V1", "inputs": [first_evidence, second_evidence], "pair": pair_evidence(first_evidence, second_evidence)}
                except RuntimeError as exc:
                    raise HTTPProblem(503, "temporal_specialist_unavailable", str(exc)) from None
                finally:
                    TEMPORAL_LOCK.release()
                self._json(result)
                return
            if route == "/api/v1/single-image":
                if self.headers.get_content_type() != "application/json":
                    raise HTTPProblem(415, "content_type", "Single-image analysis requires application/json.")
                try:
                    request = json.loads(self._body(MAX_JSON), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
                except (ValueError, UnicodeError):
                    raise HTTPProblem(400, "invalid_json", "Request body must contain valid finite JSON.") from None
                upload_token = request.get("upload_token")
                uploaded_path = None
                try:
                    if upload_token is not None:
                        if not isinstance(upload_token, str):
                            raise ValueError("A valid server upload token is required")
                        with UPLOAD_LOCK:
                            upload = UPLOADS.get(upload_token)
                            if not upload or upload["role"] != "optical":
                                raise ValueError("Upload ID is invalid, expired, or not an optical image")
                            upload["created"] = float("inf")
                            uploaded_path = upload["path"]
                    resolved = build_single_image_request(request, uploaded_path=uploaded_path)
                except ValueError as exc:
                    raise HTTPProblem(422, "invalid_single_image_request", str(exc)) from None
                except FileNotFoundError as exc:
                    raise HTTPProblem(404, "single_image_patch_missing", str(exc)) from None
                finally:
                    if isinstance(upload_token, str):
                        cleanup_uploads([upload_token])
                if resolved["route"] == "TEMPORAL_ROUTE_NOT_IMPLEMENTED":
                    self._json({"status": "BLOCKED", "task": resolved["route"], "route": resolved["route"],
                                "execution_trace": [{"step": "route_selection", "status": "BLOCKED"}],
                                "error_code": "TEMPORAL_ROUTE_NOT_IMPLEMENTED", "warnings": ["Temporal analysis is not enabled in this phase."]})
                    return
                if resolved["route"] == "OPTICAL_SAR_ANALYSIS":
                    self._json({"status": "BLOCKED", "task": resolved["route"], "route": resolved["route"],
                                "execution_trace": [{"step": "route_selection", "status": "BLOCKED"}],
                                "error_code": "PAIRED_OPTICAL_SAR_INPUT_REQUIRED", "warnings": ["Use the existing paired S1+S2 route for joint analysis."]})
                    return
                if resolved["route"] == "SINGLE_IMAGE_GROUNDING":
                    from .eo_vlm.grounding_contracts import ContractOnlyGroundingAdapter
                    adapter = ContractOnlyGroundingAdapter()
                    self._json({"status": "BLOCKED", "task": "TEXT_GUIDED_GROUNDING",
                                "route": "SINGLE_IMAGE_GROUNDING", "query": resolved["question"],
                                "image_identity": {"image_id": resolved["image_id"], "split": resolved["split"],
                                                   "dimensions": resolved["spatial_metadata"]["shape"]},
                                "selected_specialist": None, "model_tool": None,
                                "bounding_boxes": [], "mask": None, "confidence": None,
                                "visual_evidence": {"overlay_available": False, "coordinate_space": None,
                                                    "reason": "No approved grounding model or verified box/mask mapping."},
                                "execution_trace": [{"step": "route_selection", "status": "EXECUTED"},
                                                    {"step": "grounding_specialist", "status": "BLOCKED",
                                                     "reason": "learned grounding model is not implemented"}],
                                "provenance": adapter.get_provenance(),
                                "warnings": ["No box, mask, or confidence was produced."],
                                "error_code": "GROUNDING_MODEL_UNAVAILABLE"})
                    return
                if resolved["route"] != "SINGLE_IMAGE_VQA":
                    raise HTTPProblem(422, "unsupported_single_image_task", "The requested single-image task is unsupported.")
                if resolved["task_type"] not in {"binary_qa", "multiple_choice_qa"}:
                    raise HTTPProblem(422, "unsupported_single_image_task", "Only binary and multiple-choice VQA are available.")
                if not SINGLE_IMAGE_LOCK.acquire(blocking=False):
                    raise HTTPProblem(409, "single_image_busy", "A single-image inference is already running.")
                try:
                    global SINGLE_IMAGE_CONTROLLER
                    if SINGLE_IMAGE_CONTROLLER is None:
                        SINGLE_IMAGE_CONTROLLER = SingleImageVQAController()
                    result = SINGLE_IMAGE_CONTROLLER.run(
                        image_id=resolved["image_id"], split=resolved["split"], optical=resolved["optical"],
                        metadata=resolved["spatial_metadata"], question=resolved["question"],
                        task_type=resolved["task_type"], choices=resolved["choices"],
                    )
                    external = resolved.get("external_input_inspection") or {}
                    result["evidence"] = {"schema": "SATQUERY_GEOSPATIAL_EVIDENCE_V1", "inputs": [raster_evidence(resolved["image_id"], external.get("inspection"), declaration=external.get("sensor_declaration"), adapter=external.get("adapter"), compatibility=external.get("compatibility_gate"))]}
                    self._json(result)
                except FileNotFoundError as exc:
                    raise HTTPProblem(503, "single_image_model_unavailable", str(exc)) from None
                except (RuntimeError, ValueError) as exc:
                    raise HTTPProblem(422, "single_image_inference_failed", str(exc)) from None
                finally:
                    SINGLE_IMAGE_LOCK.release()
                return
            if route == "/api/v1/samples":
                dataset_root = get_settings().dataset_root
                try:
                    samples = [sample.patch_id for sample in discover_samples(dataset_root, strict=True)]
                except (ValueError, OSError, FileNotFoundError):
                    samples = []
                self._json({"samples": samples[:100], "route": "MULTIMODAL_S1_S2", "source": "strict local dataset metadata"})
                return
            if route == "/api/v1/satquery":
                if self.headers.get_content_type() != "application/json":
                    raise HTTPProblem(415, "content_type", "V1 paired S1+S2 analysis requires application/json.")
                try:
                    request = json.loads(self._body(MAX_JSON), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
                except (ValueError, UnicodeError):
                    raise HTTPProblem(400, "invalid_json", "Request body must contain valid finite JSON.") from None
                try:
                    resolved = build_v1_route_request(request)
                    controller = SatQueryV1Controller()
                    call_kwargs = {key: value for key, value in resolved.items() if key in {"task_type", "question", "s1", "s2", "patch_id", "s1_patch_id", "s2_patch_id", "spatial_metadata"}}
                    result = controller.run_satquery(**call_kwargs)
                    result["route"] = resolved["route"]
                except ValueError as exc:
                    raise HTTPProblem(422, "invalid_v1_request", str(exc)) from None
                except FileNotFoundError as exc:
                    raise HTTPProblem(404, "dataset_patch_missing", str(exc)) from None
                self._json(result, 422 if result.get("status") == "FAILED" else 200)
                return
            if route == "/api/availability/search":
                if self.headers.get_content_type() != "application/json":
                    raise HTTPProblem(415, "content_type", "Imagery discovery requires application/json.")
                try:
                    request = json.loads(self._body(MAX_JSON), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
                except (ValueError, UnicodeError):
                    raise HTTPProblem(400, "invalid_json", "Request body must contain valid finite JSON.") from None
                try:
                    result = search_imagery(request)
                except ValueError as exc:
                    raise RequestError(str(exc)) from None
                self._json(result)
                return
            if route != "/api/analyze":
                raise HTTPProblem(404, "not_found", "Route not found.")
            if self.headers.get_content_type() != "application/json":
                raise HTTPProblem(415, "content_type", "Analysis requires application/json.")
            try:
                request = json.loads(self._body(MAX_JSON), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
            except (ValueError, UnicodeError):
                raise HTTPProblem(400, "invalid_json", "Request body must contain valid finite JSON.") from None
            validate_request(request)
            if request.get("controller") is True:
                try:
                    controller_result = analyze_controller_request(request)
                except ValueError as exc:
                    raise RequestError(str(exc)) from None
                self._json(controller_result, 422 if controller_result["status"] == "BLOCKED" else 200)
                return
            acquired = ANALYSIS_LOCK.acquire(blocking=False)
            if not acquired:
                raise HTTPProblem(409, "analysis_busy", "An analysis is already running. Wait for it to finish and retry.")
            files = request.get("files", {})
            # Claim uploads under the registry lock; only IDs from this server are usable.
            with UPLOAD_LOCK:
                for role, token in files.items():
                    item = UPLOADS.get(token)
                    if not item or item["role"] != role:
                        raise RequestError("Upload ID is invalid, expired, or assigned to another modality. Upload the file again.")
                tokens = list(files.values())
                request["files"] = {role: str(UPLOADS[token]["path"]) for role, token in files.items()}
                for token in tokens:
                    UPLOADS[token]["created"] = float("inf")  # protected until this request's finally block
            task_request = build_task_request(request)
            adapted = run_analysis(task_request)
            result = dict(adapted.legacy_result)
            result["task_result"] = adapted.task_result.to_dict()
            result["scene_contract"] = adapted.scene.to_dict()
            result["capability_route"] = {
                "task_request": TASK_TYPE,
                "registry": "authoritative",
                "capability": "deterministic_scene_analysis",
                "adapter": "Pipeline3AnalysisAdapter",
            }
            save_report(result, REPORT_ROOT)
            payload, response_status = result, 422 if result["status"] == "rejected" else 200
        except HTTPProblem as exc:
            payload, response_status = {"status": "error", "error": {"code": exc.code, "message": exc.message}}, exc.status
        except RequestError as exc:
            payload, response_status = {"status": "error", "error": {"code": "invalid_request", "message": str(exc)}}, 400
        except IdentityResolutionError as exc:
            payload, response_status = {"status": "error", "error": {"code": "identity_resolution_failed", "message": str(exc)}}, 422
        except ModelContractError as exc:
            payload, response_status = {"status": "error", "error": {"code": "model_contract_failed", "message": str(exc)}}, 422
        except (socket.timeout, TimeoutError):
            payload, response_status = {"status": "error", "error": {"code": "request_timeout", "message": "Request timed out. Retry the upload or analysis."}}, 408
        except Exception:
            LOGGER.exception("POST failed")
            payload, response_status = {"status": "error", "error": {"code": "internal_error", "message": "Analysis failed. Retry or check server logs."}}, 500
        finally:
            cleanup_uploads(tokens)
            if acquired:
                ANALYSIS_LOCK.release()
        if payload is not None:
            self._json(payload, response_status)

    def _upload(self):
        if self.headers.get_content_type() != "multipart/form-data":
            raise HTTPProblem(415, "content_type", "Upload requires multipart/form-data.")
        body = self._body(MAX_UPLOAD)
        message = BytesParser(policy=policy.default).parsebytes(
            ("Content-Type: " + self.headers["Content-Type"] + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body)
        if not message.is_multipart():
            raise HTTPProblem(400, "invalid_multipart", "Malformed multipart upload.")
        staged = {}
        try:
            for part in message.iter_parts():
                role = part.get_param("name", header="content-disposition")
                filename = part.get_filename()
                if not filename or role not in {"optical", "sar", "before", "after", "t1", "t2", "scene"} or role in staged:
                    raise HTTPProblem(400, "invalid_upload_role", "Each file needs a unique optical, sar, before, after, t1, t2, or scene role.")
                allowed = {".png", ".jpg", ".jpeg", ".tif", ".tiff"} if role in {"t1", "t2", "scene"} else {".tif", ".tiff"}
                if Path(filename).suffix.lower() not in allowed:
                    raise HTTPProblem(415, "unsupported_file", "T1/T2 inputs may be PNG, JPEG, or TIFF; other routes require GeoTIFF.")
                UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
                temporary = tempfile.TemporaryDirectory(prefix="request-", dir=UPLOAD_ROOT)
                suffix = Path(filename).suffix.lower()
                path = Path(temporary.name) / f"image{suffix}"
                token = str(uuid4())
                staged[role] = {"temporary": temporary, "path": path, "created": time.monotonic(), "role": role, "token": token}
                path.write_bytes(part.get_payload(decode=True) or b"")
                try:
                    if role in {"t1", "t2"}:
                        validate_temporal_image_path(path, role.upper())
                    elif role == "scene":
                        load_rgb_image(path)
                    else:
                        info = inspect_raster(path)
                        expected = 2 if role == "sar" else 12
                        if info["band_count"] != expected or not info["crs"] or not info["finite"]:
                            raise ValueError(f"{role} requires {expected} finite bands and a CRS.")
                except (ValueError, TemporalChangeInputError) as exc:
                    raise HTTPProblem(422, "invalid_raster", str(exc)) from None
            if not staged:
                raise HTTPProblem(400, "empty_upload", "No imagery was uploaded.")
            with UPLOAD_LOCK:
                if len(UPLOADS) + len(staged) > 32:
                    raise HTTPProblem(429, "upload_capacity", "Too many pending uploads; run an analysis or wait for expiry.")
                UPLOADS.update({item["token"]: item for item in staged.values()})
            self._json({"status": "uploaded", "files": {role: item["token"] for role, item in staged.items()}, "expires_in_seconds": UPLOAD_TTL})
        except Exception:
            for item in staged.values():
                item["temporary"].cleanup()
            raise

    def _inspect_external_raster(self):
        """Inspect one bounded TIFF/PNG/JPEG without inferring sensor identity."""
        if self.headers.get_content_type() != "multipart/form-data":
            raise HTTPProblem(415, "content_type", "Raster inspection requires multipart/form-data.")
        body = self._body(MAX_UPLOAD)
        message = BytesParser(policy=policy.default).parsebytes(
            ("Content-Type: " + self.headers["Content-Type"] + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body)
        if not message.is_multipart():
            raise HTTPProblem(400, "invalid_multipart", "Malformed raster inspection upload.")
        file_part = declaration_text = requested_route = None
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "raster" and part.get_filename():
                if file_part is not None:
                    raise HTTPProblem(400, "duplicate_raster", "Submit exactly one raster file.")
                file_part = part
            elif name == "declaration" and not part.get_filename():
                declaration_text = part.get_content()
            elif name == "requested_route" and not part.get_filename():
                requested_route = part.get_content()
        if file_part is None:
            raise HTTPProblem(400, "raster_required", "Submit one TIFF, PNG, or JPEG file in the raster field.")
        declaration = None
        if declaration_text is not None:
            try:
                declaration = json.loads(declaration_text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite")))
            except (ValueError, UnicodeError):
                raise HTTPProblem(400, "invalid_sensor_declaration", "The sensor declaration must be valid finite JSON.") from None
        suffix = Path(file_part.get_filename() or "").suffix.lower()
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="raster-inspection-", dir=UPLOAD_ROOT)
        try:
            path = Path(temporary.name) / f"raster{suffix}"
            path.write_bytes(file_part.get_payload(decode=True) or b"")
            result = inspect_and_gate(str(path), declaration, str(requested_route or "") or None, inspector=inspect_generic_raster)
            result["evidence"] = raster_evidence(result["inspection"].get("file_name", "external_raster"), result["inspection"], declaration=result.get("sensor_declaration"), adapter=result.get("adapter"), compatibility=result.get("compatibility_gate"))
            self._json(result)
        except GenericRasterInspectionError as exc:
            raise HTTPProblem(422, exc.code, str(exc)) from None
        except SensorDeclarationError as exc:
            raise HTTPProblem(422, exc.code, str(exc)) from None
        finally:
            temporary.cleanup()

    def _inspect_external_pair(self):
        """Validate two declared external files without alignment or model use."""
        if self.headers.get_content_type() != "multipart/form-data":
            raise HTTPProblem(415, "content_type", "Pair inspection requires multipart/form-data.")
        body = self._body(MAX_UPLOAD)
        message = BytesParser(policy=policy.default).parsebytes(
            ("Content-Type: " + self.headers["Content-Type"] + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body)
        if not message.is_multipart():
            raise HTTPProblem(400, "invalid_multipart", "Malformed pair inspection upload.")
        parts = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name:
                parts[name] = part
        first, second = parts.get("first"), parts.get("second")
        if first is None or second is None or not first.get_filename() or not second.get_filename():
            raise HTTPProblem(400, "pair_inputs_required", "Submit exactly two external files named first and second.")
        try:
            first_decl = json.loads(parts["first_declaration"].get_content())
            second_decl = json.loads(parts["second_declaration"].get_content())
            pair_kind = parts["pair_kind"].get_content()
        except (KeyError, ValueError, UnicodeError):
            raise HTTPProblem(400, "invalid_sensor_declaration", "Both pair declarations and pair_kind are required valid JSON/text fields.") from None
        pair_id = parts.get("pair_id").get_content() if parts.get("pair_id") else None
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="raster-pair-inspection-", dir=UPLOAD_ROOT)
        try:
            first_path = Path(temporary.name) / f"first{Path(first.get_filename()).suffix.lower()}"
            second_path = Path(temporary.name) / f"second{Path(second.get_filename()).suffix.lower()}"
            first_path.write_bytes(first.get_payload(decode=True) or b"")
            second_path.write_bytes(second.get_payload(decode=True) or b"")
            first_result = inspect_and_gate(str(first_path), first_decl, None, inspector=inspect_generic_raster)
            second_result = inspect_and_gate(str(second_path), second_decl, None, inspector=inspect_generic_raster)
            pair = validate_external_pair(first_result, second_result, pair_kind=str(pair_kind), pair_id=str(pair_id) if pair_id else None)
            first_evidence = raster_evidence("first", first_result["inspection"], declaration=first_result.get("sensor_declaration"), adapter=first_result.get("adapter"), compatibility=first_result.get("compatibility_gate"))
            second_evidence = raster_evidence("second", second_result["inspection"], declaration=second_result.get("sensor_declaration"), adapter=second_result.get("adapter"), compatibility=second_result.get("compatibility_gate"))
            self._json({"status": "INSPECTED", "task": "EXTERNAL_PAIR_INSPECTION", "first": first_result, "second": second_result, "pair_compatibility": pair, "evidence": {"inputs": [first_evidence, second_evidence], "pair": pair_evidence(first_evidence, second_evidence)}})
        except (GenericRasterInspectionError, SensorDeclarationError) as exc:
            raise HTTPProblem(422, exc.code, str(exc)) from None
        finally:
            temporary.cleanup()

    def do_OPTIONS(self):
        try:
            self._origin()
            self._send(b"", 204)
        except HTTPProblem as exc:
            self._error(exc.status, exc.code, exc.message)

    def do_PUT(self):
        self._error(405, "method_not_allowed", "Use GET for retrieval or POST for analysis and uploads.")

    do_PATCH = do_PUT
    do_DELETE = do_PUT

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Serve SatQuery AI locally.")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    log_dir = PROJECT_ROOT / "experiments" / "outputs" / "runtime_logs"; log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "satquery-runtime.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), handler])
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    recovery = recover_staging(
        UPLOAD_ROOT,
        ttl_seconds=UPLOAD_TTL,
        active_paths=(item["temporary"].name for item in UPLOADS.values()),
    )
    LOGGER.info(
        "Upload staging recovery complete: deleted=%d preserved=%d",
        sum(bool(item["deleted"]) for item in recovery),
        sum(not item["deleted"] for item in recovery),
    )
    stop_cleanup = threading.Event()
    def housekeeping():
        while not stop_cleanup.wait(30):
            cleanup_uploads()
    threading.Thread(target=housekeeping, daemon=True).start()
    print(f"SatQuery AI: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_cleanup.set()
        server.server_close()
        cleanup_uploads(list(UPLOADS)); RUNTIME.release_all()

if __name__ == "__main__":
    main()
