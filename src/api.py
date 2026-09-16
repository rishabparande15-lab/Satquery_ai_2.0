"""Loopback HTTP application with bounded uploads and serialized inference."""
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
from urllib.parse import urlparse
from uuid import uuid4

from .analysis_engine import RequestError, save_report, validate_request
from .architecture_contracts import TaskRequest
from .config import PROJECT_ROOT, get_settings
from .dataset_loader import discover_samples
from .deterministic_scene_analysis import TASK_TYPE, run_deterministic_scene_analysis_with_context
from .input_validation import inspect_raster
from .imagery_availability import provider_status, registry_response, search_imagery
from .pipeline3_scene_probe import IdentityResolutionError, ModelContractError, list_pipeline3_area_ids
from .query_interpreter import interpret_query

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
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            LOGGER.info("Client disconnected; request resources released")

    def _json(self, payload, status=200):
        self._send(json.dumps(payload, allow_nan=False).encode("utf-8"), status)

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
            elif route == "/api/sources":
                status = provider_status()
                self._json({"sources": registry_response(), **status})
            elif re.fullmatch(r"/api/report/[0-9a-f-]{36}", route):
                analysis_id = route.rsplit("/", 1)[1]
                path = REPORT_ROOT / f"{analysis_id}.json"
                if not path.is_file():
                    raise HTTPProblem(404, "not_found", "Report not found.")
                self._send(path.read_bytes(), attachment=f"satquery-{analysis_id}.json")
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
                if not filename or role not in {"optical", "sar", "before", "after"} or role in staged:
                    raise HTTPProblem(400, "invalid_upload_role", "Each file needs a unique optical, sar, before or after role.")
                if Path(filename).suffix.lower() not in {".tif", ".tiff"}:
                    raise HTTPProblem(415, "unsupported_file", "Only GeoTIFF (.tif/.tiff) imagery is supported.")
                UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
                temporary = tempfile.TemporaryDirectory(prefix="request-", dir=UPLOAD_ROOT)
                path = Path(temporary.name) / "image.tif"
                token = str(uuid4())
                staged[role] = {"temporary": temporary, "path": path, "created": time.monotonic(), "role": role, "token": token}
                path.write_bytes(part.get_payload(decode=True) or b"")
                try:
                    info = inspect_raster(path)
                    expected = 2 if role == "sar" else 12
                    if info["band_count"] != expected or not info["crs"] or not info["finite"]:
                        raise ValueError(f"{role} requires {expected} finite bands and a CRS.")
                except ValueError as exc:
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
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
        cleanup_uploads(list(UPLOADS))

if __name__ == "__main__":
    main()
