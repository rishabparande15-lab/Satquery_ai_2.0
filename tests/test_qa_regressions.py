"""Failure-path regressions use synthetic fixtures, never synthetic accuracy claims."""
import json
import threading
import time
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src import api, analysis_engine as engine
from src.input_validation import inspect_raster, validate_aoi
from src.query_interpreter import interpret_query
from src.tool_selector import select_tools


@pytest.fixture
def raster(tmp_path):
    def make(name="optical", bands=12, dtype="float32", crs="EPSG:32633", resolution=10, finite=True, shape=120, described=False):
        path = tmp_path / (name + ".tif")
        values = np.arange(bands * shape * shape, dtype=np.float32).reshape(bands, shape, shape) + 100
        if not finite:
            values[0, 0, 0] = np.nan
        with rasterio.open(path, "w", driver="GTiff", count=bands, width=shape, height=shape, dtype=dtype, crs=crs, transform=from_origin(400000, 5000000, resolution, resolution)) as ds:
            ds.write(values.astype(dtype))
            if described:
                from src.dataset_loader import OPTICAL_BANDS, SAR_BANDS
                ds.descriptions = OPTICAL_BANDS if bands == 12 else SAR_BANDS
        return path
    return make


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "REPORT_ROOT", tmp_path / "reports")
    monkeypatch.setattr(api, "UPLOAD_ROOT", tmp_path / "uploads")
    server = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    def call(method, route, body=None, headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
            headers = {"Content-Type": "application/json", **(headers or {})}
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=30)
        connection.request(method, route, body=body, headers=headers or {})
        response = connection.getresponse()
        raw = response.read()
        data = json.loads(raw) if raw and "application/json" in response.getheader("Content-Type", "") else raw
        result = response.status, data, dict(response.getheaders())
        connection.close()
        return result
    yield call
    server.shutdown()
    server.server_close()
    worker.join()
    api.cleanup_uploads(list(api.UPLOADS))


def multipart(role, filename, data):
    boundary = "SatQueryTestBoundary"
    raw = (f'--{boundary}\r\nContent-Disposition: form-data; name="{role}"; filename="{filename}"\r\nContent-Type: image/tiff\r\n\r\n').encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return raw, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


@pytest.mark.parametrize("value", [None, [], {}, {"query": ""}, {"query": "   "}, {"query": 7}, {"query": "x", "files": []}, {"query": "x", "analysis_type": "imaginary"}])
def test_invalid_request_has_controlled_json(client, value):
    body = "null" if value is None else value
    status, data, _ = client("POST", "/api/analyze", body, {"Content-Type": "application/json"})
    assert status == 400
    assert isinstance(data["error"]["message"], str)
    assert "Traceback" not in json.dumps(data)


@pytest.mark.parametrize("aoi", [{}, [], {"type": "polygon"}, {"type": "point", "latitude": 91, "longitude": 2}, {"type": "bbox", "bounds": [-200, 0, 3, 5]}, {"type": "bbox", "bounds": [0, 0, float("nan"), 2]}])
def test_invalid_aoi_rejected(aoi):
    with pytest.raises(ValueError):
        validate_aoi(aoi)


def test_parser_preserves_dates_and_explicit_temporal_mode():
    p = interpret_query("From December 2025 to January 2020")
    assert (p.start_date, p.end_date) == ("2025-12-01", "2020-01-01")
    assert interpret_query("Analyze input", analysis_type="water_change_analysis").requires_temporal_pair
    assert interpret_query("SAR data", analysis_type="optical_analysis").modalities == ["optical"]


def test_integer_geotiff_is_readable(raster):
    assert inspect_raster(raster(dtype="uint16"))["finite"]


@pytest.mark.parametrize("kwargs", [{"bands": 3}, {"finite": False}, {"crs": None}])
def test_invalid_upload_cleaned(client, raster, kwargs):
    p = raster(**kwargs)
    raw, headers = multipart("optical", p.name, p.read_bytes())
    status, data, _ = client("POST", "/api/upload", raw, headers)
    assert status == 422
    assert data["error"]["code"] == "invalid_raster"
    assert not list(api.UPLOAD_ROOT.glob("request-*"))


@pytest.mark.parametrize("filename,data,expected", [("bad.txt", b"x", 415), ("bad.tif", b"not a tiff", 422)])
def test_corrupt_and_unsupported_upload(client, filename, data, expected):
    raw, headers = multipart("optical", filename, data)
    assert client("POST", "/api/upload", raw, headers)[0] == expected
    assert not list(api.UPLOAD_ROOT.glob("request-*"))


@pytest.mark.parametrize("sar_kwargs", [{"crs": "EPSG:32632"}, {"resolution": 20}, {"shape": 64}])
def test_joint_rejects_grid_mismatch(raster, sar_kwargs):
    request = {"query": "joint optical SAR", "files": {"optical": str(raster()), "sar": str(raster("sar", bands=2, **sar_kwargs))}, "band_order_confirmed": True}
    r = engine.run_analysis(request)
    assert r["status"] == "rejected"
    assert not all(r["validation"]["checks"]["optical_sar_compatibility"].values())
    assert r["model_results"]["prediction"] is None


def test_missing_sar_and_empty_input_rejected(raster):
    for files in ({}, {"optical": str(raster())}):
        r = engine.run_analysis({"query": "joint optical SAR", "files": files})
        assert not r["validation"]["valid"]
        assert len(r["execution_trace"]) == 15


def test_temporal_mode_cannot_bypass_validation(raster):
    r = engine.run_analysis({"query": "Analyze region", "analysis_type": "water_change_analysis", "sample_id": "61_39"})
    assert r["status"] == "rejected"
    assert r["temporal"]["status"] == "requires valid input"
    assert r["model_results"]["prediction"] is None
    assert r["features"]["temporal"]["status"] == "unavailable"


def test_validated_temporal_pair_does_not_claim_algorithm(raster):
    before, after = raster("before"), raster("after")
    for path, day in ((before, "2020-01-01"), (after, "2025-01-01")):
        with rasterio.open(path, "r+") as ds:
            ds.update_tags(acquisition_date=day)
    r = engine.run_analysis({"query": "compare water", "files": {"before": str(before), "after": str(after)}})
    assert r["validation"]["checks"]["temporal_pair_valid"]
    assert r["status"] == "unavailable"
    assert "not implemented" in r["temporal"]["message"]


@pytest.mark.parametrize("mode,bands,dimension", [("optical", 12, 52), ("sar", 2, 9)])
def test_single_modality_without_checkpoint_keeps_real_statistics(raster, mode, bands, dimension):
    with patch.object(engine, "croma_available", return_value=False):
        r = engine.run_analysis({"query": f"Analyze {mode}", "files": {mode: str(raster(mode, bands=bands))}, "band_order_confirmed": True})
    assert r["status"] == "partial"
    assert r["data_cube"]["modalities"] == [mode]
    assert r["features"]["spectral"]["dimension"] == dimension
    assert r["features"]["deep"]["status"] == "unavailable"
    assert r["model_results"]["prediction"] is None
    assert r["confidence"]["model_confidence"] is None
    assert r["confidence"]["accuracy_status"] == "no accuracy claim"
    json.dumps(r, allow_nan=False)


def test_croma_failure_retains_statistics_without_path_leak(raster):
    with patch.object(engine, "croma_available", return_value=True), patch.object(engine, "_deep_features", side_effect=RuntimeError("C:/private/checkpoint key=secret")):
        r = engine.run_analysis({"query": "optical", "files": {"optical": str(raster())}, "band_order_confirmed": True})
    assert r["status"] == "partial"
    assert r["features"]["spectral"]["dimension"] == 52
    assert "secret" not in json.dumps(r)
    assert "C:/" not in json.dumps(r)
    assert "retained" in r["llm_explanation"]


def test_upload_token_consumed_report_downloadable(client, raster):
    path = raster(shape=16)
    raw, headers = multipart("optical", path.name, path.read_bytes())
    status, upload, _ = client("POST", "/api/upload", raw, headers)
    assert status == 200
    assert not Path(upload["files"]["optical"]).is_absolute()
    status, report, _ = client("POST", "/api/analyze", {"query": "optical", "files": upload["files"], "band_order_confirmed": True})
    assert status == 200
    assert report["status"] == "partial"
    assert not api.UPLOADS
    assert not list(api.UPLOAD_ROOT.glob("request-*"))
    api.Handler.reports.clear()  # Downloads no longer depend on an in-memory registry.
    status, downloaded, headers = client("GET", "/api/report/" + report["analysis_id"])
    assert status == 200 and downloaded == report
    assert headers["Content-Disposition"].startswith("attachment")
    assert str(path.parent) not in json.dumps(report)


def test_raw_filesystem_path_never_accepted_by_api(client):
    status, _, _ = client("POST", "/api/analyze", {"query": "optical", "files": {"optical": "C:/private/file.tif"}})
    assert status == 400


def test_upload_expiration_cleanup(client, raster):
    path = raster()
    raw, headers = multipart("optical", path.name, path.read_bytes())
    _, data, _ = client("POST", "/api/upload", raw, headers)
    token = data["files"]["optical"]
    api.UPLOADS[token]["created"] = time.monotonic() - api.UPLOAD_TTL - 1
    api.cleanup_uploads()
    assert not api.UPLOADS
    assert not list(api.UPLOAD_ROOT.glob("request-*"))


def test_http_boundaries_and_static_assets(client, monkeypatch):
    assert client("GET", "/")[0] == 200
    assert client("GET", "/static/app.js")[0] == 200
    assert client("GET", "/static/style.css")[0] == 200
    assert client("GET", "/missing")[0] == 404
    assert client("GET", "/api/report/" + "a" * 36)[0] == 404
    assert client("POST", "/api/analyze", "NaN", {"Content-Type": "application/json"})[0] == 400
    assert client("POST", "/api/analyze", "{}", {"Content-Type": "text/plain"})[0] == 415
    assert client("GET", "/api/health", headers={"Origin": "https://example.org"})[0] == 403
    assert client("OPTIONS", "/api/analyze", headers={"Origin": "https://example.org"})[0] == 403
    monkeypatch.setattr(api, "MAX_UPLOAD", 4)
    assert client("POST", "/api/upload", b"12345", {"Content-Type": "multipart/form-data; boundary=x"})[0] == 413


def test_duplicate_requests_rejected_health_remains_responsive(client):
    with api.ANALYSIS_LOCK:
        assert client("POST", "/api/analyze", {"query": "optical"})[0] == 409
        status, health, _ = client("GET", "/api/health")
        assert status == 200 and health["busy"]


def test_internal_failure_format_does_not_expose_exception(client):
    with patch.object(api, "run_analysis", side_effect=RuntimeError("secret")):
        status, body, _ = client("POST", "/api/analyze", {"query": "optical"})
    assert status == 500 and body["error"]["code"] == "internal_error"
    assert "secret" not in json.dumps(body)
    assert not api.ANALYSIS_LOCK.locked()


def test_http_analysis_constructs_task_request_and_uses_registry_dispatch(client, monkeypatch):
    from types import SimpleNamespace
    from src.architecture_contracts import TaskRequest
    from src.deterministic_scene_analysis import TASK_TYPE

    observed = []
    legacy = {
        "analysis_id": "2c706ffc-bb67-4680-bdcd-df356f80f9f5",
        "status": "completed", "features": {}, "validation": {}, "warnings": [],
    }
    class Result:
        status = "success"
        def to_dict(self): return {"status": self.status, "task_type": TASK_TYPE}
    class Scene:
        def to_dict(self): return {"scene_id": "61_39"}
    def dispatch(request):
        assert isinstance(request, TaskRequest)
        observed.append(request)
        return SimpleNamespace(legacy_result=legacy, task_result=Result(), scene=Scene())
    monkeypatch.setattr(api, "run_analysis", dispatch)
    monkeypatch.setattr(api, "save_report", lambda *args, **kwargs: None)

    status, body, _ = client("POST", "/api/analyze", {"query": "optical", "sample_id": "61_39"})
    assert status == 200
    assert len(observed) == 1 and observed[0].task_type == TASK_TYPE
    assert body["capability_route"]["registry"] == "authoritative"


@pytest.mark.parametrize("dates", [("2025-01-01", "2020-01-01"), ("2025-02-30", None)])
def test_invalid_date_is_controlled(dates):
    r = engine.run_analysis({"query": "optical", "sample_id": "61_39", "start_date": dates[0], "end_date": dates[1]})
    assert r["status"] == "rejected"
    assert not r["validation"]["checks"]["dates"]


def test_unavailable_tools_never_selected():
    p = interpret_query("Describe scene", analysis_type="single_image_vqa").to_dict()
    s = select_tools(p, {"croma": False})
    assert "single_image_vqa" not in s.selected_tools
    assert any(t["tool"] == "single_image_vqa" for t in s.unavailable_tools)


def test_frontend_state_regressions():
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for pure JavaScript UI state tests")
    process = subprocess.run([node, "--test", str(Path(__file__).with_name("frontend_state.test.cjs"))], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr


def test_bad_iso_month_is_not_silently_reinterpreted():
    r = engine.run_analysis({"query": "optical on 2025-13-01", "sample_id": "61_39"})
    assert not r["validation"]["checks"]["dates"]


def test_aoi_non_intersection_is_rejected(raster):
    r = engine.run_analysis({"query": "optical", "files": {"optical": str(raster())},
                             "aoi": {"type": "point", "latitude": 0, "longitude": 0}})
    assert r["status"] == "rejected"
    assert any("intersect" in message for message in r["validation"]["errors"])


def test_unknown_band_order_is_not_assumed(raster):
    r = engine.run_analysis({"query": "optical", "files": {"optical": str(raster())}})
    assert r["status"] == "rejected"
    assert "band-order" in r["llm_explanation"]


def test_wrong_sar_band_count_rejected(client, raster):
    path = raster("sar", bands=1)
    body, headers = multipart("sar", "sar.tif", path.read_bytes())
    assert client("POST", "/api/upload", body, headers)[0] == 422


def test_report_id_is_immutable_and_json_is_finite(tmp_path):
    r = engine.run_analysis({"query": "optical"})
    engine.save_report(r, tmp_path)
    with pytest.raises(FileExistsError):
        engine.save_report(r, tmp_path)
    r["analysis_id"] = "../unsafe"
    with pytest.raises(ValueError):
        engine.save_report(r, tmp_path)


def test_cpu_device_selected_without_cuda(monkeypatch):
    from types import SimpleNamespace
    import torch
    seen=[]
    class Adapter:
        def __init__(self, source, checkpoint, device):
            seen.append(device)
            self.device = torch.device(device)
        def infer_modality(self, optical, sar):
            return {"optical_GAP": torch.ones(1, 768)}
    monkeypatch.setattr(engine, "_adapter", None)
    monkeypatch.setattr(engine, "CROMAAdapter", Adapter)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    deep, hybrid = engine._deep_features({"optical": np.ones((12,120,120),dtype="float32")}, np.ones(52,dtype="float32"))
    assert seen == ["cpu"] and deep["device"] == "cpu"
    assert hybrid["status"] == "not applicable"


def test_temporal_metadata_needs_actual_grid():
    from src.temporal import validate_temporal_pair, TemporalPairError, compare_representations
    with pytest.raises(TemporalPairError):
        validate_temporal_pair({"sample_id": "b", "acquisition_date": "2020-01-01"}, {"sample_id": "a", "acquisition_date": "2021-01-01"})
    with pytest.raises(TemporalPairError):
        compare_representations(np.array([]), np.array([]))
    assert np.isfinite(compare_representations(np.array([-3e38]), np.array([3e38]))["l2_distance"])


def test_three_sample_training_guard_cannot_be_bypassed():
    from src.train_landcover import validate_training_population
    from src.training_data import make_splits
    with pytest.raises(RuntimeError, match="prohibited"):
        validate_training_population(make_splits(["61_39", "61_40", "61_41"]))


def test_backend_timeout_is_controlled_and_releases_lock(client):
    with patch.object(api, "run_analysis", side_effect=TimeoutError):
        status, error, _ = client("POST", "/api/analyze", {"query": "optical"})
    assert status == 408 and error["error"]["code"] == "request_timeout"
    assert not api.ANALYSIS_LOCK.locked()
