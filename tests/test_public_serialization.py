import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from http.server import ThreadingHTTPServer
from urllib.request import urlopen
from uuid import uuid4

import pytest

from src.eo_vlm_adapter import MODEL_ID, Qwen25VLRGBAdapter
from src.public_serialization import sanitize_public_value
from src.single_image_vqa import QWEN_REVISION


@pytest.mark.parametrize(
    "route",
    [
        "SINGLE_IMAGE_VQA",
        "SINGLE_IMAGE_SCENE_DESCRIPTION",
        "OPTICAL_SAR_ANALYSIS",
        "TEMPORAL_CHANGE_DESCRIPTION",
    ],
)
def test_api_json_serialization_redacts_local_paths_and_preserves_provenance(monkeypatch, route):
    from src import api

    local_snapshot = r"C:\Users\Rishab\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots" + "\\" + QWEN_REVISION
    payload = {
        "route": route,
        "model_id": MODEL_ID,
        "qwen_revision": QWEN_REVISION if route != "OPTICAL_SAR_ANALYSIS" else local_snapshot,
        "evidence": {"filename": "input.tif", "crs": "EPSG:32629", "band_count": 12},
        "provenance": {
            "model_id": MODEL_ID,
            "qwen_revision": QWEN_REVISION if route != "OPTICAL_SAR_ANALYSIS" else local_snapshot,
            "checkpoint_sha256": "a" * 64,
            "resolved_model_location": local_snapshot,
        },
    }
    sent = {}
    monkeypatch.setattr(api.Handler, "_send", lambda self, data, status=200, **kwargs: sent.update(body=data, status=status))
    handler = object.__new__(api.Handler)
    handler._json(payload)
    response = json.loads(sent["body"])
    export = json.loads(json.dumps(response))

    assert sent["status"] == 200
    assert "C:" not in json.dumps(export)
    assert "huggingface" not in json.dumps(export).lower()
    assert export["route"] == route
    assert export["model_id"] == MODEL_ID
    assert export["provenance"]["model_id"] == MODEL_ID
    if route == "OPTICAL_SAR_ANALYSIS":
        assert export["qwen_revision"] == "[local path redacted]"
        assert export["provenance"]["qwen_revision"] == "[local path redacted]"
    else:
        assert export["provenance"]["qwen_revision"] == QWEN_REVISION
    assert export["provenance"]["checkpoint_sha256"] == "a" * 64
    assert export["evidence"]["filename"] == "input.tif"
    assert export["evidence"]["crs"] == "EPSG:32629"


def test_saved_json_report_download_uses_public_sanitizer(monkeypatch, tmp_path):
    from src import api

    monkeypatch.setattr(api, "REPORT_ROOT", tmp_path)
    report_id = str(uuid4())
    report_path = tmp_path / f"{report_id}.json"
    report_path.write_text(json.dumps({"provenance": {"cache": "D:/private/huggingface/hub/models--Qwen/snapshots/rev"}}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/report/{report_id}") as response:
            body = response.read().decode("utf-8")
            assert response.status == 200
            assert response.headers["Content-Disposition"].endswith(f"satquery-{report_id}.json\"")
        assert "huggingface" not in body.lower()
        assert "D:/" not in body
        assert json.loads(body)["provenance"]["cache"] == "[local path redacted]"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.mark.parametrize("path", [r"D:\data\upload.tif", "/home/user/model/cache", "/tmp/satquery-upload/image.tif"])
def test_public_path_sanitizer_redacts_local_paths(path):
    assert sanitize_public_value({"value": path})["value"] == "[local path redacted]"


@pytest.mark.parametrize(
    "path",
    [
        "experiments/outputs/web_uploads/request-ab12cd34/image.tif",
        "web_uploads/raster-inspection-12345678/raster.tif",
        "models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
    ],
)
def test_public_path_sanitizer_redacts_relative_local_path_fragments(path):
    assert sanitize_public_value({"value": path})["value"] == "[local path redacted]"


def test_public_path_sanitizer_preserves_model_ids_and_logical_revision():
    public = sanitize_public_value({"model_id": MODEL_ID, "qwen_revision": QWEN_REVISION})
    assert public == {"model_id": MODEL_ID, "qwen_revision": QWEN_REVISION}


def test_qwen_loader_still_receives_local_snapshot_path(monkeypatch):
    calls = []
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoProcessor = SimpleNamespace(
        from_pretrained=lambda path, **kwargs: calls.append(("processor", path, kwargs)) or object()
    )
    fake_transformers.AutoModelForImageTextToText = SimpleNamespace(
        from_pretrained=lambda path, **kwargs: calls.append(("model", path, kwargs)) or object()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16", bfloat16="bfloat16"))

    local_snapshot = Path("C:/Users/Rishab/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots") / QWEN_REVISION
    runtime = Qwen25VLRGBAdapter().load_model(local_snapshot, dtype="float16", device="cuda")

    assert [call[1] for call in calls] == [str(local_snapshot), str(local_snapshot)]
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert runtime["model"] is not None


def test_optical_sar_controller_keeps_local_snapshot_internal(monkeypatch):
    from src import satquery_v1

    calls = []

    class FakeModel:
        def eval(self):
            return self

    class FakeCroma:
        model = SimpleNamespace(parameters=lambda: ())

    def load_model(_adapter, snapshot, **_kwargs):
        calls.append(snapshot)
        return {"model": FakeModel(), "processor": SimpleNamespace(tokenizer=object())}

    monkeypatch.setattr(satquery_v1, "load_verified_joint_projector", lambda *_args, **_kwargs: (object(), "joint-sha", "fingerprint"))
    monkeypatch.setattr(satquery_v1, "CROMAAdapter", lambda *_args, **_kwargs: FakeCroma())
    monkeypatch.setattr(satquery_v1.Qwen25VLRGBAdapter, "load_model", load_model)
    monkeypatch.setattr(satquery_v1, "freeze_qwen", lambda _model: None)
    monkeypatch.setattr(satquery_v1, "hash_state", lambda _model: "qwen-fingerprint")
    monkeypatch.setattr(satquery_v1, "module_fingerprint", lambda _model: "fingerprint")

    controller = satquery_v1.SatQueryV1Controller(device="cpu")
    controller.load()

    assert calls == [satquery_v1.SNAPSHOT]
    assert controller.qwen_model_id == MODEL_ID
    assert controller.qwen_revision == QWEN_REVISION
    assert "huggingface" not in sanitize_public_value(str(calls[0])).lower()