import os
from pathlib import Path
import tempfile

import pytest

from src import upload_staging


TTL = 900


def stage(root: Path, name="request-abcdefgh", *, age=TTL + 60, content="image.tif") -> Path:
    path = root / name
    path.mkdir()
    if content:
        (path / content).write_bytes(b"synthetic")
    old = path.stat().st_mtime - age
    os.utime(path, (old, old))
    return path


def test_stale_recognized_directory_is_deleted(tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    stale = stage(root)

    result = upload_staging.recover_staging(root, ttl_seconds=TTL)

    assert result[0]["classification"] == upload_staging.STALE_SAFE_TO_DELETE
    assert result[0]["deleted"] is True
    assert not stale.exists()


def test_recent_recognized_directory_is_preserved(tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    recent = stage(root, age=TTL - 10)

    result = upload_staging.recover_staging(root, ttl_seconds=TTL)

    assert result[0]["classification"] == upload_staging.RECENT
    assert result[0]["deleted"] is False
    assert recent.exists()


def test_active_directory_is_preserved_even_when_expired(tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    active = stage(root)

    result = upload_staging.recover_staging(root, ttl_seconds=TTL, active_paths=[active])

    assert result[0]["classification"] == upload_staging.ACTIVE
    assert result[0]["deleted"] is False
    assert active.exists()


def test_active_path_generator_protects_every_owned_directory(tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    active = [stage(root, name="request-abcdefgh"), stage(root, name="request-ijklmnop")]

    results = upload_staging.recover_staging(root, ttl_seconds=TTL, active_paths=(path for path in active))

    assert [item["classification"] for item in results] == [upload_staging.ACTIVE, upload_staging.ACTIVE]
    assert all(path.exists() for path in active)


def test_unknown_name_or_contents_are_preserved(tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    unknown_name = stage(root, name="other-abcdefgh")
    unknown_contents = stage(root, name="request-ijklmnop", content="unexpected.bin")

    results = upload_staging.recover_staging(root, ttl_seconds=TTL)

    assert all(item["classification"] == upload_staging.UNKNOWN for item in results)
    assert unknown_name.exists() and unknown_contents.exists()


def test_outside_root_and_traversal_are_rejected(tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    outside = stage(tmp_path, name="request-abcdefgh")

    result = upload_staging.classify_staging_directory(root, outside, ttl_seconds=TTL)
    traversal = upload_staging.classify_staging_directory(root, root / ".." / outside.name, ttl_seconds=TTL)

    assert result["classification"] == upload_staging.UNKNOWN
    assert traversal["classification"] == upload_staging.UNKNOWN
    assert outside.exists()


def test_symlink_escape_is_rejected_when_supported(tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "request-abcdefgh"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory symlink creation is unavailable: {error}")

    result = upload_staging.recover_staging(root, ttl_seconds=TTL)

    assert result[0]["classification"] == upload_staging.UNKNOWN
    assert result[0]["deleted"] is False
    assert outside.exists()


def test_staging_root_symlink_is_never_resolved_or_scanned(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    stale = stage(outside)
    root_link = tmp_path / "staging-link"
    try:
        root_link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory symlink creation is unavailable: {error}")
    assert upload_staging.recover_staging(root_link, ttl_seconds=TTL) == []
    assert stale.exists()


def test_reparse_escape_is_rejected_even_when_windows_cannot_create_symlinks(tmp_path, monkeypatch):
    root = tmp_path / "staging"
    root.mkdir()
    candidate = stage(root)
    original = upload_staging._is_reparse
    monkeypatch.setattr(upload_staging, "_is_reparse", lambda path: Path(path) == candidate or original(path))

    result = upload_staging.recover_staging(root, ttl_seconds=TTL)

    assert result[0]["classification"] == upload_staging.UNKNOWN
    assert result[0]["deleted"] is False
    assert candidate.exists()


def test_cleanup_permission_failure_is_nonfatal_and_preserves_directory(tmp_path, monkeypatch):
    root = tmp_path / "staging"
    root.mkdir()
    stale = stage(root)

    def denied(_path):
        raise PermissionError("synthetic cleanup denial")

    monkeypatch.setattr(upload_staging.shutil, "rmtree", denied)
    result = upload_staging.recover_staging(root, ttl_seconds=TTL)

    assert result[0]["classification"] == upload_staging.STALE_SAFE_TO_DELETE
    assert result[0]["deleted"] is False
    assert "preserved" in result[0]["reason"]
    assert stale.exists()


def test_health_and_readiness_remain_available_when_recovery_cleanup_is_denied(tmp_path, monkeypatch):
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen

    from src import api

    root = tmp_path / "staging"
    root.mkdir()
    stale = stage(root)
    monkeypatch.setattr(upload_staging.shutil, "rmtree", lambda _path: (_ for _ in ()).throw(PermissionError()))
    recovery = upload_staging.recover_staging(root, ttl_seconds=TTL)
    monkeypatch.setattr(api, "run_preflight", lambda: {
        "status": "READY", "models": {"scene_model": {"exists": True}, "chg2cap": {"exists": True}},
        "warnings": [], "errors": [],
    })
    server = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/v1/health") as response:
            health = json.loads(response.read())
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/v1/ready") as response:
            ready = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)

    assert recovery[0]["deleted"] is False
    assert stale.exists()
    assert health["status"] == "ok"
    assert ready["status"] == "READY_WITH_LIMITATIONS"
    assert ready["routes"]["SINGLE_IMAGE_GROUNDING"] == "BLOCKED"


def test_recovery_only_scans_immediate_children(tmp_path):
    root = tmp_path / "staging"
    nested = root / "nested"
    nested.mkdir(parents=True)
    stale = stage(nested)

    result = upload_staging.recover_staging(root, ttl_seconds=TTL)
    assert len(result) == 1
    assert result[0]["classification"] == upload_staging.UNKNOWN
    assert result[0]["deleted"] is False
    assert stale.exists()


def test_normal_upload_token_cleanup_still_removes_temporary_directory(tmp_path, monkeypatch):
    from src import api

    root = tmp_path / "staging"
    root.mkdir()
    temporary = tempfile.TemporaryDirectory(prefix="request-", dir=root)
    Path(temporary.name, "image.tif").write_bytes(b"synthetic")
    monkeypatch.setattr(api, "UPLOADS", {"synthetic-token": {"temporary": temporary}})

    api.cleanup_uploads(["synthetic-token"])

    assert not Path(temporary.name).exists()
    assert api.UPLOADS == {}
