from __future__ import annotations

import hashlib

import pytest

from src.acquisition import AcquisitionError, AcquisitionItem, AcquisitionPlan, acquire_plan


def _item(item_id: str, path: str, content: bytes) -> AcquisitionItem:
    return AcquisitionItem(item_id, f"memory://{item_id}", path, len(content), hashlib.sha256(content).hexdigest())


def test_manifest_determinism_and_duplicate_prevention():
    first = AcquisitionPlan("fixture", "v1", (_item("b", "b.bin", b"b"), _item("a", "a.bin", b"a")), {"ids": ["a", "b"]})
    second = AcquisitionPlan("fixture", "v1", (_item("a", "a.bin", b"a"), _item("b", "b.bin", b"b")), {"ids": ["a", "b"]})
    assert first.fingerprint == second.fingerprint
    with pytest.raises(ValueError, match="duplicate acquisition item_id"):
        AcquisitionPlan("fixture", "v1", (_item("a", "a.bin", b"a"), _item("a", "b.bin", b"b")), {})


def test_selective_download_resume_checksum_and_reuse(tmp_path):
    content = {"a": b"abcdefgh", "b": b"0123456789"}
    plan = AcquisitionPlan("fixture", "v1", tuple(_item(key, f"data/{key}.bin", value) for key, value in content.items()), {"requested": sorted(content)})
    calls = []

    def reader(url, start, length):
        key = url.rsplit("/", 1)[-1]
        calls.append((key, start, length))
        return content[key][start:start + length]

    partial = tmp_path / "data" / "a.bin.part"
    partial.parent.mkdir()
    partial.write_bytes(content["a"][:3])
    report = acquire_plan(plan, tmp_path, chunk_size=3, read_range=reader)
    assert report["status"] == "complete"
    assert (tmp_path / "data/a.bin").read_bytes() == content["a"]
    assert calls[0] == ("a", 3, 3)
    before = len(calls)
    acquire_plan(plan, tmp_path, chunk_size=3, read_range=reader)
    assert len(calls) == before


def test_failure_report_preserves_partial_and_does_not_claim_complete(tmp_path):
    item = _item("a", "a.bin", b"abcdefgh")
    plan = AcquisitionPlan("fixture", "v1", (item,), {"requested": ["a"]})

    def reader(url, start, length):
        return b"x" * length

    with pytest.raises(AcquisitionError) as error:
        acquire_plan(plan, tmp_path, chunk_size=4, attempts=1, read_range=reader)
    assert error.value.report["status"] == "failed"
    assert error.value.report["failures"][0]["item_id"] == "a"
    assert not (tmp_path / "a.bin").exists()
    assert (tmp_path / "a.bin.part").exists()


def test_invalid_plan_paths_fail_closed():
    with pytest.raises(ValueError, match="relative_path"):
        _item("a", "../a.bin", b"a")