from pathlib import Path

from src.api import build_v1_route_request


def test_build_v1_route_request_uses_live_dataset_patch():
    patch_id = "S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11"
    payload = build_v1_route_request({
        "patch_id": patch_id,
        "task_type": "caption",
        "question": "Describe the scene.",
    })

    assert payload["route"] == "MULTIMODAL_S1_S2"
    assert payload["patch_id"] == patch_id
    assert payload["s1_patch_id"] == patch_id
    assert payload["s2_patch_id"] == patch_id
    assert payload["s1"].shape == (2, 120, 120)
    assert payload["s2"].shape == (12, 120, 120)
    assert payload["spatial_metadata"]["shape"] == [120, 120]
