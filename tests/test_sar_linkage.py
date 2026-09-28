import pytest

from src.eo_vlm.sar_linkage import require_exact_s1_link, sar_annotation_applicability


def test_exact_dual_identity_is_required():
    annotation = {"image_id": "S2_A", "sar_identity": "S1_A", "task_type": "binary_qa"}
    require_exact_s1_link(annotation, {"patch_id": "S2_A", "s1_name": "S1_A"})
    with pytest.raises(ValueError, match="IMAGE_LINKAGE_BLOCKED"):
        require_exact_s1_link(annotation, {"patch_id": "S2_A", "s1_name": "S1_other"})


def test_no_optical_to_sar_label_transfer_is_assumed():
    assert sar_annotation_applicability({"task_type": "multiple_choice_qa"})[0] == "SAR_UNKNOWN"
    with pytest.raises(ValueError):
        sar_annotation_applicability({"task_type": "caption"})
