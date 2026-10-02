"""Read-only admission policy for future sensor-specific validation.

No function in this module reads imagery, builds model tensors, modifies a
grid, or authorizes a model merely because a file is locally present.
"""
from __future__ import annotations

from typing import Any, Mapping


AUTHORIZED_DEVELOPMENT = "AUTHORIZED_DEVELOPMENT"
INSPECTION_ONLY = "INSPECTION_ONLY"
RESTRICTED_EVALUATION = "RESTRICTED_EVALUATION"
UNKNOWN = "UNKNOWN"


def classify_authorization(metadata: Mapping[str, Any] | None) -> str:
    """Classify only explicit provenance and split evidence; absence is unknown."""
    if not isinstance(metadata, Mapping):
        return UNKNOWN
    status = str(metadata.get("authorization") or "").upper()
    if status not in {AUTHORIZED_DEVELOPMENT, INSPECTION_ONLY, RESTRICTED_EVALUATION}:
        return UNKNOWN
    if status == AUTHORIZED_DEVELOPMENT and str(metadata.get("split") or "").lower() not in {"train", "validation", "development"}:
        return UNKNOWN
    return status


def product_descriptor(sensor: str, inspection: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Preserve discovered metadata without inventing product/band semantics."""
    return {"sensor": sensor, "authorization": classify_authorization(metadata),
            "product_type": (metadata or {}).get("product_type"), "product_level": (metadata or {}).get("product_level"),
            "band_count": inspection.get("band_count"), "band_names": inspection.get("band_descriptions"),
            "dtype": inspection.get("dtypes"), "radiometric_range": {"minimum": inspection.get("finite_value_checks", {}).get("finite_min"), "maximum": inspection.get("finite_value_checks", {}).get("finite_max")},
            "crs": inspection.get("crs"), "resolution": inspection.get("resolution"), "bounds": inspection.get("bounds"),
            "nodata": inspection.get("nodata"), "metadata_sidecars": (metadata or {}).get("sidecars", []),
            "status": "DESCRIBED_NOT_MODEL_COMPATIBLE"}


def model_validation_gate(*, sensor: str, authorization: str, product_understood: bool,
                          transformation_defined: bool, missing_channels_fabricated: bool,
                          domain_mismatch_documented: bool, test_access: int) -> dict[str, Any]:
    """Decide readiness for a *future* bounded test; this does not execute one."""
    reasons: list[str] = []
    if authorization != AUTHORIZED_DEVELOPMENT: reasons.append("AUTHORIZED_DEVELOPMENT data is required.")
    if not product_understood: reasons.append("Product metadata is insufficient.")
    if not transformation_defined: reasons.append("A scientifically defined input transformation is required.")
    if missing_channels_fabricated: reasons.append("Missing bands or polarizations cannot be fabricated.")
    if not domain_mismatch_documented: reasons.append("Domain mismatch must be documented.")
    if test_access != 0: reasons.append("TEST/evaluation access must remain zero during development admission.")
    return {"sensor": sensor, "model_compatibility_test_authorized": not reasons,
            "status": "FUTURE_TEST_MAY_BE_PROPOSED" if not reasons else "NOT_AUTHORIZED",
            "reasons": reasons, "model_executed": False}
