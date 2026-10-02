"""Explicit sensor declarations and fail-closed adapter compatibility gates.

The foundations in this module preserve a user's sensor declaration. They do
not infer a satellite platform, synthesize missing band names, resample an
image, or claim cross-sensor model generalization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


S2_CANONICAL_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
SENSOR_ALIASES = {
    "sentinel-2": "sentinel-2", "s2": "sentinel-2",
    "sentinel-1": "sentinel-1", "s1": "sentinel-1",
    "cartosat-2s": "cartosat-2s", "cartosat2s": "cartosat-2s",
    "risat": "risat",
    "generic-rgb": "generic-rgb", "generic-multispectral": "generic-multispectral", "generic-sar": "generic-sar",
}
VALID_MODALITIES = frozenset({"rgb", "optical", "multispectral", "sar"})
VALID_ROLES = frozenset({"SINGLE", "S1", "S2", "T1", "T2"})
VALID_ROUTES = frozenset({"SINGLE_IMAGE_VQA", "SINGLE_IMAGE_SAR_VQA", "SINGLE_IMAGE_SCENE_DESCRIPTION", "OPTICAL_SAR_ANALYSIS", "TEMPORAL_CHANGE_DESCRIPTION"})


class SensorDeclarationError(ValueError):
    """A user-safe declaration error with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SensorDeclaration:
    sensor: str
    modality: str
    role: str
    band_order: tuple[str, ...]
    band_order_confirmed: bool
    source: str = "user_declaration"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["band_order"] = list(self.band_order)
        return result


def parse_sensor_declaration(raw: Mapping[str, Any], inspection: Mapping[str, Any]) -> SensorDeclaration:
    """Validate user-declared sensor and role metadata without guessing."""
    if not isinstance(raw, Mapping):
        raise SensorDeclarationError("SENSOR_DECLARATION_REQUIRED", "A sensor, modality, and role declaration is required for an external raster.")
    provided_sensor = str(raw.get("sensor") or "").strip().lower()
    sensor = SENSOR_ALIASES.get(provided_sensor)
    if not sensor:
        raise SensorDeclarationError("SENSOR_DECLARATION_REQUIRED", "Declare sensor as sentinel-2, sentinel-1, cartosat-2s, risat, generic-rgb, generic-multispectral, or generic-sar. Sensor identity is never inferred from pixels.")
    modality = str(raw.get("modality") or "").strip().lower()
    if modality not in VALID_MODALITIES:
        raise SensorDeclarationError("INVALID_SENSOR_MODALITY", "Declare modality as rgb, optical, multispectral, or sar.")
    role = str(raw.get("role") or "").strip().upper()
    if role not in VALID_ROLES:
        raise SensorDeclarationError("INVALID_SENSOR_ROLE", "Declare role as SINGLE, S1, S2, T1, or T2.")
    order = raw.get("band_order")
    if order is None:
        descriptions = inspection.get("band_descriptions") or []
        # Generic/foundation declarations may preserve source positions without
        # attaching semantic band names. Known Sentinel contracts cannot.
        if provided_sensor in {"generic-rgb", "generic-multispectral", "generic-sar", "cartosat-2s", "cartosat2s", "risat"}:
            order = [item if str(item).strip() else f"SOURCE_BAND_{index + 1}" for index, item in enumerate(descriptions)]
        else:
            order = descriptions
    if not isinstance(order, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in order):
        raise SensorDeclarationError("INVALID_BAND_DECLARATION", "band_order must be an explicit list of non-empty declared band labels.")
    if len(order) != int(inspection.get("band_count", 0)):
        raise SensorDeclarationError("BAND_DECLARATION_MISMATCH", "The declared band_order length must equal the inspected band count.")
    if len(set(order)) != len(order):
        raise SensorDeclarationError("DUPLICATE_BAND_DECLARATION", "Declared band labels must be unique; no order will be inferred.")
    expected_modalities = {
        "sentinel-2": {"optical", "multispectral"}, "sentinel-1": {"sar"},
        "cartosat-2s": {"rgb", "optical", "multispectral"}, "risat": {"sar"},
        "generic-rgb": {"rgb"}, "generic-multispectral": {"optical", "multispectral"}, "generic-sar": {"sar"},
    }
    if modality not in expected_modalities[sensor]:
        raise SensorDeclarationError("SENSOR_MODALITY_MISMATCH", f"Declared sensor {sensor} is incompatible with declared modality {modality}.")
    return SensorDeclaration(sensor=sensor, modality=modality, role=role, band_order=tuple(order), band_order_confirmed=raw.get("band_order_confirmed") is True)


def adapter_foundation(declaration: SensorDeclaration) -> dict[str, Any]:
    """Return only the adapter contract; no pixel conversion occurs here."""
    common = {"sensor": declaration.sensor, "modality": declaration.modality, "role": declaration.role,
              "band_order": list(declaration.band_order), "band_order_confirmed": declaration.band_order_confirmed,
              "transforms_applied": [], "resampling_applied": False, "band_reordering_applied": False}
    if declaration.sensor == "cartosat-2s":
        return {**common, "adapter": "cartosat-2s-foundation", "status": "FOUNDATION_ONLY",
                "message": "Cartosat-2S metadata is preserved, but no Cartosat-2S model adapter or model validation is available."}
    if declaration.sensor == "risat":
        return {**common, "adapter": "risat-foundation", "status": "FOUNDATION_ONLY",
                "message": "RISAT metadata is preserved, but no RISAT model adapter or model validation is available."}
    if declaration.sensor == "sentinel-2":
        return {**common, "adapter": "sentinel-2-declared", "status": "DECLARED_COMPATIBILITY_CHECK_REQUIRED"}
    if declaration.sensor == "sentinel-1":
        return {**common, "adapter": "sentinel-1-declared", "status": "DECLARED_COMPATIBILITY_CHECK_REQUIRED"}
    return {**common, "adapter": declaration.sensor, "status": "DECLARED_COMPATIBILITY_CHECK_REQUIRED"}


def compatibility_gate(inspection: Mapping[str, Any], declaration: SensorDeclaration, requested_route: str | None = None) -> dict[str, Any]:
    """Gate model use from source facts and user declarations only."""
    route = str(requested_route or "").strip().upper() or None
    if route is not None and route not in VALID_ROUTES:
        raise SensorDeclarationError("UNKNOWN_TARGET_ROUTE", "requested_route is not a registered external-input compatibility target.")
    finite_summary = inspection.get("finite_value_checks", {})
    finite = bool(finite_summary.get("all_unmasked_values_finite"))
    if not finite:
        return {"status": "INCOMPATIBLE", "code": "NONFINITE_RASTER_VALUES", "eligible_for_agent": False,
                "requested_route": route, "message": "Raster contains non-finite unmasked values. No model input was constructed.", "warnings": []}
    if declaration.sensor in {"cartosat-2s", "risat"}:
        return {"status": "BLOCKED", "code": "SENSOR_DOMAIN_NOT_VALIDATED", "eligible_for_agent": False,
                "requested_route": route, "message": f"{declaration.sensor} is accepted for inspection and provenance only; no SatQuery specialist has validated compatibility for this sensor.",
                "warnings": ["No Cartosat-2S/RISAT-to-Sentinel equivalence is assumed.", "No resampling, band reordering, or model execution was performed."]}
    if declaration.sensor == "sentinel-2" and route == "SINGLE_IMAGE_VQA":
        exact_order = declaration.band_order == S2_CANONICAL_BANDS
        if not (declaration.role == "SINGLE" and declaration.modality in {"optical", "multispectral"} and inspection.get("band_count") == 12 and inspection.get("shape") == [120, 120] and inspection.get("crs") and exact_order and declaration.band_order_confirmed):
            return {"status": "INCOMPATIBLE", "code": "S2_VQA_CONTRACT_UNMET", "eligible_for_agent": False, "requested_route": route,
                    "message": "SINGLE_IMAGE_VQA requires an explicitly declared 12-band, 120x120, CRS-bearing Sentinel-2 raster in canonical B01..B12 order with confirmed order.",
                    "warnings": ["The gate did not reorder or resample any band."]}
        return {"status": "COMPATIBLE", "code": None, "eligible_for_agent": True, "requested_route": route,
                "message": "Declared raster meets the existing external Sentinel-2 VQA input contract.",
                "warnings": ["External Sentinel-2 input remains subject to the route's existing internal-validation limitation."]}
    if declaration.sensor == "sentinel-1" and route == "SINGLE_IMAGE_SAR_VQA":
        exact_order = declaration.band_order == ("VV", "VH")
        if not (declaration.role == "SINGLE" and declaration.modality == "sar" and inspection.get("band_count") == 2 and inspection.get("shape") == [120, 120] and inspection.get("crs") and exact_order and declaration.band_order_confirmed):
            return {"status": "INCOMPATIBLE", "code": "S1_SAR_VQA_CONTRACT_UNMET", "eligible_for_agent": False, "requested_route": route,
                    "message": "SINGLE_IMAGE_SAR_VQA requires an explicitly declared 2-band, 120x120, CRS-bearing Sentinel-1 VV/VH raster with confirmed order.",
                    "warnings": ["The gate did not reorder or resample any band."]}
        return {"status": "COMPATIBLE", "code": None, "eligible_for_agent": True, "requested_route": route,
                "message": "Declared raster meets the existing Sentinel-1 SAR VQA input contract.",
                "warnings": ["SAR-only semantic validation remains limited; no optical fallback is used."]}
    if declaration.sensor == "generic-rgb" and route == "SINGLE_IMAGE_SCENE_DESCRIPTION":
        if declaration.role == "SINGLE" and inspection.get("band_count") in {3, 4}:
            return {"status": "COMPATIBLE_WITH_LIMITATIONS", "code": None, "eligible_for_agent": True, "requested_route": route,
                "message": "Declared generic RGB raster is eligible only for the existing explicit-RGB scene-description path.",
                "warnings": ["Sensor identity is user-declared, not inferred from pixels.", "Scene-description benchmark performance for this external source is not established.", *(["All pixels are zero or constant; no value repair was applied."] if finite_summary.get("all_unmasked_values_constant") else [])]}
    return {"status": "FOUNDATION_ONLY", "code": "SPECIALIST_INPUT_CONTRACT_NOT_ESTABLISHED", "eligible_for_agent": False, "requested_route": route,
            "message": "The raster was inspected and its declaration preserved, but no approved specialist contract is established for this external sensor/role/route combination.",
            "warnings": ["No conversion, resampling, band reordering, or specialist execution was performed."]}


def generic_classification(inspection: Mapping[str, Any]) -> str:
    """Describe the file class without assigning a satellite identity."""
    if inspection.get("input_kind") == "RGB_IMAGE":
        return "GENERIC_RGB"
    if int(inspection.get("band_count") or 0) in {3, 4}:
        return "GENERIC_RGB_CANDIDATE"
    return "GENERIC_MULTISPECTRAL"


def _missing_declaration_result(inspection: Mapping[str, Any], requested_route: str | None) -> dict[str, Any]:
    route = str(requested_route or "").strip().upper() or None
    return {"status": "INSPECTED", "task": "GENERIC_RASTER_INSPECTION", "inspection": dict(inspection),
            "generic_classification": generic_classification(inspection), "sensor_declaration": None,
            "adapter": {"adapter": "none", "status": "DECLARATION_REQUIRED", "transforms_applied": [], "resampling_applied": False, "band_reordering_applied": False},
            "compatibility_gate": {"status": "BLOCKED", "code": "MISSING_SENSOR_DECLARATION", "eligible_for_agent": False,
                                   "requested_route": route, "message": "The file was inspected, but no sensor declaration was supplied. Sensor identity is never inferred from band count or pixels.", "warnings": []},
            "agent_handoff": {"eligible": False, "requested_route": route, "reason": "A sensor declaration is required before specialist routing."}}


def inspect_and_gate(path: str, declaration: Mapping[str, Any] | None, requested_route: str | None = None, *, inspector: Any) -> dict[str, Any]:
    """Compose inspection, declaration, adapter foundation, and compatibility gate."""
    inspection = inspector(path)
    if declaration is None:
        return _missing_declaration_result(inspection, requested_route)
    declared = parse_sensor_declaration(declaration, inspection)
    adapter = adapter_foundation(declared)
    gate = compatibility_gate(inspection, declared, requested_route)
    return {"status": "INSPECTED", "task": "GENERIC_RASTER_INSPECTION", "inspection": inspection,
            "generic_classification": generic_classification(inspection), "sensor_declaration": declared.to_dict(), "adapter": adapter, "compatibility_gate": gate,
            "agent_handoff": {"eligible": gate["eligible_for_agent"], "requested_route": gate["requested_route"], "reason": gate["message"]}}


def _spatial_pair_error(first: Mapping[str, Any], second: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Validate only explicit spatial facts. No image alignment is attempted."""
    first_crs, second_crs = first.get("crs"), second.get("crs")
    if not first_crs or not second_crs:
        return "CRS_REQUIRED", "Both geospatial pair inputs require valid CRS metadata."
    if first_crs != second_crs:
        return "PAIR_CRS_MISMATCH", "Pair inputs declare different CRS values."
    first_bounds, second_bounds = first.get("bounds"), second.get("bounds")
    if not isinstance(first_bounds, list) or not isinstance(second_bounds, list) or len(first_bounds) != 4 or len(second_bounds) != 4:
        return "CRS_REQUIRED", "Both geospatial pair inputs require valid bounds."
    overlap_x = max(first_bounds[0], second_bounds[0]) < min(first_bounds[2], second_bounds[2])
    overlap_y = max(first_bounds[1], second_bounds[1]) < min(first_bounds[3], second_bounds[3])
    if not (overlap_x and overlap_y):
        return "PAIR_NO_SPATIAL_OVERLAP", "Pair bounds do not overlap; spatial correspondence is not established."
    return None, None


def validate_external_pair(first: Mapping[str, Any], second: Mapping[str, Any], *, pair_kind: str, pair_id: str | None = None) -> dict[str, Any]:
    """Fail closed on externally declared pair metadata without adapting a model."""
    kind = str(pair_kind or "").upper()
    if kind not in {"OPTICAL_SAR", "TEMPORAL"}:
        raise SensorDeclarationError("INVALID_PAIR_KIND", "pair_kind must be OPTICAL_SAR or TEMPORAL.")
    one, two = first.get("sensor_declaration") or {}, second.get("sensor_declaration") or {}
    if not one or not two:
        return {"status": "BLOCKED", "code": "MISSING_SENSOR_DECLARATION", "eligible_for_agent": False, "pair_id": pair_id, "warnings": []}
    roles = {str(one.get("role") or "").upper(), str(two.get("role") or "").upper()}
    expected = {"S1", "S2"} if kind == "OPTICAL_SAR" else {"T1", "T2"}
    if roles != expected:
        return {"status": "BLOCKED", "code": "TEMPORAL_ORDER_REQUIRED" if kind == "TEMPORAL" else "SENSOR_METADATA_MISMATCH", "eligible_for_agent": False, "pair_id": pair_id, "warnings": []}
    first_inspection, second_inspection = first.get("inspection") or {}, second.get("inspection") or {}
    raster_pair = first_inspection.get("input_kind") == "RASTER" or second_inspection.get("input_kind") == "RASTER"
    if raster_pair:
        code, message = _spatial_pair_error(first_inspection, second_inspection)
        if code:
            return {"status": "BLOCKED", "code": code, "message": message, "eligible_for_agent": False, "pair_id": pair_id, "warnings": []}
    return {"status": "FOUNDATION_ONLY", "code": "COREGISTRATION_NOT_VERIFIED", "eligible_for_agent": False, "pair_id": pair_id,
            "message": "Pair metadata passed basic declaration checks, but SatQuery did not align, resample, or verify co-registration for external files.",
            "warnings": ["COREGISTRATION_NOT_VERIFIED", "No specialist execution was performed for this external pair."]}
