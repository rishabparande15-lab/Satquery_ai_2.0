import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from src import api
from src.imagery_availability import normalize_aoi, registry_response, search_imagery, validate_search_request


def request_body(aoi=None, **overrides):
    body = {
        "aoi": aoi or {"type": "bbox", "bounds": [77.0, 12.0, 77.1, 12.1]},
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "requested_resolution_m": 10,
        "modality": "optical",
        "source": "sentinel-2",
    }
    body.update(overrides)
    return body


def test_aoi_model_preserves_geometry_and_metadata():
    result = normalize_aoi({"type": "bbox", "bounds": [77.0, 12.0, 77.1, 12.1]}, "rectangle")
    assert result["geometry"]["type"] == "Polygon"
    assert result["bbox"] == [77.0, 12.0, 77.1, 12.1]
    assert result["centroid"] == {"latitude": 12.05, "longitude": 77.05}
    assert result["crs"] == "EPSG:4326"
    assert result["source"] == "rectangle"
    assert result["area_km2"] > 0


def test_polygon_aoi_requires_closed_ring():
    polygon = {"type": "Polygon", "coordinates": [[[77, 12], [77.1, 12], [77.1, 12.1], [77, 12]]]} 
    assert normalize_aoi(polygon)["geometry"]["type"] == "Polygon"
    with pytest.raises(ValueError, match="closed"):
        normalize_aoi({"type": "Polygon", "coordinates": [[[77, 12], [77.1, 12], [77.1, 12.1], [77, 12.1]]]})


def test_search_request_validates_dates_resolution_and_modality():
    assert validate_search_request(request_body())["sources"] == ["sentinel-2"]
    with pytest.raises(ValueError, match="start_date"):
        validate_search_request(request_body(start_date="2025-02-01", end_date="2025-01-01"))
    with pytest.raises(ValueError, match="requested_resolution"):
        validate_search_request(request_body(requested_resolution_m=0))
    with pytest.raises(ValueError, match="modality"):
        validate_search_request(request_body(modality="thermal"))


def test_registry_distinguishes_verified_and_unavailable_integrations():
    sources = {item["source"]: item for item in registry_response()}
    assert sources["sentinel-2"]["native_resolution_m"] == 10.0
    assert sources["sentinel-1"]["modality"] == "sar"
    assert sources["landsat"]["integration_status"] == "planned"
    assert sources["cartosat-2s"]["integration_status"] == "unavailable"
    assert sources["risat"]["integration_status"] == "unavailable"


def test_unconfigured_provider_returns_controlled_result(monkeypatch):
    monkeypatch.delenv("GEE_PROJECT", raising=False)
    result = search_imagery(request_body())
    assert result["status"] == "requires_configuration"
    assert result["provider"] == "google-earth-engine"
    assert result["options"][0]["coverage_status"] == "not_checked"
    assert "GEE_PROJECT" in result["warnings"][0]


def test_availability_route_returns_json_without_touching_analysis(client=None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        body = json.dumps(request_body()).encode()
        connection.request("POST", "/api/availability/search", body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        assert response.status == 200
        assert payload["status"] in {"requires_configuration", "available", "unavailable"}
        assert "options" in payload
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
