from src.analysis_engine import run_analysis
from src.multimodal_cube import MultimodalCube
from src.query_interpreter import interpret_query
from src.tool_selector import select_tools


def test_query_interpreter_extracts_temporal_water_request():
    plan = interpret_query("Compare water from January 2020 to January 2025 at 12.3, 77.6")
    assert plan.task == "water_change_analysis"
    assert plan.requires_temporal_pair is True
    assert plan.start_date == "2020-01-01"
    assert plan.end_date == "2025-01-01"
    assert plan.aoi["type"] == "point"


def test_tool_selection_never_marks_pending_head_available():
    selection = select_tools(interpret_query("detect water").to_dict(), {"croma": False})
    assert any(item["tool"] == "water_detection" and item["status"] == "pending training" for item in selection.unavailable_tools)


def test_temporal_request_without_pair_is_rejected_honestly():
    result = run_analysis({"query": "Compare water in 2020 and 2025", "sample_id": "not-a-real-sample"})
    assert result["validation"]["valid"] is False
    assert "Temporal analysis requires a valid spatially corresponding before-and-after image pair." in result["validation"]["errors"]


def test_data_cube_rejects_misaligned_modalities():
    import numpy as np
    cube = MultimodalCube(np.zeros((12, 4, 4)), np.zeros((2, 5, 4)))
    try:
        cube.validate()
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("misaligned cube was accepted")


def test_api_starts_and_serves_frontend_and_health():
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen

    from src.api import Handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/health") as response:
            assert response.status == 200
            assert b"SatQuery AI" in response.read()
        with urlopen(f"http://127.0.0.1:{server.server_port}/") as response:
            assert b"Ask the Earth" in response.read()
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=2)
