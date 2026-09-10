import importlib.util


def authenticate(project: str | None) -> dict:
    status = {"package_available": False, "authentication_successful": False, "project": project, "error": None}
    if not project:
        status["error"] = "GEE_PROJECT is not set; Earth Engine API 1.7.43 requires an explicit Cloud project in this environment"
        return status
    if importlib.util.find_spec("ee") is None:
        status["error"] = "earthengine-api is not installed"
        return status
    import ee
    status["package_available"] = True
    try:
        ee.Initialize(project=project)
        status["authentication_successful"] = True
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
    return status


def require_authentication(project: str | None) -> dict:
    status = authenticate(project)
    if not status["authentication_successful"]:
        raise RuntimeError(status["error"] or "Earth Engine authentication failed")
    return status