from pathlib import Path
import json

from experiments.pipelines.gee_temporal.data_access import retrieve


def load_pipeline2_outputs(project_root: Path) -> tuple[dict, dict[str, dict[str, Path]]]:
	report_path = project_root / "experiments" / "outputs" / "real_gee" / "gee_temporal" / "execution_report.json"
	report = json.loads(report_path.read_text(encoding="utf-8"))
	if report.get("data_source") != "Google Earth Engine" or not report.get("authentication_successful") or report.get("fallback_used"):
		raise RuntimeError("Pipeline 2 report is not a successful real-GEE run")
	paths = {}
	for output in report.get("outputs", []):
		if "archive" not in output:
			continue
		label = Path(output["archive"]).stem
		paths[label] = {Path(path).name.split(".")[-2]: Path(path) for path in output["rasters"]}
	required = {"before_sentinel_1", "after_sentinel_1", "before_sentinel_2", "after_sentinel_2"}
	if set(paths) != required:
		raise RuntimeError(f"Pipeline 2 output labels incomplete: {sorted(paths)}")
	return report, paths