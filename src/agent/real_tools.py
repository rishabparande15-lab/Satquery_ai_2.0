"""Adapters that let the controller call already-validated components.

They are intentionally lazy: importing this module does not load a checkpoint.
All payloads are supplied by the caller and no blocked capability is represented.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .controller import Tool, ToolResult
from .evidence import ConfidenceSource, Evidence, EvidenceType
from .query_types import AnalysisRequest


def _one_payload(request: AnalysisRequest) -> Mapping[str, Any]:
    if len(request.inputs) != 1:
        raise ValueError("this real tool requires exactly one input payload")
    return request.inputs[0]


class QwenRGBTool(Tool):
    """Real Qwen RGB execution, loaded only for an explicit controller call."""

    name = "qwen_rgb"

    def __init__(self, snapshot: str | Path, *, dtype: str = "float16", device: str = "cuda", max_new_tokens: int = 32) -> None:
        self.snapshot, self.dtype, self.device, self.max_new_tokens = Path(snapshot), dtype, device, max_new_tokens
        self._adapter = None
        self._runtime: Mapping[str, Any] | None = None

    def _load(self) -> tuple[Any, Mapping[str, Any]]:
        if self._adapter is None:
            from src.eo_vlm_adapter import Qwen25VLRGBAdapter
            self._adapter = Qwen25VLRGBAdapter()
            self._runtime = self._adapter.load_model(self.snapshot, dtype=self.dtype, device=self.device)
        assert self._runtime is not None
        return self._adapter, self._runtime

    def execute(self, request: AnalysisRequest, *, representations: tuple[str, ...]) -> ToolResult:
        payload = _one_payload(request)
        image = payload.get("array")
        if not isinstance(image, np.ndarray):
            raise ValueError("RGB input payload requires an ndarray at 'array'")
        adapter, runtime = self._load()
        prepared = adapter.prepare_inputs(__import__("src.eo_vlm_adapter", fromlist=["EOInput"]).EOInput("OPTICAL_RGB", optical=image))
        result = adapter.generate(runtime, prepared, request.query, max_new_tokens=self.max_new_tokens)
        provenance = {**result["provenance"], "model_id": adapter.model_id, "adapter_version": adapter.adapter_id,
                      "generation_configuration": {"max_new_tokens": self.max_new_tokens}, "latency_seconds": result["generation_seconds"]}
        item = Evidence(f"rgb:{payload['id']}", EvidenceType.MODEL_LANGUAGE_OUTPUT, self.name, result["text"], "language_response",
                        modality="rgb", representation="RGB", confidence_source=ConfidenceSource.UNKNOWN, provenance=provenance)
        return ToolResult({"text": result["text"], "generated_tokens": result["generated_tokens"]}, (item,), provenance)


class TemporalRGBTool(Tool):
    """Real validated pair execution; it emits language, never temporal evidence."""

    name = "temporal_rgb"

    def __init__(self, qwen_tool: QwenRGBTool) -> None:
        self._qwen_tool = qwen_tool

    def execute(self, request: AnalysisRequest, *, representations: tuple[str, ...]) -> ToolResult:
        payload = _one_payload(request)
        temporal_input, t1, t2 = payload.get("temporal_input"), payload.get("t1"), payload.get("t2")
        if not isinstance(t1, np.ndarray) or not isinstance(t2, np.ndarray):
            raise ValueError("temporal RGB payload requires ndarray t1 and t2")
        from src.eo_vlm.temporal_rgb_adapter import TemporalRGBAdapter
        adapter = TemporalRGBAdapter()
        prepared = adapter.prepare_temporal_rgb(temporal_input, t1, t2)
        _, runtime = self._qwen_tool._load()
        result = adapter.generate(runtime, prepared, max_new_tokens=self._qwen_tool.max_new_tokens)
        provenance = {**prepared.provenance, **result["provenance"], "model_id": self._qwen_tool._adapter.model_id,
                      "generation_configuration": {"max_new_tokens": self._qwen_tool.max_new_tokens}, "latency_seconds": result["generation_seconds"]}
        item = Evidence(f"temporal-rgb:{payload['id']}", EvidenceType.MODEL_LANGUAGE_OUTPUT, self.name, result["text"], "language_response",
                        modality="optical_rgb_pair", representation="RGB_T1_RGB_T2", confidence_source=ConfidenceSource.UNKNOWN, provenance=provenance)
        return ToolResult({"text": result["text"], "generated_tokens": result["generated_tokens"]}, (item,), provenance)


class TemporalChangeDescriptionTool(Tool):
    """Adapter for the admitted Chg2Cap PRE/POST caption specialist only."""

    name = "chg2cap"

    def __init__(self, controller) -> None:
        self._controller = controller

    def execute(self, request: AnalysisRequest, *, representations: tuple[str, ...]) -> ToolResult:
        if len(request.inputs) != 2:
            raise ValueError("Chg2Cap requires exactly two ordered image inputs")
        t1, t2 = request.inputs
        result = self._controller.run(
            t1_path=t1.get("path"), t2_path=t2.get("path"), query=request.query,
            metadata={**request.metadata, "temporal_order": "PRE_POST"},
        )
        provenance = dict(result["provenance"])
        item = Evidence(
            f"temporal-change:{t1['id']}:{t2['id']}", EvidenceType.MODEL_LANGUAGE_OUTPUT,
            self.name, result["change_description"], "change_description",
            modality="optical_rgb_pair", representation="RGB_T1_RGB_T2",
            confidence_source=ConfidenceSource.UNKNOWN,
            provenance=provenance,
        )
        return ToolResult(result, (item,), provenance)


class FrozenScientificCacheTool(Tool):
    """Execute the immutable 830-D probe over an existing checksum-validated cache."""

    name = "scientific_predictor"

    def __init__(self, cache_root: str | Path, checkpoint: str | Path) -> None:
        self.cache_root, self.checkpoint = Path(cache_root), Path(checkpoint)

    def execute(self, request: AnalysisRequest, *, representations: tuple[str, ...]) -> ToolResult:
        payload = _one_payload(request)
        sample_id = str(payload["id"])
        from scripts.run_pipeline3_5000_baseline import _load_probe, load_cache
        import torch
        started = perf_counter()
        cache = load_cache(self.cache_root, include_test=True)
        indices = np.flatnonzero(cache["area_ids"] == sample_id)
        if len(indices) != 1:
            raise ValueError("scientific cache requires one exact authoritative area id")
        model = _load_probe(self.checkpoint, 830)
        hybrid = np.concatenate((cache["physical"][indices[0]], cache["joint_croma"][indices[0]])).astype(np.float32)
        with torch.inference_mode():
            prediction = model.predict(torch.from_numpy(hybrid).unsqueeze(0))[0].tolist()
        provenance = {"input_id": sample_id, "representation": "hybrid_830d", "feature_order": "physical_62d + joint_croma_gap_768d",
                      "latency_seconds": perf_counter() - started, "dataset_fingerprint": "7dfd5cd5077e7fd0307acd3fb442d4745aa829a03611c7522e08acbc7f027625",
                      "split_fingerprint": "2232ac5bc65d3ed20c6f39deb6037bb8e0543100b247fd6c9e538eddf9feb86c"}
        item = Evidence(f"scientific:{sample_id}", EvidenceType.SCIENTIFIC_PREDICTION, self.name, prediction, "scene_prediction",
                        modality="optical_sar", representation="hybrid_830d", confidence_source=ConfidenceSource.UNKNOWN, provenance=provenance)
        return ToolResult({"prediction": prediction}, (item,), provenance)


class S2ProjectorTool(Tool):
    """Run the real learned S2 projector only; it does not fabricate text generation."""

    name = "qwen_s2"

    def __init__(self, checkpoint: str | Path, receipt: str | Path) -> None:
        self.checkpoint, self.receipt = Path(checkpoint), Path(receipt)

    def execute(self, request: AnalysisRequest, *, representations: tuple[str, ...]) -> ToolResult:
        payload = _one_payload(request)
        cube = payload.get("array")
        if not isinstance(cube, np.ndarray) or cube.shape != (12, 120, 120):
            raise ValueError("S2 input payload requires finite [12,120,120] ndarray")
        if not np.isfinite(cube).all():
            raise ValueError("S2 input contains non-finite values")
        import json, torch
        from src.eo_vlm.multispectral_projector import S2MultispectralProjector
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        model = S2MultispectralProjector(); model.load_state_dict(torch.load(self.checkpoint, map_location="cpu", weights_only=True)["state_dict"]); model.eval()
        started = perf_counter()
        with torch.inference_mode(): tokens = model(torch.from_numpy(cube.astype(np.float32)).unsqueeze(0))
        provenance = {**receipt["provenance"], "adapter_checkpoint_sha256": receipt["checkpoint_sha256"], "latency_seconds": perf_counter() - started}
        item = Evidence(f"s2-projector:{payload['id']}", EvidenceType.METADATA_EVIDENCE, self.name,
                        "learned S2 visual tokens produced; language generation not invoked by this projector-only tool", "s2_visual_token_execution",
                        modality="s2", representation="learned_s2_representation", confidence_source=ConfidenceSource.UNKNOWN, provenance=provenance)
        return ToolResult({"token_shape": list(tokens.shape)}, (item,), provenance)
