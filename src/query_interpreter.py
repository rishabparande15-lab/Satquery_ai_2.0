"""Query interpretation with an optional-provider boundary and safe local fallback."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import os
import re
from typing import Any


@dataclass(frozen=True)
class QueryPlan:
    task: str
    aoi: dict[str, Any] | None
    start_date: str | None
    end_date: str | None
    modalities: list[str]
    datasets: list[str]
    requires_temporal_pair: bool
    required_tools: list[str]
    expected_outputs: list[str]
    constraints: list[str]
    interpreter: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMProvider:
    """Provider contract. Network adapters are deliberately opt-in, never implicit."""
    name = "unconfigured"

    def interpret(self, query: str, context: dict[str, Any]) -> QueryPlan:  # pragma: no cover - integration boundary
        raise NotImplementedError


def configured_provider() -> LLMProvider | None:
    """Expose configuration state without reading or embedding a provider secret."""
    provider = os.getenv("SATQUERY_LLM_PROVIDER", "").strip().lower()
    if provider in {"ollama", "gemini", "nvidia_nim", "groq", "openrouter"}:
        # Remote calls require a deployed adapter and explicit credentials.  Returning
        # None preserves deterministic, auditable behavior in the local POC.
        return None
    return None


def _iso_dates(text: str) -> list[str]:
    found = re.findall(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    dates = [f"{int(y):04}-{int(m):02}-{int(d):02}" for y, m, d in found]
    if dates:
        return list(dict.fromkeys(dates))
    months = {name: number for number, name in enumerate(('january','february','march','april','may','june','july','august','september','october','november','december'), 1)}
    named = re.findall(r'\b(' + '|'.join(months) + r')\s+(20\d{2})\b', text.lower())
    if named:
        return [f'{year}-{months[month]:02}-01' for month, year in named]
    years = re.findall(r"\b(20\d{2})\b", text)
    return [f"{year}-01-01" for year in dict.fromkeys(years)]


def _coordinates(text: str) -> dict[str, Any] | None:
    match = re.search(r"(?:lat(?:itude)?\s*[:=]?\s*)?(-?\d{1,2}\.\d+)\s*[, ]\s*(?:lon(?:gitude)?\s*[:=]?\s*)?(-?\d{1,3}\.\d+)", text, re.I)
    if not match:
        return None
    return {"type": "point", "latitude": float(match.group(1)), "longitude": float(match.group(2))}


def interpret_query(query: str, *, aoi: dict[str, Any] | None = None, start_date: str | None = None,
                    end_date: str | None = None, analysis_type: str | None = None) -> QueryPlan:
    """Interpret a request deterministically when no explicitly deployed LLM is available."""
    text = query.lower()
    temporal = bool(re.search(r"\b(compare|change|before|after|increase|decrease|trend|temporal)\b", text))
    water = bool(re.search(r"\b(water|lake|river|flood|wetland)\b", text))
    sar = bool(re.search(r"\b(sar|radar|sentinel[- ]?1|vv|vh)\b", text))
    optical = bool(re.search(r"\b(optical|sentinel[- ]?2|ndvi|ndwi|spectral)\b", text))
    task = analysis_type or ("water_change_analysis" if water and temporal else "water_detection" if water else "change_detection" if temporal else "sar_analysis" if sar and not optical else "optical_analysis" if optical and not sar else "joint_optical_sar_analysis")
    temporal = temporal or task in {"water_change_analysis", "change_detection", "temporal_comparison"}
    modalities = ["optical", "sar"] if (sar or task in {"joint_optical_sar_analysis", "water_change_analysis", "change_detection"}) else ["optical"]
    if task == "optical_analysis":
        modalities = ["optical"]
    if task == "sar_analysis":
        modalities = ["sar"]
    dates = _iso_dates(query)
    start = start_date or (dates[0] if dates else None)
    end = end_date or (dates[-1] if len(dates) > 1 else None)
    if temporal and not end:
        end = start
    datasets = (["Sentinel-2", "Sentinel-1"] if set(modalities) == {"optical", "sar"} else ["Sentinel-1"] if modalities == ["sar"] else ["Sentinel-2"])
    tools = ["input_validation", "preprocessing", "feature_extraction"]
    if len(modalities) == 2:
        tools.append("hybrid_fusion")
    if water:
        tools += ["water_detection", "area_calculation"]
    if temporal:
        tools += ["temporal_comparison", "change_detection"]
    constraints = ["No supervised accuracy claim without a trained, evaluated model."]
    if temporal:
        constraints.append("Requires a spatially corresponding before-and-after image pair.")
    return QueryPlan(task, aoi or _coordinates(query), start, end, modalities, datasets, temporal, tools,
                     ["validation", "features", "evidence", "explanation"], constraints, "deterministic_local_parser")
