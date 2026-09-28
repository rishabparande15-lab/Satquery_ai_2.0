"""Model-independent future evaluator interface; implementations must be versioned."""
from abc import ABC, abstractmethod
from typing import Iterable, Mapping

from .contracts import BenchmarkPrediction, MetricResult


class BenchmarkEvaluator(ABC):
    evaluator_id: str
    evaluator_version: str

    @abstractmethod
    def validate_predictions(self, predictions: Iterable[BenchmarkPrediction]) -> None: ...

    @abstractmethod
    def validate_targets(self, predictions: Iterable[BenchmarkPrediction]) -> None: ...

    @abstractmethod
    def compute_metrics(self, predictions: Iterable[BenchmarkPrediction]) -> tuple[MetricResult, ...]: ...

    @abstractmethod
    def summarize(self, metrics: Iterable[MetricResult]) -> Mapping[str, object]: ...

    @abstractmethod
    def get_provenance(self) -> Mapping[str, object]: ...

    def evaluate(self, predictions: Iterable[BenchmarkPrediction]) -> tuple[MetricResult, ...]:
        records = tuple(predictions)
        self.validate_predictions(records)
        self.validate_targets(records)
        return self.compute_metrics(records)
