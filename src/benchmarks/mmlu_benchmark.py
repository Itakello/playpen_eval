from dataclasses import dataclass

from datasets import load_dataset
from itakello_logging import ItakelloLogging
from weave import Dataset as WeaveDataset

from ..models.base_model import BaseModel
from ..scoring_functions import exact_match
from .base_benchmark import BenchmarkCategory, BenchmarkType, HfBenchmark

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class MMLUBenchmark(HfBenchmark):
    id: str = "tasksource/mmlu"
    name: str = "MMLU"
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.REASONING
    description: str = "Benchmark testing the model's reasoning abilities"
    scoring_function = exact_match

    def _build_dataset(self) -> WeaveDataset:
        raise NotImplementedError

    def evaluate(self, model: BaseModel) -> dict:
        return super().evaluate(model)

    def _load_dataset(self) -> dict:
        dataset = load_dataset(self.id, "abstract_algebra")
        return dataset["test"]
