from dataclasses import dataclass

from .base_benchmark import HuggingfaceBenchmark, BenchmarkCategory, BenchmarkType
from ..classes.score import Score
from ..models.base_model import Model

@dataclass
class EwokBenchmark(HuggingfaceBenchmark):
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.WORLD_KNOWLEDGE
    description: str = (
        ""
    )
    id: str = "ewok-core/ewok-core-1.0"

    def evaluate(self, model: Model) -> Score:
        pass