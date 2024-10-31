from dataclasses import dataclass

from ..classes.score import Score
from ..models.base_model import Model
from .base_benchmark import BaseBenchmark, BenchmarkCategory, BenchmarkType


@dataclass
class EwokBenchmark(BaseBenchmark):
    id: str = "ewok-core/ewok-core-1.0"
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.WORLD_KNOWLEDGE
    description: str = "TEMP"

    def evaluate(self, model: Model) -> Score:
        pass
