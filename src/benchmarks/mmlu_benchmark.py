from dataclasses import dataclass

from ..models.base_model import Model
from .base_benchmark import BenchmarkCategory, BenchmarkType
from .lighteval_benchmark import LightEvalBenchmark


@dataclass
class MMLUBenchmark(LightEvalBenchmark):
    tasks: str = "helm|mmlu:high_school_geography|5|0"
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.REASONING
    description: str = "A benchmark testing the model's ability to reason"

    def evaluate(self, model: Model) -> dict:
        results = super().evaluate(model)
        # Add any custom post-processing here
        return results
