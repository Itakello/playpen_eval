from dataclasses import dataclass

from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask

from ..classes.score import Score
from ..models.base_model import Model
from .base_benchmark import Benchmark, BenchmarkCategory, BenchmarkType


@dataclass
class BbhBenchmark(Benchmark):
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.REASONING
    description: str = (
        "Assesses complex logical deduction, multi-step reasoning, and the application of knowledge in various contexts, using exact match scoring to measure performance on specific, often real-world-like tasks."
    )
    benchmark: MMLU = MMLU(n_shots=5, tasks=[MMLUTask.ASTRONOMY])

    def evaluate(self, model: Model) -> Score:
        score = self.benchmark.evaluate(model)
        return Score(score.overall, score.per_task)
