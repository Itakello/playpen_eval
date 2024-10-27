from dataclasses import dataclass

from datasets import load_dataset
from evaluate import evaluator
from itakello_logging import ItakelloLogging
from tqdm import tqdm

from ..classes.score import Score
from ..config.config import N_SHOTS
from ..models.base_model import Model
from .base_benchmark import Benchmark, BenchmarkCategory, BenchmarkType

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class MmluBenchmark(Benchmark):
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.REASONING
    description: str = (
        "Assesses complex logical deduction, multi-step reasoning, and the application of knowledge in various contexts, using exact match scoring to measure performance on specific, often real-world-like tasks."
    )

    def __post_init__(self) -> None:
        self.dataset = load_dataset("cais/mmlu", "all")
        self.task_evaluator = evaluator("text-classification")

    def evaluate(self, model: Model) -> Score:
        results = {}

        test_set = self.dataset.items().get("test")
        task_results: dict = self.task_evaluator.compute(
            model_or_pipeline=model,
            data=test_set,
            metric="exact_match",
            strategy="bootstrap",
            n_resamples=100,
        )

        # Extract the score and confidence intervals
        task_accuracy = task_results["exact_match"]["score"]
        confidence_interval = task_results["exact_match"]["confidence_interval"]

        results[task] = {
            "accuracy": task_accuracy,
            "confidence_interval": confidence_interval,
        }
