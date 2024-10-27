from dataclasses import dataclass

import evaluate
from datasets import load_dataset
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
        self.metric = evaluate.load("exact_match")

    def evaluate(self, model: Model) -> Score:
        results = {}
        total_correct = 0
        total_samples = 0

        for task, task_dataset in tqdm(
            self.dataset.items(), desc="Evaluating MMLU tasks"
        ):
            task_correct = 0
            task_total = 0

            for sample in tqdm(task_dataset["test"], desc=f"Task: {task}", leave=False):
                context = sample["context"] if "context" in sample else ""
                question = sample["question"]
                choices = sample["choices"]
                correct_answer = sample["answer"]

                # Generate the model's prediction
                prediction = model.generate(
                    context=context, question=question, choices=choices
                )

                # Compute exact match
                is_correct = self.metric.compute(
                    predictions=[prediction], references=[correct_answer]
                )["exact_match"]

                task_correct += is_correct
                task_total += 1

            task_accuracy = task_correct / task_total
            results[task] = task_accuracy
            total_correct += task_correct
            total_samples += task_total

        overall_accuracy = total_correct / total_samples

        return Score(overall=overall_accuracy, per_task=results)
