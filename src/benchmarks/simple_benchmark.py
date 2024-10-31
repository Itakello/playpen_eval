from dataclasses import dataclass

from ..classes.score import Score
from ..models.base_model import Model
from .base_benchmark import BaseBenchmark, BenchmarkCategory, BenchmarkType


@dataclass
class SimpleBenchmark(BaseBenchmark):
    name: str = "simple"
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.REASONING
    description: str = "A simple benchmark testing basic reasoning capabilities"

    def __post_init__(self) -> None:
        # Simple test cases that should be easy for any LLM
        self.test_cases = [
            {"prompt": "What is 2+2? Answer with just the number.", "expected": "4"},
            {
                "prompt": "Is the sky blue? Answer with just yes or no.",
                "expected": "yes",
            },
            {"prompt": "Complete the sequence: 1, 2, 3, ...", "expected": "4"},
        ]

    def evaluate(self, model: Model) -> Score:
        correct = 0
        per_task_scores = {}

        for i, test_case in enumerate(self.test_cases):
            response = model.generate(test_case["prompt"]).strip().lower()
            is_correct = test_case["expected"] in response.lower()
            score = 1.0 if is_correct else 0.0
            correct += score

            task_name = f"Task_{i+1}"
            per_task_scores[task_name] = score

        overall_score = correct / len(self.test_cases)

        return Score(overall=overall_score, per_task=per_task_scores)
