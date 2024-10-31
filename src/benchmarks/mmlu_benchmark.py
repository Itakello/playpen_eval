from typing import Callable

from datasets import Dataset as HfDataset
from datasets import DatasetDict, load_dataset
from pydantic import BaseModel

from ..utils.scoring_functions import exact_match
from .custom_benchmark import BenchmarkCategory, BenchmarkType, HfBenchmark


class MMLUBenchmark(HfBenchmark, BaseModel):
    id: str = "tasksource/mmlu"
    name: str = "MMLU"
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.REASONING
    description: str = "Benchmark testing the model's reasoning abilities"
    scoring_function: Callable | None = exact_match

    def _load_test_dataset(self) -> list[dict]:
        dataset = load_dataset(self.id, "abstract_algebra")
        assert type(dataset) is DatasetDict
        test_set = dataset["test"].to_list()
        for entry in test_set:
            choices = "\n".join(
                [f"{idx+1} {choice}" for idx, choice in enumerate(entry["choices"])]
            )
            entry["question"] = entry["question"] + choices + "\n"
            del entry["choices"]
        return test_set
