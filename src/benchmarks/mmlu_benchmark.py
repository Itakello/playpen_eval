from typing import Callable

import weave
from datasets import Dataset as HfDataset
from datasets import DatasetDict, load_dataset
from pydantic import BaseModel
from weave import Dataset as WeaveDataset

from ..utils.scoring_functions import exact_match
from .custom_benchmark import BenchmarkCategory, BenchmarkType, HfBenchmark


class MMLUBenchmark(HfBenchmark, BaseModel):
    id: str = "tasksource/mmlu"
    name: str = "MMLU"
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.REASONING
    description: str = "Benchmark testing the model's reasoning abilities"
    scoring_function: Callable | None = exact_match

    def _build_dataset(self) -> WeaveDataset:
        assert self.hf_dataset
        rows = self.hf_dataset.to_list()
        table_rows = weave.Table(rows)
        return WeaveDataset(name=self.name, rows=table_rows)

    def _load_test_dataset(self) -> HfDataset:
        dataset = load_dataset(self.id, "abstract_algebra")
        assert type(dataset) is DatasetDict
        test_set = dataset["test"]
        assert type(test_set) is HfDataset
        return test_set
