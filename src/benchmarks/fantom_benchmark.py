import json
from dataclasses import dataclass, field

from weave.flow.dataset import Dataset as WeaveDataset

from .base_benchmark import BaseBenchmark, BenchmarkCategory, BenchmarkType, CloudBenchmark
from ..models import BaseModel


@dataclass()
class FantomBenchmark(CloudBenchmark):
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.WORLD_KNOWLEDGE
    description: str = ""
    tasks: list = field(default_factory=lambda: ["full", "short"])
    url: str = "https://drive.google.com/drive/folders/1LFHUS06Ir6IBWomeKuzqz1qjEUe3Fq88"

    def evaluate(self, model: BaseModel) -> dict:
        pass

    def _build_dataset(self) -> WeaveDataset:
        # a list containing all dataset files if a task is not specified, otherwise only those associated with a task.
        raw_dataset = self.load_dataset_from_drive(self.url, self.name, self.task)
        print(raw_dataset)