import json
from dataclasses import dataclass

from weave.flow.dataset import Dataset as WeaveDataset

from .base_benchmark import BaseBenchmark, BenchmarkCategory, BenchmarkType
from ..models import BaseModel


@dataclass()
class FantomBenchmark(BaseBenchmark):
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.WORLD_KNOWLEDGE
    description: str = ""
    tasks = ["full", "short"]

    def _load_dataset(self) -> WeaveDataset:
        file_url = "https://drive.google.com/drive/folders/1LFHUS06Ir6IBWomeKuzqz1qjEUe3Fq88"
        dataset_folder_path = self._download_from_url(file_url, self.name)
        task = self._get_task_if_exists()

        dataset = []
        if task is None:
            for file_path in dataset_folder_path.glob('*.json'):  # Only looks for JSON files
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    print(f"Loaded JSON data from {file_path}:")
                    print(data)  # Process the JSON data as needed



    def evaluate(self, model: BaseModel) -> dict:
        pass

    def _build_dataset(self) -> WeaveDataset:
        pass