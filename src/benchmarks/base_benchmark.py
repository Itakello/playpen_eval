import gdown
import os
import json
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable
from pathlib import Path


from datasets import Dataset as HfDataset, load_dataset
from itakello_logging import ItakelloLogging
from weave import Dataset as WeaveDataset

from ..classes.base_class import BaseClass
from ..models.base_model import BaseModel

logger = ItakelloLogging().get_logger(__name__)

class BenchmarkType(Enum):
    FUNCTIONAL = auto()
    FORMAL = auto()
    MIXED = auto()


class BenchmarkCategory(Enum):
    WORLD_KNOWLEDGE = auto()
    MISCELLANEOUS = auto()
    REASONING = auto()


@dataclass
class BaseBenchmark(BaseClass, ABC):
    name: str
    type: BenchmarkType
    category: BenchmarkCategory
    description: str
    weave_dataset: WeaveDataset = field(init=False)
    scoring_function: Callable = field(init=False)
    available_tasks: list[str] = field(init=False)
    task: str = field(default=None, init=False)

    @abstractmethod
    def evaluate(self, model: BaseModel) -> dict:
        """Abstract method to evaluate a model."""
        pass

    @abstractmethod
    def _build_dataset(self) -> WeaveDataset:
        """Abstract method to build a dataset."""
        pass


@dataclass
class HfBenchmark(BaseBenchmark, ABC):
    id: str
    hf_dataset: HfDataset = field(init=False)


    def __post_init__(self) -> None:
        self.hf_dataset = load_dataset(self.id)

@dataclass
class CloudBenchmark(BaseBenchmark, ABC):
    url: str

    def _download_from_url(self, url, dataset_name, output_folder=None):
        output_folder = os.path.join(Path(__file__).resolve().parent.parent.parent, "datasets", dataset_name) if output_folder is None else output_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            gdown.download_folder(url, output=output_folder, quiet=False)

            dowloaded_file = os.path.join(output_folder, f"{dataset_name}.zip")
            with zipfile.ZipFile(dowloaded_file, 'r') as zip_ref:
                zip_ref.extractall(output_folder)

            os.remove(dowloaded_file)

        return Path(os.path.join(output_folder))

    def load_dataset_from_drive(self, url, dataset_name, task=None) -> list:
        if task and task not in self.available_tasks:
            raise ValueError(f"Task '{task}' not available")

        dataset_folder_path = self._download_from_url(url, dataset_name)

        if task is None:
            return [
                json.load(open(file_path, 'r'))
                for file_path in dataset_folder_path.glob('*.json')
            ]

        return [
            json.load(open(dataset_folder_path / f, 'r'))
            for f in os.listdir(dataset_folder_path)
            if f.startswith(task) and f.endswith('.json')
        ]
