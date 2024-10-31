from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

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

    def _get_task_if_exists(self):
        task = None
        if self.task is not None:
            if self.task in self.available_tasks:
                task = self.task
            else:
                raise ValueError(f"{self.task} not in available tasks")
        return task

    @abstractmethod
    def evaluate(self, model: BaseModel) -> dict:
        """Abstract method to evaluate a model."""
        pass

    @abstractmethod
    def _load_dataset(self) -> WeaveDataset:
        """Abstract method to build a dataset."""
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
