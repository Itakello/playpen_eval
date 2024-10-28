from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

from datasets import load_dataset

from ..classes.base_class import BaseClass
from ..classes.score import Score
from ..models.base_model import Model


class BenchmarkType(Enum):
    FUNCTIONAL = auto()
    FORMAL = auto()
    MIXED = auto()


class BenchmarkCategory(Enum):
    WORLD_KNOWLEDGE = auto()
    MISCELLANEOUS = auto()
    REASONING = auto()


@dataclass
class Benchmark(BaseClass, ABC):
    name: str
    type: BenchmarkType
    category: BenchmarkCategory
    description: str

    @abstractmethod
    def evaluate(self, model: Model) -> dict:
        pass

    @classmethod
    def create(cls, name: str) -> "Benchmark":
        # Get all available benchmark implementations
        benchmark_classes = cls.get_others()

        return benchmark_classes[name](name)


@dataclass
class HuggingfaceBenchmark(Benchmark, ABC):
    id: str
    api_key: str = field(init=False)

    def __post_init__(self) -> None:
        self.api_key = self.load_credentials("huggingface")
        kwargs = {"token": self.api_key} if self.api_key else {}
        self.benchmark = load_dataset(self.id, **kwargs)
