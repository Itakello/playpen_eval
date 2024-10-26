from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

from datasets import load_dataset

from ..classes.base_class import BaseClass
from ..classes.score import Score
from ..models.base_model import Model


class BenchmarkType(Enum):
    FUNCTIONAL = auto()
    PERFORMANCE = auto()
    SAFETY = auto()


class BenchmarkCategory(Enum):
    MATH = auto()
    REASONING = auto()


@dataclass
class Benchmark(BaseClass, ABC):
    name: str
    type: BenchmarkType
    category: BenchmarkCategory
    description: str

    @abstractmethod
    def evaluate(self, model: Model) -> Score:
        pass

    @classmethod
    def create(cls, name: str) -> "Benchmark":
        # Get all available benchmark implementations
        benchmark_classes = cls.get_others()

        return benchmark_classes[name](name)


@dataclass
class HuggingfaceBenchmark(Benchmark, ABC):
    def __post_init__(self) -> None:
        self.benchmark = load_dataset(self.id)
