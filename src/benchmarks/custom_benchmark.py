import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Optional

from datasets import Dataset as HfDataset
from itakello_logging import ItakelloLogging
from pydantic import BaseModel, Field, model_validator
from weave import Dataset as WeaveDataset
from weave import Evaluation, Table

from ..classes.custom_class import CustomClass
from ..models.custom_model import BaseModel as BaseModelABC
from ..utils.enums import BenchmarkCategory, BenchmarkType

logger = ItakelloLogging().get_logger(__name__)


class CustomBenchmark(CustomClass, ABC, BaseModel):
    name: str
    type: BenchmarkType
    category: BenchmarkCategory
    description: str
    weave_dataset: Optional[WeaveDataset] = Field(default=None)
    scoring_function: Optional[Callable] = Field(default=None)

    def _create_weave_dataset(self) -> "CustomBenchmark":
        self.weave_dataset = self._build_dataset()
        return self

    def evaluate(self, model: BaseModelABC) -> None:
        assert self.weave_dataset
        assert self.scoring_function
        evaluation = Evaluation(
            dataset=self.weave_dataset, scorers=[self.scoring_function]
        )
        asyncio.run(evaluation.evaluate(model))

    @abstractmethod
    def _build_dataset(self) -> WeaveDataset:
        """Abstract method to build a dataset."""
        pass


class HfBenchmark(CustomBenchmark, ABC, BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    id: str
    hf_dataset: Optional[list[dict]] = Field(default=None)

    @model_validator(mode="after")
    def create_hf_dataset(self) -> "HfBenchmark":
        # First load the HF dataset
        self.hf_dataset = self._load_test_dataset()
        self._create_weave_dataset()
        return self

    def _build_dataset(self) -> WeaveDataset:
        assert self.hf_dataset
        table_rows = Table(self.hf_dataset)
        return WeaveDataset(name=self.name, rows=table_rows)

    @abstractmethod
    def _load_test_dataset(self) -> list[dict]:
        pass

    def __str__(self) -> str:
        return f"{self.name} ({self.id})"
