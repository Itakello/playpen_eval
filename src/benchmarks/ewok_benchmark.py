from dataclasses import dataclass

from weave.flow.dataset import Dataset as WeaveDataset

from ..classes.score import Score
from ..models.base_model import BaseModel
from .base_benchmark import HfBenchmark, BenchmarkCategory, BenchmarkType


@dataclass
class EwokBenchmark(HfBenchmark):
    id: str = "ewok-core/ewok-core-1.0"
    type: BenchmarkType = BenchmarkType.FUNCTIONAL
    category: BenchmarkCategory = BenchmarkCategory.WORLD_KNOWLEDGE
    description: str = "TEMP"

    def _build_dataset(self) -> WeaveDataset:
        data = None
        self.weave_dataset = WeaveDataset(name="ewok", rows = data)

    def evaluate(self, model: BaseModel) -> Score:
        pass
