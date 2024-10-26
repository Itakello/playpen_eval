from abc import ABC, abstractmethod
from dataclasses import dataclass

from deepeval.models.base_model import DeepEvalBaseLLM


@dataclass
class Model(ABC):
    model_name: str
    api_key: str | None = None
