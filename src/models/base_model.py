from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from ..classes.base_class import BaseClass


class ModelBackend(Enum):
    HUGGINGFACE = auto()
    OPENAI = auto()


@dataclass
class Model(BaseClass, ABC):
    name: str
    backend: ModelBackend
    api_key: str = field(init=False)

    def __post_init__(self) -> None:
        self.api_key = self.load_credentials(self.backend)

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    """@abstractmethod
    def batch_generate(self, prompts: list[str]) -> list[str]:
        pass"""

    @classmethod
    def create(cls, model_name: str, backend: str) -> "Model":
        # Get all available model implementations
        model_classes = cls.get_others()

        # Find the appropriate model class for the backend
        return model_classes[backend](model_name, backend)

    @abstractmethod
    def load_model(self) -> Any:
        pass
