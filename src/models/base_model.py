import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

from deepeval.models.base_model import DeepEvalBaseLLM

from ..classes.base_class import BaseClass


class ModelBackend(Enum):
    HUGGINGFACE = auto()
    OPENAI = auto()


@dataclass
class Model(DeepEvalBaseLLM, BaseClass, ABC):
    model_name: str
    backend: ModelBackend
    api_key: str = field(init=False)

    def __post_init__(self) -> None:
        self.api_key = self._load_credentials()

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    """@abstractmethod
    def batch_generate(self, prompts: list[str]) -> list[str]:
        pass"""

    def _load_credentials(self) -> str:
        """Load API key from environment variables."""
        env_var_name = f"{self.backend.upper()}_API_KEY"
        key = os.getenv(env_var_name)
        if key is None:
            raise ValueError(f"API key for {self.backend.name} not found.")
        return key

    @classmethod
    def create(cls, model_name: str, backend: str) -> "Model":
        # Get all available model implementations
        model_classes = cls.get_others()

        # Find the appropriate model class for the backend
        return model_classes[backend](model_name, backend)
