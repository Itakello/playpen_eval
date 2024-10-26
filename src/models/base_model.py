import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto


class ModelBackend(Enum):
    OPENAI = auto()
    HUGGINGFACE = auto()


@dataclass
class Model(ABC):
    model_name: str
    backend: ModelBackend
    api_key: str = field(init=False)

    def __post_init__(self) -> None:
        self.api_key = self._load_credentials()

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    def _load_credentials(self) -> str:
        """Load API key from environment variables."""
        env_var_name = f"{self.backend.name}_API_KEY"
        key = os.getenv(env_var_name)
        if key is None:
            raise ValueError(f"API key for {self.backend.name} not found.")
        return key

    @classmethod
    def create(cls, model_name: str, backend: str) -> "Model":
        try:
            backend_enum = ModelBackend[backend.upper()]
        except KeyError:
            raise ValueError(f"Unsupported backend: {backend}")

        if backend_enum == ModelBackend.HUGGINGFACE:
            from .huggingface_model import HuggingfaceModel

            return HuggingfaceModel(model_name, ModelBackend.HUGGINGFACE)
        elif backend_enum == ModelBackend.OPENAI:
            from .openai_model import OpenaiModel

            return OpenaiModel(model_name, ModelBackend.OPENAI)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
