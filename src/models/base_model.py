from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Model(ABC):
    model_name: str
    api_key: str | None = None

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
