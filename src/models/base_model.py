from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import Type, TypeVar

from ..classes.base_class import BaseClass

T = TypeVar("T", bound="BaseModel")


class ModelBackend(Enum):
    HUGGINGFACE = auto()
    OPENAI = auto()


@dataclass
class BaseModel(BaseClass, ABC):
    id: str
    backend: ModelBackend

    @classmethod
    def create(cls: Type[T], backend: str, id: str) -> T:
        # Get all available model implementations
        model_classes = cls.get_specific_subclasses([backend])

        # Find the appropriate model class for the backend
        model_class = model_classes[backend]
        return model_class(backend=backend, id=id)
