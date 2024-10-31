from abc import ABC
from typing import TypeVar

from pydantic import BaseModel
from weave import Model as WeaveModel

from ..utils.enums import ModelBackend

T = TypeVar("T", bound="BaseModel")


class CustomModel(ABC, WeaveModel, BaseModel):
    id: str
    backend: ModelBackend

    def __str__(self) -> str:
        return f"{self.backend} model: {self.id}"
