import json

from itakello_logging import ItakelloLogging

from ..config.config import MODEL_REGISTRY_PATH
from .base_model import BaseModel

logger = ItakelloLogging().get_logger(__name__)


def get_model(name: str) -> BaseModel:
    entry = get_model_from_registry(name)
    model = BaseModel.create(
        id=entry["id"], name=entry["name"], backend=entry["backend"]
    )
    logger.debug(f"Model {name} loaded.")
    return model


def get_model_id(name: str) -> str:
    entry = get_model_from_registry(name)
    return entry["model_id"]


def get_model_from_registry(name: str) -> dict:
    registry = json.loads(MODEL_REGISTRY_PATH.read_text())

    entry = next(
        (entry for entry in registry if entry.get("name") == name),
        None,
    )

    if entry is None:
        raise ValueError(f"Model {name} not found.")
    return entry
