import json

from itakello_logging import ItakelloLogging

from ..config.config import MODEL_REGISTRY_PATH
from .custom_model import CustomModel, ModelBackend
from .hf_model import HfModel

logger = ItakelloLogging().get_logger(__name__)


def get_model(name: str) -> CustomModel:
    entry = get_model_from_registry(name)
    backend = ModelBackend(entry["backend"])
    if backend == ModelBackend.HUGGINGFACE:
        model = HfModel(id=entry["id"], name=entry["name"])
    elif backend == ModelBackend.OPENAI:
        raise NotImplementedError("OpenAI models are not yet supported.")
    else:
        raise ValueError(f"Model backend {backend} not supported.")
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
