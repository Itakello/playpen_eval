import json

from ..config.config import MODEL_REGISTRY_PATH
from .base_model import Model


def get_model(model_name: str) -> Model:
    model_entry = get_model_from_registry(model_name)
    model = Model.create(
        model_name=model_entry["model_id"], backend=model_entry["backend"]
    )
    return model


def get_model_id(model_name: str) -> str:
    model_entry = get_model_from_registry(model_name)
    return model_entry['model_id']


def get_model_from_registry(model_name: str) -> dict:
    model_registry = json.loads(MODEL_REGISTRY_PATH.read_text())

    model_entry = next(
        (entry for entry in model_registry if entry.get("model_name") == model_name),
        None,
    )

    if model_entry is None:
        raise ValueError(f"Model {model_name} not found.")
    return model_entry
