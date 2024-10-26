import json

from ..config.config import MODEL_REGISTRY_PATH
from .base_model import Model


def get_model(model_name: str) -> Model:
    model_registry = json.loads(MODEL_REGISTRY_PATH.read_text())

    model_entry = next(
        (entry for entry in model_registry if entry.get("model_name") == model_name),
        None,
    )

    if model_entry is None:
        raise ValueError(f"Model {model_name} not found.")

    model = Model.create(
        model_name=model_entry["model_id"], backend=model_entry["backend"]
    )

    return model
