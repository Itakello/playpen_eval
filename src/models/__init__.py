import json
import os
from pathlib import Path
from typing import Optional


from .huggingface_model import HuggingfaceModel


def load_credentials(backend: str) -> Optional[str]:
    """Load API key from environment variables."""
    env_var_name = f"{backend.upper()}_API_KEY"
    return os.getenv(env_var_name)


def get_model(model_name: str) -> HuggingfaceModel:
    model_registry_path = (
        Path(__file__).parent.parent / "config" / "model_registry.json"
    )
    model_registry = json.loads(model_registry_path.read_text())

    model_entry = next(
        (entry for entry in model_registry if entry.get("model_name") == model_name),
        None,
    )

    if model_entry is None:
        raise ValueError(f"Model {model_name} not found.")

    if model_entry["backend"] == "huggingface":
        api_key = (
            load_credentials(model_entry["backend"])
            if model_entry["requires_api_key"]
            else None
        )
        return HuggingfaceModel(model_entry["model_id"], api_key)

    # TODO: OpenAI
    raise ValueError(f"Unsupported backend: {model_entry['backend']}")
