from pathlib import Path

import torch

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_REGISTRY_PATH: Path = Path("src/config/model_registry.json")
