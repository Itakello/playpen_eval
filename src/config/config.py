from pathlib import Path

import torch

# Constants
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS: int = 200
N_SHOTS: int = 5

# Paths
MODEL_REGISTRY_PATH: Path = Path("src/config/model_registry.json")
