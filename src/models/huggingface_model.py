from dataclasses import dataclass, field
from typing import Optional, Type

import torch
from itakello_logging import ItakelloLogging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.config.config import DEVICE

from .base_model import Model, ModelBackend

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class HuggingfaceModel(Model):
    model: Optional[PreTrainedModel] = field(default=None, init=False)
    tokenizer: Optional[PreTrainedTokenizer] = field(default=None, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        kwargs = {"token": self.api_key} if self.api_key else {}
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)

    @staticmethod
    def get_backend_class(backend: ModelBackend) -> Type["Model"]:
        if backend == Model.ModelBackend.HUGGINGFACE:
            return HuggingfaceModel
        raise ValueError(f"Unsupported backend: {backend}")

    def generate(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not initialized.")
            return ""

        # Ensure model is on the correct device
        self.model.to(DEVICE)  # type: ignore

        # Tokenize input
        model_inputs = self.tokenizer([prompt], return_tensors="pt")
        model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs, do_sample=True, num_return_sequences=1
            )

        # Decode and return
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
