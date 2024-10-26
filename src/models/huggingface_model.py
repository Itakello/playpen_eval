from dataclasses import dataclass, field
from typing import Optional

import torch
from itakello_logging import ItakelloLogging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.config.config import DEVICE, MAX_NEW_TOKENS

from .base_model import Model

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class HuggingfaceModel(Model):
    model: Optional[PreTrainedModel] = field(default=None, init=False)
    tokenizer: Optional[PreTrainedTokenizer] = field(default=None, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        kwargs = {"token": self.api_key} if self.api_key else {}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto", **kwargs
            )
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise e
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)

    def generate(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not initialized.")
            return ""

        # Tokenize input
        model_inputs = self.tokenizer([prompt], return_tensors="pt")
        model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                do_sample=True,
                num_return_sequences=1,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        # Decode and return
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self) -> None:
        return self.model
