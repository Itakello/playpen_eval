from typing import Optional

import torch
import weave
from itakello_logging import ItakelloLogging
from pydantic import BaseModel, Field, model_validator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from src.config.config import DEVICE

from ..utils.enums import ModelBackend
from .custom_model import CustomModel

logger = ItakelloLogging().get_logger(__name__)


class HfModel(CustomModel, BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    backend: ModelBackend = ModelBackend.HUGGINGFACE
    model: Optional[PreTrainedModel] = Field(default=None)
    tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = Field(
        default=None
    )

    @model_validator(mode="after")
    def initialize_model(self) -> "HfModel":
        assert self.name, logger.error("Model name must be provided")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.id, device_map=DEVICE, max_memory={0: "5.5GB"}
            )
        except Exception as e:
            logger.error(f"Error loading model {self.id}: {e}")
            exit(1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.id)
        return self

    @weave.op
    def predict(self, question: str) -> str:
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not initialized.")
            return ""

        # Tokenize input
        model_inputs = self.tokenizer([question], return_tensors="pt")
        model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                do_sample=True,
                num_return_sequences=1,
                max_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and return
        return self.tokenizer.decode(
            generated_ids[0],
            padding=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
