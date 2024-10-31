from dataclasses import dataclass, field

import torch
import weave
from itakello_logging import ItakelloLogging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from weave import Model as WeaveModel

from src.config.config import DEVICE

from .base_model import BaseModel

logger = ItakelloLogging().get_logger(__name__)


@dataclass
class HfModel(BaseModel, WeaveModel):
    model: PreTrainedModel | None = field(init=False, default=None)
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        assert self.name is not None, logger.error("Model name must be provided")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.name, device_map=DEVICE, max_memory={0: "5.5GB"}
            )
        except Exception as e:
            logger.error(f"Error loading model {self.name}: {e}")
            exit(1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

    @weave.op
    def predict(self, input: str) -> str:
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not initialized.")
            return ""

        # Tokenize input
        model_inputs = self.tokenizer([input], return_tensors="pt")
        model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                do_sample=False,
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
