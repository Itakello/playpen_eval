import logging
from dataclasses import dataclass

import torch
from huggingface_hub.utils import GatedRepoError
from itakello_logging import ItakelloLogger
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.config import DEVICE
from src.eval.backends.base_model import Model

logger = ItakelloLogger()


@dataclass
class HuggingfaceModel(Model):
    model: AutoModelForCausalLM | None = None
    tokenizer: AutoTokenizer | None = None

    def __post_init__(self) -> None:
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        try:
            kwargs = {"token": self.api_key} if self.api_key else {}
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        except GatedRepoError as e:
            logger.error(
                f"Access to model {self.model_name} is restricted. Error: {str(e)}"
            )
            logger.info(
                f"Please visit https://huggingface.co/{self.model_name} to request access."
            )
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")

    def generate(self, prompt: str) -> str:
        # Ensure model and tokenizer are on the correct device
        self.model.to(DEVICE)
        self.tokenizer.to(DEVICE)

        # Tokenize input
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(DEVICE)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs, do_sample=True, num_return_sequences=1
            )

        # Decode and return
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
