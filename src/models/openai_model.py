from dataclasses import dataclass

from .base_model import Model


@dataclass
class OpenaiModel(Model):

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("OpenAI model generation not implemented")
