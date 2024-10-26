from .base_model import Model


class OpenaiModel(Model):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("OpenAI model generation not implemented")
