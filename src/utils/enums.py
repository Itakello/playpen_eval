from enum import Enum, auto


class ModelBackend(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class BenchmarkType(Enum):
    FUNCTIONAL = auto()
    FORMAL = auto()
    MIXED = auto()


class BenchmarkCategory(Enum):
    WORLD_KNOWLEDGE = auto()
    MISCELLANEOUS = auto()
    REASONING = auto()
