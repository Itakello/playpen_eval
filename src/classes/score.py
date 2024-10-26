from dataclasses import dataclass

from .task import Task


@dataclass
class Score:
    overall: float
    per_task: dict[Task, float]
