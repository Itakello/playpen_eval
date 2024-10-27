from dataclasses import dataclass

from pandas import DataFrame


@dataclass
class Score:
    overall: float
    per_task: DataFrame
