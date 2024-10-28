from ..classes.score import Score
from ..models.base_model import Model
from .base_benchmark import Benchmark
from .lighteval_benchmark import LightEvalBenchmark


def get_benchmark(benchmark_name: str) -> Benchmark:
    benchmark = Benchmark.create(benchmark_name)

    return benchmark


def run_benchmark(model: Model, benchmark: Benchmark) -> dict[str, Score]:
    results = {}
    result = benchmark.evaluate(model)
    results[benchmark.name] = result
    return results

def run_benchmark_lighteval(model_name:str, benchmark: LightEvalBenchmark) -> dict[str, Score]:
    results = {}
    result = benchmark.evaluate(model_name)
    results[benchmark.name] = result
    return results
