from ..classes.score import Score
from ..models.base_model import Model
from .base_benchmark import Benchmark


def get_benchmarks(benchmark_names: list[str]) -> list[Benchmark]:
    if len(benchmark_names) == 1 and benchmark_names[0] == "all":
        benchmark_names = list(Benchmark.get_others().keys())

    benchmarks = [Benchmark.create(name) for name in benchmark_names]

    return benchmarks


def run_benchmarks(model: Model, benchmarks: list[Benchmark]) -> dict[str, Score]:
    results = {}
    for benchmark in benchmarks:
        result = benchmark.evaluate(model)
        results[benchmark.name] = result
    return results
