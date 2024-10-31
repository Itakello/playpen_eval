from itakello_logging import ItakelloLogging

from .base_benchmark import BaseBenchmark

logger = ItakelloLogging().get_logger(__name__)


def get_benchmarks(benchmark_names: list[str]) -> dict[str, BaseBenchmark]:
    if len(benchmark_names) == 1 and benchmark_names[0] == "all":
        benchmarks = BaseBenchmark.get_all_subclasses()
    else:
        benchmarks = BaseBenchmark.get_specific_subclasses(benchmark_names)

    if None in benchmarks.values():
        logger.error("One or more benchmarks could not be found")
        exit(1)
    logger.debug(
        "Evaluating the following benchmarks:\n"
        + "".join([f"- {benchmark}\n" for benchmark in benchmarks])
    )
    return benchmarks


"""def run_benchmark(model: Model, benchmark: BaseBenchmark) -> dict[str, Score]:
    results = {}
    result = benchmark.evaluate(model)
    results[benchmark.name] = result
    return results"""
