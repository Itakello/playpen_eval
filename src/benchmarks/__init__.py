from typing import Type

from itakello_logging import ItakelloLogging

from .custom_benchmark import CustomBenchmark

logger = ItakelloLogging().get_logger(__name__)


def get_benchmarks(benchmark_names: list[str]) -> dict[str, CustomBenchmark]:
    if len(benchmark_names) == 1 and benchmark_names[0] == "all":
        benchmarks = CustomBenchmark.get_all_subclasses()
    else:
        benchmarks = CustomBenchmark.get_specific_subclasses(benchmark_names)

    benchmarks = {name: benchmark() for name, benchmark in benchmarks.items()}  # type: ignore

    if None in benchmarks.values():
        logger.error("One or more benchmarks could not be found")
        exit(1)
    logger.debug(
        "Evaluating the following benchmarks:\n"
        + "".join([f"- {benchmark}\n" for benchmark in benchmarks])
    )
    return benchmarks
