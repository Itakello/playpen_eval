import argparse

from dotenv import load_dotenv
from itakello_logging import ItakelloLogging

from src.benchmarks import run_benchmark, run_benchmark_lighteval
from src.benchmarks.base_benchmark import Benchmark
from src.benchmarks.lighteval_benchmark import LightEvalBenchmark
from src.models import get_model, get_model_id

# Load environment variables from .env file
load_dotenv()

logger = ItakelloLogging(debug=True).get_logger(__name__)


def main(args: argparse.Namespace) -> None:
    if len(args.benchmark) == 1 and args.benchmark[0] == "all":
        benchmarks = Benchmark.get_all_subclasses()
    else:
        benchmarks = [Benchmark.get_specific_subclass(benchmark) for benchmark in args.benchmark]
    logger.debug(
        f"Evaluating the following benchmarks\n: {['-' + benchmark + '\n' for benchmark in benchmarks]}"
    )
    logger.debug(f"Running with model: {args.model}")
    if issubclass(benchmark.__class__, LightEvalBenchmark):
        model_name = get_model_id(args.model)
        results = run_benchmark_lighteval(model_name, benchmark)
    else:
        model = get_model(args.model)
        results = run_benchmark(model, benchmark)
    for benchmark, df in results.items():
        logger.info(f"{benchmark}:\n{df}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Name of the model to use"
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        required=True,
        help="Name of the benchmark to run.",
    )
    main(parser.parse_args())
