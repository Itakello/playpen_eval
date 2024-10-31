import argparse

import weave
from dotenv import load_dotenv
from itakello_logging import ItakelloLogging
from tqdm import tqdm

from src.benchmarks import get_benchmarks
from src.models import get_model

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = ItakelloLogging(
    debug=False, excluded_modules=["gql.transport.requests", "urllib3.connectionpool"]
).get_logger(__name__)

# Set up Weave
weave.init("Playpen Evaluation")


def main(args: argparse.Namespace) -> None:
    """mmlu = MMLUBenchmark()
    hf = HfModel(name="Llama-3.2-1B-Instruct", id="meta-llama/Llama-3.2-1B-Instruct")"""
    model = get_model(args.model)
    benchmarks = get_benchmarks(args.benchmark)
    for name, benchmark in tqdm(benchmarks.items(), desc="Running benchmarks"):
        benchmark.evaluate(model=model)
        print("hello")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Name of the model to use"
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        nargs="+",
        required=True,
        help="Names of the benchmarks to run",
    )
    main(parser.parse_args())
