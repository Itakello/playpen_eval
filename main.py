import argparse

from dotenv import load_dotenv
from itakello_logging import ItakelloLogging

from src.benchmarks import get_benchmarks, run_benchmarks
from src.models import get_model

# Load environment variables from .env file
load_dotenv()

logger = ItakelloLogging(debug=True).get_logger(__name__)


def main(args: argparse.Namespace) -> None:
    logger.debug(f"Running with model: {args.model}")
    model = get_model(args.model)
    benchmarks = get_benchmarks(args.benchmarks)
    run_benchmarks(model, benchmarks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Name of the model to use"
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        type=str,
        nargs="*",
        required=True,
        help="Benchmark names to run. Use 'all' to run all benchmarks",
    )
    main(parser.parse_args())
