import argparse

import weave
from dotenv import load_dotenv
from itakello_logging import ItakelloLogging

from src.benchmarks import get_benchmarks
from src.benchmarks.mmlu_benchmark import MMLUBenchmark
from src.models import get_model
from src.models.base_model import ModelBackend
from src.models.huggingface_model import HfModel

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = ItakelloLogging(debug=True).get_logger(__name__)

# Set up Weave
# weave.init("Playpen Evaluation")


def main(args: argparse.Namespace) -> None:
    # mmlu_benchmark = MMLUBenchmark()
    # print(mmlu_benchmark)
    hf_model = HfModel(
        # name="Llama-3.2-1B-Instruct",
        id="meta-llama/Llama-3.2-1B-Instruct",
        backend=ModelBackend.HUGGINGFACE,
    )
    print(hf_model)
    """benchmarks = get_benchmarks(args.benchmark)
    model = get_model(args.model)
    for benchmark, df in results.items():
        logger.info(f"{benchmark}:\n{df}\n")"""


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
