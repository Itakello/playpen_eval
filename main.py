import argparse

from src.models import get_model

# from src.eval.benchmarks import get_benchmarks, run_benchmarks


def main(args: argparse.Namespace) -> None:
    model = get_model(args.model)
    print(model)
    # benchmarks = get_benchmarks(args.benchmarks)
    # run_benchmarks(model, benchmarks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument(
        "-b", "--benchmarks", type=str, nargs="*", help="Value can be 'all'"
    )
    main(parser.parse_args())
