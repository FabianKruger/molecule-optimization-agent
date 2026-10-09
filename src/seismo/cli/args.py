import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM-driven molecular optimization experiment."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML experiment config",
    )
    return parser.parse_args()
