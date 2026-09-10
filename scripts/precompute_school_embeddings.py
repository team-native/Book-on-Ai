"""Precompute BookOn school-book embeddings into .npy + metadata files."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import build_school_embedding_store, print_precompute_result  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute BookOn school book embeddings.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new Gemini embeddings to create in this run.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to wait between Gemini embedding calls.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_precompute_result(
        build_school_embedding_store(
            max_new_embeddings=args.limit,
            sleep_seconds=args.sleep,
        )
    )
