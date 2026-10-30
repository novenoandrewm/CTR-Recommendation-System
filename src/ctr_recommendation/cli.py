"""Command-line interface for the CTR recommendation system."""

import argparse
from importlib.metadata import version


def main() -> None:
    """Run the project command-line interface."""
    parser = argparse.ArgumentParser(
        prog="ctr",
        description="Huawei TechArena CTR Recommendation System",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('ctr-recommendation-system')}",
    )

    parser.parse_args()


if __name__ == "__main__":
    main()
