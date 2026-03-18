from __future__ import annotations

"""Run the clustering pipeline from the command line using a JSON config file."""

import argparse

from space_time_cluster.config import load_config
from space_time_cluster.pipeline import run_pipeline


def main() -> None:
    """Parse CLI arguments, load config, and execute the pipeline.

    Inputs:
        None. Arguments are read from the command line.

    Returns:
        None. The function runs the pipeline for the provided config file.
    """
    parser = argparse.ArgumentParser(description="Run the unix-seconds space-time clustering pipeline.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
