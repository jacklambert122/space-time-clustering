from __future__ import annotations

import argparse

from space_time_cluster.config import load_config
from space_time_cluster.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unix-seconds space-time clustering pipeline.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
