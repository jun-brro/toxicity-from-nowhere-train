"""CLI entrypoint for δ_int extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..utils.logging import configure_logging
from ..extract.runner import ExtractRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract δ_int activations")
    parser.add_argument("--config", type=Path, nargs="+", help="YAML config files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    cfg = load_config(args.config)
    runner = ExtractRunner(cfg.extract)
    runner.run()


if __name__ == "__main__":
    main()
