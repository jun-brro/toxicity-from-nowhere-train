"""CLI for SAE training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..sae.io import load_shards, save_train_result
from ..sae.train import train_sae
from ..utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Top-K SAE")
    parser.add_argument("--config", type=Path, nargs="+", help="Config YAML paths")
    parser.add_argument("--layer", type=int, help="Layer index")
    parser.add_argument("--shards", type=Path, nargs="+", help="Shard npz files")
    parser.add_argument("--output", type=Path, required=True, help="Output prefix for model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    cfg = load_config(args.config)
    shard_arrays = list(load_shards(args.shards))
    result = train_sae(args.layer, shard_arrays, cfg.sae)
    save_train_result(result, args.output)


if __name__ == "__main__":
    main()
