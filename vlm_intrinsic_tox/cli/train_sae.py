"""CLI for SAE training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..utils.config import load_config
from ..sae.io import load_shards, save_train_result
from ..sae.train import train_sae
from ..utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Top-K SAE")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="Config overrides (e.g., layers=[0,2,4])")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    cfg = load_config([args.config], overrides=args.overrides)
    
    # Train SAE for each layer
    for layer in cfg.layers:
        shard_files = list(Path(cfg.io.extract_dir).glob(f"*layer_{layer}*.npz"))
        shard_arrays = list(load_shards(shard_files))
        result = train_sae(layer, shard_arrays, cfg.sae)
        output_path = Path(cfg.io.output_dir) / f"sae_layer{layer}"
        save_train_result(result, output_path)


if __name__ == "__main__":
    main()
