#!/usr/bin/env python3
"""SAE training CLI."""

import argparse
from pathlib import Path

from ..utils.config import load_config
from ..utils.logging import configure_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoders")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="Config overrides (e.g., sae.lr=0.001)")
    
    args = parser.parse_args()
    
    configure_logging()
    logger = get_logger(__name__)
    
    # Load configuration
    config = load_config([args.config], overrides=args.overrides)
    logger.info(f"Loaded config from {args.config}")
    logger.info(f"SAE config: {config.sae}")
    
    # TODO: Implement SAE training logic
    logger.info("SAE training not yet implemented")


if __name__ == "__main__":
    main()