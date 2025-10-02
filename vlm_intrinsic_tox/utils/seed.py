"""Seed utilities for reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch

from .logging import get_logger

LOGGER = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set global random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    LOGGER.info("Setting global seed to %d", seed)
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


__all__ = ["set_seed"]