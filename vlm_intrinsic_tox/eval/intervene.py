"""Latent intervention utilities."""

from __future__ import annotations

import numpy as np


def gate_latents(z: np.ndarray, idxs) -> np.ndarray:
    gated = z.copy()
    gated[:, idxs] = 0.0
    return gated


def project_out_residual(h: np.ndarray, Wd: np.ndarray, z: np.ndarray, idxs) -> np.ndarray:
    component = z[:, idxs] @ Wd[idxs]
    return h - component


__all__ = ["gate_latents", "project_out_residual"]
