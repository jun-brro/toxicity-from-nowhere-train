"""Aggregate latent metric summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .metrics import auroc, consensus, dfreq, dmag, kl


@dataclass
class LatentSummary:
    layer: int
    dfreq: np.ndarray
    dmag: np.ndarray
    auroc: np.ndarray
    kl: np.ndarray
    consensus: np.ndarray

    def top_latents(self, top_k: int) -> List[int]:
        indices = np.argsort(self.consensus)[::-1]
        return indices[:top_k].tolist()


def analyze_latents(layer: int, z: np.ndarray, labels: np.ndarray, weights: Dict[str, float]) -> LatentSummary:
    scores = {
        "dfreq": dfreq(z, labels),
        "dmag": dmag(z, labels),
        "auroc": auroc(z, labels),
        "kl": kl(z, labels),
    }
    combined = consensus(scores, weights)
    return LatentSummary(
        layer=layer,
        dfreq=scores["dfreq"],
        dmag=scores["dmag"],
        auroc=scores["auroc"],
        kl=scores["kl"],
        consensus=combined,
    )


__all__ = ["LatentSummary", "analyze_latents"]
