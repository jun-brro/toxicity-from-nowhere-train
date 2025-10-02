"""Metric computations for latent toxicity alignment."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy


def dfreq(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    active = (z != 0).astype(np.float32)
    harmful = labels == 1
    benign = labels == 0
    freq_harmful = _safe_mean(active[harmful], axis=0)
    freq_benign = _safe_mean(active[benign], axis=0)
    return freq_harmful - freq_benign


def dmag(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    mag = np.abs(z)
    harmful = labels == 1
    benign = labels == 0
    return _safe_mean(mag[harmful], axis=0) - _safe_mean(mag[benign], axis=0)


def auroc(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.abs(z)
    results = []
    for idx in range(scores.shape[1]):
        try:
            results.append(roc_auc_score(labels, scores[:, idx]))
        except ValueError:
            results.append(float("nan"))
    return np.asarray(results)


def kl(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    harmful = z[labels == 1]
    benign = z[labels == 0]
    results = []
    for idx in range(z.shape[1]):
        h_hist, bins = np.histogram(harmful[:, idx], bins=50, density=True)
        b_hist, _ = np.histogram(benign[:, idx], bins=bins, density=True)
        h_hist += 1e-9
        b_hist += 1e-9
        results.append(entropy(h_hist, b_hist))
    return np.asarray(results)


def consensus(scores: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    total = None
    for key, value in scores.items():
        weight = weights.get(key, 1.0)
        if total is None:
            total = weight * _standardize(value)
        else:
            total += weight * _standardize(value)
    return total


def _standardize(arr: np.ndarray) -> np.ndarray:
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    return (arr - mean) / max(std, 1e-6)


def _safe_mean(arr: np.ndarray, axis: int) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(arr.shape[1])
    return np.nanmean(arr, axis=axis)


__all__ = ["dfreq", "dmag", "auroc", "kl", "consensus"]
