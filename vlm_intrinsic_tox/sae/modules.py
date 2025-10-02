"""Sparse Autoencoder modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class TopK(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k <= 0:
            return torch.zeros_like(x)
        topk = torch.topk(torch.abs(x), k=min(self.k, x.shape[-1]), dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk.indices, 1.0)
        return x * mask


class TopKSAE(nn.Module):
    def __init__(self, d: int, m: int, k: int) -> None:
        super().__init__()
        self.input_dim = d
        self.latent_dim = m
        self.topk_k = k
        self.encoder = nn.Linear(d, m, bias=True)
        self.decoder = nn.Linear(m, d, bias=True)
        self.topk = TopK(k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.encoder(x)
        return self.topk(logits)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


__all__ = ["TopKSAE"]
