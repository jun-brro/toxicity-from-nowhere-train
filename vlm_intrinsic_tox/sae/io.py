"""Persistence helpers for SAE artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, List

import numpy as np
import torch

from ..utils.logging import get_logger
from .modules import TopKSAE
from .train import Scaler, TrainResult


LOGGER = get_logger(__name__)


def save_train_result(result: TrainResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), path.with_suffix(".pt"))
    meta = {
        "input_dim": result.model.input_dim,
        "latent_dim": result.model.latent_dim,
        "topk": result.model.topk_k,
    }
    with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    with open(path.with_suffix(".scaler.json"), "w", encoding="utf-8") as handle:
        json.dump({"mean": result.scaler.mean.tolist(), "std": result.scaler.std.tolist()}, handle, indent=2)
    with open(path.with_suffix(".history.json"), "w", encoding="utf-8") as handle:
        json.dump(result.history, handle, indent=2)


def load_scaler(path: Path) -> Scaler:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    return Scaler(mean=mean, std=std)


def load_shards(paths: Iterable[Path]) -> Iterator[np.ndarray]:
    """Load activation shards only (for SAE training)."""
    for path in paths:
        data = np.load(path, allow_pickle=True)
        # Extract runner saves as 'delta' key
        if "delta" in data:
            yield data["delta"].astype(np.float32)
        elif "activations" in data:
            # Backward compatibility
            yield data["activations"].astype(np.float32)
        else:
            raise ValueError(f"Invalid shard file {path}: missing 'delta' or 'activations' key")


def load_shards_with_labels(paths: Iterable[Path], label_key: str = "label") -> tuple[np.ndarray, np.ndarray]:
    """Load activations and labels from shards for evaluation.
    
    Args:
        paths: Paths to shard files
        label_key: Key to extract from metadata (default: 'label')
        
    Returns:
        Tuple of (activations, labels) arrays
    """
    all_activations = []
    all_labels = []
    
    for path in paths:
        # Load activations
        data = np.load(path, allow_pickle=True)
        if "delta" in data:
            activations = data["delta"].astype(np.float32)
        elif "activations" in data:
            activations = data["activations"].astype(np.float32)
        else:
            raise ValueError(f"Invalid shard file {path}: missing 'delta' or 'activations' key")
        
        # Load metadata with labels
        metadata = load_shard_metadata(path)
        
        # Extract labels from metadata
        labels = []
        for meta in metadata:
            label = meta.get(label_key)
            if label is None:
                # Try alternative keys
                if "toxicity_type" in meta:
                    # Map toxicity_type string to numeric label
                    toxicity_map = {"benign": 0, "explicit": 1, "implicit": 2}
                    label = toxicity_map.get(meta["toxicity_type"], -1)
                else:
                    label = -1  # Unknown label
            labels.append(label)
        
        labels = np.array(labels, dtype=np.int32)
        
        if len(labels) != len(activations):
            raise ValueError(f"Mismatch: {len(activations)} activations but {len(labels)} labels in {path}")
        
        all_activations.append(activations)
        all_labels.append(labels)
    
    return np.concatenate(all_activations, axis=0), np.concatenate(all_labels, axis=0)


def load_shard_metadata(path: Path) -> List[dict]:
    meta_path = path.with_suffix(".meta.json")
    with open(meta_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("metadata", [])


def load_model(prefix: Path, device: str = "cpu") -> TopKSAE:
    with open(prefix.with_suffix(".meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    model = TopKSAE(d=meta["input_dim"], m=meta["latent_dim"], k=meta["topk"])
    state = torch.load(prefix.with_suffix(".pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


__all__ = ["save_train_result", "load_scaler", "load_shards", "load_shards_with_labels", "load_shard_metadata", "load_model"]
