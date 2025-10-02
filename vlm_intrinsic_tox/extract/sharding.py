"""Shard writer for δ_int activations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from ..utils.logging import get_logger


LOGGER = get_logger(__name__)


class ShardWriter:
    def __init__(self, base_dir: Path, layer: int, max_rows: int) -> None:
        self.base_dir = base_dir
        self.layer = layer
        self.max_rows = max_rows
        self.rows: List[np.ndarray] = []
        self.metadata: List[Dict[str, object]] = []
        self.counter = 0
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def add(self, array: np.ndarray, meta: Dict[str, object]) -> None:
        self.rows.append(array.astype(np.float16))
        self.metadata.append(meta)
        if len(self.rows) >= self.max_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        stacked = np.stack(self.rows, axis=0)
        shard_id = self.counter
        path = self.base_dir / f"layer{self.layer:02d}_shard{shard_id:05d}.npz"
        meta_path = path.with_suffix(".meta.json")
        LOGGER.info("Writing shard layer=%s rows=%s to %s", self.layer, len(stacked), path)
        np.savez_compressed(path, activations=stacked)
        meta_payload = {
            "rows": len(stacked),
            "md5": hashlib.md5(stacked.tobytes()).hexdigest(),
            "metadata": self.metadata,
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True))
        self.rows.clear()
        self.metadata.clear()
        self.counter += 1

    def close(self) -> None:
        self.flush()


__all__ = ["ShardWriter"]
