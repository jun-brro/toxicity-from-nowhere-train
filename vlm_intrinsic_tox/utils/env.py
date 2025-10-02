"""Environment helpers (seeds, reproducibility metadata)."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch


@dataclass
class RunMetadata:
    repo_sha: str
    model_name: str
    dataset_name: str
    dataset_split: str
    seed: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "repo_sha": self.repo_sha,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "seed": self.seed,
        }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_metadata(metadata: RunMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True))


def resolve_repo_sha() -> str:
    head = os.environ.get("GIT_COMMIT_SHA")
    if head:
        return head
    git_head = Path(".git/HEAD")
    if not git_head.exists():
        return "unknown"
    content = git_head.read_text().strip()
    if content.startswith("ref:"):
        ref_path = Path(".git") / content.split(" ", 1)[1]
        return ref_path.read_text().strip()
    return content


__all__ = ["RunMetadata", "set_global_seed", "save_metadata", "resolve_repo_sha"]
