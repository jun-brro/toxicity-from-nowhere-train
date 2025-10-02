"""Configuration dataclasses and loader for the intrinsic toxicity pipeline."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import yaml


@dataclasses.dataclass
class DatasetConfig:
    name: str = "oneonlee/Meme-Safety-Bench"
    split: str = "test"
    image_column: str = "meme_image"
    instruction_column: str = "instruction"
    label_column: Optional[str] = None
    streaming: bool = False
    prompt_template: str = "<image>\n{instruction}"
    label_mapping_path: Optional[str] = None


@dataclasses.dataclass
class ExtractLayerConfig:
    layers: Sequence[int] = dataclasses.field(default_factory=lambda: [0, 2, 4])
    pooling: str = "text_only_mean"
    text_span: str = "content"
    pooling_kwargs: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ExtractConfig:
    model_name: str = "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"
    device: str = "cuda"
    batch_size: int = 1
    num_workers: int = 0
    mixed_precision: str = "bf16"
    output_dir: str = "artifacts/extract"
    layers: ExtractLayerConfig = dataclasses.field(default_factory=ExtractLayerConfig)
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    seed: int = 17
    shards_per_file: int = 1024


@dataclasses.dataclass
class SAETrainConfig:
    latent_ratio: float = 4.0
    topk: int = 64
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 10
    weight_decay: float = 0.0
    val_fraction: float = 0.05
    output_dir: str = "artifacts/sae"
    device: str = "cuda"
    seed: int = 17


@dataclasses.dataclass
class EvalConfig:
    topk_latents: int = 50
    output_dir: str = "artifacts/eval"
    intervention_enabled: bool = True
    weights: Optional[dict] = None
    layers: Sequence[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RootConfig:
    extract: ExtractConfig = dataclasses.field(default_factory=ExtractConfig)
    sae: SAETrainConfig = dataclasses.field(default_factory=SAETrainConfig)
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True))


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(paths: Iterable[Path]) -> RootConfig:
    """Load configuration from one or more YAML files."""

    merged: dict = {}
    for path in paths:
        merged = _merge_dicts(merged, _load_yaml(path))
    data = _merge_dicts(dataclasses.asdict(RootConfig()), merged)
    return _dict_to_config(data)


def _merge_dicts(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            result[key] = _merge_dicts(base[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(data: dict) -> RootConfig:
    extract = ExtractConfig(
        **{**data.get("extract", {}), "layers": ExtractLayerConfig(**data.get("extract", {}).get("layers", {}))}
    )
    sae = SAETrainConfig(**data.get("sae", {}))
    eval_cfg = EvalConfig(**data.get("eval", {}))
    return RootConfig(extract=extract, sae=sae, eval=eval_cfg)


__all__ = [
    "DatasetConfig",
    "ExtractLayerConfig",
    "ExtractConfig",
    "SAETrainConfig",
    "EvalConfig",
    "RootConfig",
    "load_config",
]
