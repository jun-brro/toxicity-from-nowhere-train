"""Extraction runner for δ_int activations."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration

from ..config import DatasetConfig, ExtractConfig
from ..data.datasets import Labeler, MemeSafetyBenchAdapter, load_label_mapping
from ..utils.env import RunMetadata, resolve_repo_sha, save_metadata, set_global_seed
from ..utils.logging import get_logger
from .hooks import ResidualCapture, cross_attention_off, install_post_attention_hook, remove_hooks
from .pooling import POOLING_FNS
from .sharding import ShardWriter


LOGGER = get_logger(__name__)


def load_image(image_ref) -> Image.Image:
    if isinstance(image_ref, Image.Image):
        return image_ref
    if isinstance(image_ref, dict):
        if "path" in image_ref and image_ref["path"]:
            return Image.open(image_ref["path"]).convert("RGB")
        if "bytes" in image_ref:
            return Image.open(BytesIO(image_ref["bytes"])).convert("RGB")
    return Image.open(image_ref).convert("RGB")


class ExtractRunner:
    def __init__(self, cfg: ExtractConfig) -> None:
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model: LlavaForConditionalGeneration = LlavaForConditionalGeneration.from_pretrained(
            cfg.model_name, torch_dtype=_mixed_precision_dtype(cfg.mixed_precision), device_map="auto"
        )
        self.model.eval()
        set_global_seed(cfg.seed)

    def run(self) -> None:
        dataset_cfg = self.cfg.dataset
        label_mapping = load_label_mapping(dataset_cfg.label_mapping_path)
        labeler = Labeler(label_mapping, dataset_cfg.label_column)
        adapter = MemeSafetyBenchAdapter(
            dataset_cfg.name,
            dataset_cfg.split,
            dataset_cfg.image_column,
            dataset_cfg.instruction_column,
            dataset_cfg.streaming,
        )
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = RunMetadata(
            repo_sha=resolve_repo_sha(),
            model_name=self.cfg.model_name,
            dataset_name=dataset_cfg.name,
            dataset_split=dataset_cfg.split,
            seed=self.cfg.seed,
        )
        save_metadata(metadata, output_dir / "metadata.json")
        for layer_idx in self.cfg.layers.layers:
            shard_writer = ShardWriter(output_dir / f"layer_{layer_idx:02d}", layer_idx, self.cfg.shards_per_file)
            capture = ResidualCapture()
            handles = install_post_attention_hook(self.model, [layer_idx], capture)
            try:
                for sample in adapter.iter_samples(self.cfg.dataset.prompt_template, labeler):
                    delta = self._extract_delta(sample, layer_idx, capture)
                    shard_writer.add(delta, self._sanitize_metadata(sample))
                    capture.clear()
            finally:
                shard_writer.close()
                remove_hooks(handles)

    def _extract_delta(self, sample, layer_idx: int, capture: ResidualCapture) -> np.ndarray:
        image = load_image(sample.image)
        inputs = self.processor(images=image, text=sample.prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            _ = self.model(**inputs)
            on_hidden = capture.activations[layer_idx]
        with torch.no_grad():
            decoder_layer = self.model.language_model.layers[layer_idx]
            with cross_attention_off(decoder_layer):
                _ = self.model(**inputs)
                off_hidden = capture.activations[layer_idx]
        delta = (on_hidden - off_hidden).to(torch.float32)
        pooling_fn = POOLING_FNS[self.cfg.layers.pooling]
        input_ids = inputs["input_ids"]
        image_token = getattr(self.processor.tokenizer, "image_token", None)
        image_token_id = None
        if image_token is not None:
            image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        content_indices = [i for i, token in enumerate(input_ids[0].tolist()) if token != image_token_id]
        if not content_indices:
            content_indices = list(range(input_ids.shape[1]))
        pooled = pooling_fn(delta.unsqueeze(0), content_indices, **self.cfg.layers.pooling_kwargs)
        return pooled.squeeze(0).cpu().numpy()

    def _sanitize_metadata(self, sample) -> Dict[str, object]:
        payload: Dict[str, object] = {"label": sample.label}
        for key, value in sample.metadata.items():
            if key == self.cfg.dataset.image_column:
                continue
            payload[key] = _to_jsonable(value)
        return payload


def _mixed_precision_dtype(mode: str) -> torch.dtype:
    mode = mode.lower()
    if mode in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if mode in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


__all__ = ["ExtractRunner"]


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)
