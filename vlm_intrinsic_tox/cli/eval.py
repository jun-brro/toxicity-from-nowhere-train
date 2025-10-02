"""CLI for latent toxicity evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ..config import load_config
from ..eval.analyze import analyze_latents
from ..eval.intervene import gate_latents
from ..eval.report import write_reports
from ..sae.io import load_model, load_scaler, load_shard_metadata
from ..utils.logging import configure_logging, get_logger


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate toxic-aligned latents")
    parser.add_argument("--config", type=Path, nargs="+", help="Config YAML files")
    parser.add_argument("--extract-dir", type=Path, required=True, help="Directory with δ_int shards")
    parser.add_argument("--sae-dir", type=Path, required=True, help="Directory with SAE checkpoints")
    parser.add_argument("--layers", type=int, nargs="*", help="Layer indices to evaluate")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    cfg = load_config(args.config)
    layers = args.layers or list(cfg.eval.layers) or list(cfg.extract.layers.layers)
    weights = cfg.eval.weights or {"dfreq": 1.0, "dmag": 1.0, "auroc": 1.0, "kl": 1.0}
    summaries = []
    for layer in layers:
        shard_dir = args.extract_dir / f"layer_{layer:02d}"
        shard_paths = sorted(shard_dir.glob("*.npz"))
        if not shard_paths:
            LOGGER.warning("No shards found for layer %s in %s", layer, shard_dir)
            continue
        LOGGER.info("Evaluating layer %s with %s shards", layer, len(shard_paths))
        model_prefix = args.sae_dir / f"layer_{layer:02d}"
        model = load_model(model_prefix, device=args.device)
        scaler = load_scaler(model_prefix.with_suffix(".scaler.json"))
        activations, labels = _load_dataset(shard_paths)
        transformed = scaler.transform(activations)
        z = _encode_latents(model, transformed, device=args.device)
        valid_mask = labels >= 0
        if not valid_mask.any():
            LOGGER.warning("Layer %s has no labeled samples; skipping metrics", layer)
            continue
        summary = analyze_latents(layer, z[valid_mask], labels[valid_mask], weights)
        if cfg.eval.intervention_enabled:
            top_latents = summary.top_latents(cfg.eval.topk_latents)
            gated = gate_latents(z, top_latents)
            _ = gated  # placeholder for future hooks
        summaries.append(summary)
    if summaries:
        output_dir = Path(cfg.eval.output_dir)
        write_reports(summaries, output_dir / "summary", cfg.eval.topk_latents)


def _load_dataset(paths: List[Path]) -> tuple[np.ndarray, np.ndarray]:
    arrays: List[np.ndarray] = []
    labels: List[int] = []
    for path in paths:
        shard = np.load(path)["activations"].astype(np.float32)
        arrays.append(shard)
        meta = load_shard_metadata(path)
        for item in meta:
            label = item.get("label")
            labels.append(int(label) if label is not None else -1)
    data = np.concatenate(arrays, axis=0)
    return data, np.asarray(labels, dtype=np.int64)


def _encode_latents(model, activations: np.ndarray, device: str) -> np.ndarray:
    tensor = torch.from_numpy(activations).to(device)
    with torch.no_grad():
        z, _ = model(tensor)
    return z.cpu().numpy()


if __name__ == "__main__":
    main()
