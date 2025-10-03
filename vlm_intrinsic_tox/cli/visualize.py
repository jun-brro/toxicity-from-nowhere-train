"""CLI for SAE visualization and analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset

from ..utils.config import load_config
from ..utils.logging import configure_logging, get_logger
from ..utils.visualization import plot_tsne_latents, plot_top_latents_summary
from ..sae.io import load_sae_model, load_scaler
from ..eval.metrics import compute_all_metrics
from ..data.registry import get_dataset

LOGGER = get_logger(__name__)


def load_meme_safety_bench(max_samples: int = 500, balanced: bool = True) -> list[dict]:
    """Load Meme Safety Bench dataset for visualization.
    
    Args:
        max_samples: Maximum number of samples to load
        balanced: Whether to balance harmful/benign samples
        
    Returns:
        List of records with image, text, label
    """
    ds = load_dataset("oneonlee/Meme-Safety-Bench", "full")
    ds = ds["test"]

    positives = []  # benign label=0
    negatives = []  # toxic label=1
    
    for ex in ds:
        img = ex["meme_image"]
        txt = ex["instruction"]
        sentiment = str(ex.get("sentiment", "")).strip().lower()
        lbl = 1 if sentiment == "negative" else 0
        rec = {"image": img, "text": txt, "label": lbl}
        (negatives if lbl == 1 else positives).append(rec)

    if balanced and max_samples:
        k = max_samples // 2
        rng = np.random.default_rng(0)
        pos_sel = positives if len(positives) <= k else list(rng.choice(positives, k, replace=False))
        neg_sel = negatives if len(negatives) <= k else list(rng.choice(negatives, k, replace=False))
        records = neg_sel + pos_sel
        rng.shuffle(records)
    else:
        records = negatives + positives
        if max_samples and len(records) > max_samples:
            records = records[:max_samples]

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SAE latents and features")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--layer", type=int, required=True, help="Layer to visualize")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples for visualization")
    parser.add_argument("--output-dir", type=Path, default="artifacts/visualizations", help="Output directory")
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE visualization")
    parser.add_argument("--metrics", action="store_true", help="Generate metrics summary")
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    cfg = load_config([args.config], overrides=args.overrides)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SAE model and scaler
    sae_path = Path(cfg.io.sae_dir) / f"sae_layer{args.layer}.pt"
    scaler_path = Path(cfg.io.sae_dir) / f"scaler_layer{args.layer}.json"
    
    if not sae_path.exists():
        LOGGER.error(f"SAE model not found: {sae_path}")
        return
    
    if not scaler_path.exists():
        LOGGER.error(f"Scaler not found: {scaler_path}")
        return
    
    sae_model = load_sae_model(sae_path)
    scaler = load_scaler(scaler_path)
    
    LOGGER.info(f"Loaded SAE model for layer {args.layer}")
    
    # Load activation data
    activ_dir = Path(cfg.io.activ_dir)
    shard_files = list(activ_dir.glob(f"*layer_{args.layer}*.npz"))
    
    if not shard_files:
        LOGGER.error(f"No activation shards found for layer {args.layer} in {activ_dir}")
        return
    
    # Load and concatenate activations
    all_deltas = []
    all_labels = []
    
    for shard_file in shard_files[:5]:  # Limit to first 5 shards for visualization
        data = np.load(shard_file)
        deltas = data["delta"]
        labels = data.get("labels", np.zeros(len(deltas)))  # Default to benign if no labels
        
        all_deltas.append(deltas)
        all_labels.append(labels)
    
    deltas = np.vstack(all_deltas)
    labels = np.concatenate(all_labels)
    
    # Limit samples for visualization
    if len(deltas) > args.max_samples:
        indices = np.random.choice(len(deltas), args.max_samples, replace=False)
        deltas = deltas[indices]
        labels = labels[indices]
    
    LOGGER.info(f"Loaded {len(deltas)} samples for visualization")
    
    # Standardize deltas
    deltas_std = (deltas - scaler["mean"]) / scaler["std"]
    
    # Encode with SAE
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_model = sae_model.to(device)
    
    with torch.no_grad():
        deltas_tensor = torch.from_numpy(deltas_std.astype(np.float32)).to(device)
        latents = sae_model.encode(deltas_tensor).cpu().numpy()
    
    LOGGER.info(f"Encoded to latent space: {latents.shape}")
    
    # Generate visualizations
    if args.tsne:
        LOGGER.info("Generating t-SNE visualization...")
        plot_tsne_latents(
            latents=latents,
            labels=labels,
            title=f"t-SNE of SAE Latents (Layer {args.layer})",
            save_path=output_dir / f"tsne_layer_{args.layer}.png"
        )
    
    if args.metrics:
        LOGGER.info("Computing metrics and generating summary...")
        
        # Compute metrics
        metrics = compute_all_metrics(latents, labels)
        
        # Plot top latents summary
        plot_top_latents_summary(
            metrics_dict=metrics,
            top_k=20,
            save_path=output_dir / f"top_latents_layer_{args.layer}.png"
        )
        
        # Save metrics to file
        metrics_file = output_dir / f"metrics_layer_{args.layer}.npz"
        np.savez(metrics_file, **metrics)
        LOGGER.info(f"Saved metrics to {metrics_file}")
    
    LOGGER.info(f"Visualization complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()