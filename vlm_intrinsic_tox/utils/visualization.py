"""Visualization utilities for SAE analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def plot_tsne_latents(
    latents: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE Visualization of SAE Latents",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30,
    random_state: int = 42
) -> None:
    """Create t-SNE visualization of SAE latents colored by labels.
    
    Args:
        latents: Array of shape [N, latent_dim] containing SAE latent activations
        labels: Array of shape [N] containing binary labels (0=benign, 1=harmful)
        title: Plot title
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
    """
    LOGGER.info(f"Computing t-SNE with perplexity={perplexity}")
    
    # Compute t-SNE embedding
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    embedding = tsne.fit_transform(latents)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points colored by label
    benign_mask = labels == 0
    harmful_mask = labels == 1
    
    if np.any(benign_mask):
        ax.scatter(
            embedding[benign_mask, 0], 
            embedding[benign_mask, 1],
            c='blue', 
            alpha=0.6, 
            label=f'Benign (n={np.sum(benign_mask)})',
            s=20
        )
    
    if np.any(harmful_mask):
        ax.scatter(
            embedding[harmful_mask, 0], 
            embedding[harmful_mask, 1],
            c='red', 
            alpha=0.6, 
            label=f'Harmful (n={np.sum(harmful_mask)})',
            s=20
        )
    
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        LOGGER.info(f"Saved t-SNE plot to {save_path}")
    
    plt.show()


def plot_latent_activation_distribution(
    latents: np.ndarray,
    labels: np.ndarray,
    latent_idx: int,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """Plot distribution of a specific latent's activations for harmful vs benign samples.
    
    Args:
        latents: Array of shape [N, latent_dim] containing SAE latent activations
        labels: Array of shape [N] containing binary labels (0=benign, 1=harmful)
        latent_idx: Index of the latent to visualize
        title: Optional plot title
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    if title is None:
        title = f"Latent {latent_idx} Activation Distribution"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    benign_activations = latents[labels == 0, latent_idx]
    harmful_activations = latents[labels == 1, latent_idx]
    
    # Histogram
    ax1.hist(benign_activations, bins=50, alpha=0.7, label='Benign', color='blue', density=True)
    ax1.hist(harmful_activations, bins=50, alpha=0.7, label='Harmful', color='red', density=True)
    ax1.set_xlabel('Activation Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title} - Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot([benign_activations, harmful_activations], 
                labels=['Benign', 'Harmful'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='black', linewidth=2))
    ax2.set_ylabel('Activation Value')
    ax2.set_title(f'{title} - Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        LOGGER.info(f"Saved activation distribution plot to {save_path}")
    
    plt.show()


def plot_top_latents_summary(
    metrics_dict: Dict[str, np.ndarray],
    top_k: int = 20,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """Plot summary of top latents across different metrics.
    
    Args:
        metrics_dict: Dictionary with metric names as keys and arrays of per-latent scores as values
        top_k: Number of top latents to show
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    n_metrics = len(metrics_dict)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif n_metrics <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (metric_name, scores) in enumerate(metrics_dict.items()):
        ax = axes[i]
        
        # Get top-k latents for this metric
        top_indices = np.argsort(np.abs(scores))[-top_k:][::-1]
        top_scores = scores[top_indices]
        
        # Create bar plot
        bars = ax.bar(range(len(top_scores)), top_scores, 
                     color='red' if np.mean(top_scores) > 0 else 'blue',
                     alpha=0.7)
        
        ax.set_title(f'Top {top_k} Latents by {metric_name}')
        ax.set_xlabel('Latent Rank')
        ax.set_ylabel(f'{metric_name} Score')
        ax.grid(True, alpha=0.3)
        
        # Add latent indices as x-tick labels
        ax.set_xticks(range(0, len(top_scores), max(1, len(top_scores) // 10)))
        ax.set_xticklabels([str(top_indices[j]) for j in range(0, len(top_scores), max(1, len(top_scores) // 10))],
                          rotation=45)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        LOGGER.info(f"Saved top latents summary to {save_path}")
    
    plt.show()


__all__ = [
    "plot_tsne_latents",
    "plot_latent_activation_distribution", 
    "plot_top_latents_summary"
]