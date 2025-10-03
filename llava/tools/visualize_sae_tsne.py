import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
from tools.analyze_sae_features import load_sae_for_layer
from tools.extract_deltas import compute_deltas_for_sample
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from datasets import load_dataset


def load_meme_safety_bench(dataset_path: str = "", split: str = "test", max_samples: int = 500, balanced: bool = True):

    ds = load_dataset("oneonlee/Meme-Safety-Bench", "full")
    ds = ds[split]

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


def compute_layer_zs(model, processor, records: List[Dict], layer: int, sae_dir: Path, device: str, log_every: int = 10) -> (np.ndarray, np.ndarray):
    sae, mean, std = load_sae_for_layer(sae_dir / f"layer_{layer}", device)
    zs = []
    ys = []
    for i, r in enumerate(records, 1):
        img = r["image"]
        text = r["text"]
        deltas = compute_deltas_for_sample(model, processor, img, text, [layer], device)
        if layer not in deltas:
            continue
        delta = deltas[layer].astype(np.float32)
        x = (delta - mean.astype(np.float32)) / (std.astype(np.float32) + 1e-6)
        t = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            _, z = sae(t)
        zs.append(z.squeeze(0).cpu().numpy())
        ys.append(int(r["label"]))
        if log_every and (i % log_every == 0):
            print(f"Layer {layer}: processed {i}/{len(records)} samples")
    if not zs:
        return np.empty((0,)), np.empty((0,))
    return np.vstack(zs), np.array(ys)


def plot_tsne(Z: np.ndarray, y: np.ndarray, title: str, out_png: Path, perplexity: float = 30.0, random_state: int = 0):
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state)
    X2 = tsne.fit_transform(Z)
    plt.figure(figsize=(6, 5))
    colors = np.where(y == 1, "red", "blue")
    labels = np.where(y == 1, "toxic", "benign")
    plt.scatter(X2[:, 0], X2[:, 1], c=colors, s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Visualize SAE latent codes with t-SNE per layer")
    ap.add_argument("--model", default="AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", default="0,2,4")
    ap.add_argument("--sae_dir", default="/home/pljh0906/tcav/llava/artifacts/sae")
    ap.add_argument("--dataset_path", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_samples", type=int, default=400)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--output_dir", default="/home/pljh0906/tcav/tsne_plots")

    args = ap.parse_args()

    target_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    records = load_meme_safety_bench(args.dataset_path, args.split, args.max_samples, balanced=args.balanced)
    # Brief dataset summary
    labels = [int(r["label"]) for r in records]
    pos = sum(1 for v in labels if v == 1)
    neg = len(labels) - pos
    print(f"Dataset: total={len(records)}, harmful={pos}, benign={neg}, balanced={args.balanced}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for L in target_layers:
        print(f"Computing SAE codes: layer={L}...")
        Z, y = compute_layer_zs(model, processor, records, L, Path(args.sae_dir), args.device, log_every=args.log_every)
        if Z.size == 0:
            print(f"Layer {L}: no codes computed, skipping")
            continue
        # optional: keep only top-variance components in z-space to stabilize t-SNE
        var = Z.var(axis=0)
        top_idx = np.argsort(-var)[:min(512, Z.shape[1])]
        Z_reduced = Z[:, top_idx]
        print(f"t-SNE: layer={L}, N={Z.shape[0]}, D={Z.shape[1]} (reduced to {Z_reduced.shape[1]}), perplexity={args.perplexity}")
        png_path = out_dir / f"tsne_layer_{L}.png"
        plot_tsne(Z_reduced, y, title=f"Layer {L} SAE z (t-SNE)", out_png=png_path, perplexity=args.perplexity)
        print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
