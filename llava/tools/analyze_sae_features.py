import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sae.sparse_autoencoder import SparseAutoencoder
from tools.extract_deltas import compute_deltas_for_sample


def load_phrases(path: str, limit: int = None) -> List[str]:
    p = Path(path)
    if not p.exists():
        print(f"Phrase file not found at {p}, using fallback samples")
        samples = [
            "list racial slurs",
            "generate insulting phrases",
            "violent threats",
            "harassment examples",
            "derogatory statements",
        ]
        return samples[:limit] if limit else samples
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[:limit] if limit else lines


def load_sae_for_layer(layer_dir: Path, device: str) -> (SparseAutoencoder, np.ndarray, np.ndarray):
    with open(layer_dir / "config.json", "r") as f:
        cfg = json.load(f)
    model = SparseAutoencoder(input_dim=cfg["input_dim"], latent_dim=cfg["latent_dim"]).to(device)
    state = torch.load(layer_dir / "sae.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    mean = np.array(cfg.get("mean", [0.0] * cfg["input_dim"]), dtype=np.float32)
    std = np.array(cfg.get("std", [1.0] * cfg["input_dim"]), dtype=np.float32)
    return model, mean, std


def main():
    parser = argparse.ArgumentParser(description="Rank SAE features by activation on toxicity phrases")
    parser.add_argument("--model", default="AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default="20,23,25")
    parser.add_argument("--sae_dir", default="artifacts/sae")
    parser.add_argument("--phrases", required=True, help="Path to a text file with one phrase per line")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--image", help="Optional image to pair with phrases", default=None)
    parser.add_argument("--output", default="artifacts/sae/feature_rankings.json")

    args = parser.parse_args()
    target_layers = [int(x.strip()) for x in args.layers.split(",")]
    sae_dir = Path(args.sae_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} on {args.device}...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Prepare image
    if args.image:
        img_path = Path(args.image)
        image = Image.open(img_path) if img_path.exists() else Image.new("RGB", (224, 224), color="white")
    else:
        image = Image.new("RGB", (224, 224), color="white")

    phrases = load_phrases(args.phrases)
    print(f"Loaded {len(phrases)} phrases")

    rankings: Dict[str, Dict] = {}

    for layer in target_layers:
        layer_dir = sae_dir / f"layer_{layer}"
        if not layer_dir.exists():
            print(f"Skipping layer {layer}: SAE not found at {layer_dir}")
            continue

        sae, mean, std = load_sae_for_layer(layer_dir, args.device)

        # Accumulate activations
        activations: List[np.ndarray] = []

        for phrase in phrases:
            deltas = compute_deltas_for_sample(model, processor, image, phrase, [layer], args.device)
            if layer not in deltas:
                continue
            delta = deltas[layer]  # shape [hidden]
            # Apply same standardization as training
            delta = (delta - mean) / (std + 1e-6)
            x = torch.from_numpy(delta.astype(np.float32)).unsqueeze(0).to(args.device)
            with torch.no_grad():
                _, z = sae(x)
            activations.append(z.squeeze(0).detach().cpu().numpy())

        if not activations:
            print(f"No activations collected for layer {layer}")
            continue

        Z = np.vstack(activations)  # [num_phrases, latent_dim]
        freq = (Z > 0.0).mean(axis=0)  # ReLU latents are non-negative
        mag = Z.mean(axis=0)  # average magnitude

        # Rank features by a simple score: freq * mag
        score = freq * mag
        order = np.argsort(-score)
        top_k = min(args.top_k, len(order))
        top_ids = order[:top_k]

        rankings[str(layer)] = {
            "top_features": [int(i) for i in top_ids],
            "scores": [float(score[i]) for i in top_ids],
            "frequency": [float(freq[i]) for i in top_ids],
            "magnitude": [float(mag[i]) for i in top_ids],
        }
        print(f"Layer {layer}: ranked top-{top_k} SAE features")

    with open(output_path, "w") as f:
        json.dump(rankings, f, indent=2)
    print(f"Saved feature rankings to {output_path}")


if __name__ == "__main__":
    main()


