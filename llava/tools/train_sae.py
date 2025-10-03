import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sae.sparse_autoencoder import SparseAutoencoder


class NumpyDeltaDataset(Dataset):
    def __init__(self, deltas: np.ndarray):
        self.x = torch.from_numpy(deltas.astype(np.float32))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


def train_sae_on_layer(deltas: np.ndarray, latent_dim: int, epochs: int, batch_size: int, lr: float, l1_coeff: float, device: str):
    if deltas.ndim != 2:
        deltas = deltas.reshape(deltas.shape[0], -1)
    input_dim = deltas.shape[1]

    dataset = NumpyDeltaDataset(deltas)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    metrics = {"epoch": [], "loss": [], "mse": [], "l1": []}

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_mse = 0.0
        total_l1 = 0.0
        steps = 0
        for batch in loader:
            batch = batch.to(device)
            optim.zero_grad(set_to_none=True)
            recon, z = model(batch)
            loss, parts = SparseAutoencoder.loss(recon, batch, z, l1_coeff=l1_coeff)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            total_mse += parts["mse"]
            total_l1 += parts["l1"]
            steps += 1
        metrics["epoch"].append(epoch)
        metrics["loss"].append(total_loss / max(steps, 1))
        metrics["mse"].append(total_mse / max(steps, 1))
        metrics["l1"].append(total_l1 / max(steps, 1))
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"Epoch {epoch}: loss={metrics['loss'][-1]:.6f} mse={metrics['mse'][-1]:.6f} l1={metrics['l1'][-1]:.6f}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoders on h_on - h_off deltas")
    parser.add_argument("--input_dir", required=True, help="Directory containing layer_*_delta.npy (SIUO deltas)")
    parser.add_argument("--output_dir", default="artifacts/sae", help="Where to save trained SAEs")
    parser.add_argument("--latent_ratio", type=float, default=4.0, help="Latent dim = ratio * input dim")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all layer deltas
    layer_files = sorted(input_dir.glob("layer_*_delta.npy"))
    if not layer_files:
        raise FileNotFoundError(f"No deltas found in {input_dir}")

    summary: Dict[str, Dict] = {}

    for path in layer_files:
        layer = int(path.stem.split("_")[1])
        print(f"Training SAE for layer {layer}...")
        deltas = np.load(path)
        if deltas.size == 0:
            print(f"Skipping layer {layer}: empty deltas")
            continue

        if deltas.ndim != 2:
            deltas = deltas.reshape(deltas.shape[0], -1)
        input_dim = deltas.shape[1]
        # Use float32 for stable statistics (avoid float16 overflow)
        deltas = deltas.astype(np.float32, copy=False)
        # Standardize per-feature
        mean = deltas.mean(axis=0)
        std = deltas.std(axis=0) + 1e-6
        deltas = (deltas - mean) / std
        latent_dim = int(max(1, round(args.latent_ratio * input_dim)))

        model, metrics = train_sae_on_layer(
            deltas,
            latent_dim=latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            l1_coeff=args.l1,
            device=args.device,
        )

        # Save weights
        layer_dir = output_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), layer_dir / "sae.pt")
        config = {
            "layer": layer,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "l1": args.l1,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
        with open(layer_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        with open(layer_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved SAE for layer {layer} to {layer_dir}")

        summary[str(layer)] = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "final_loss": metrics["loss"][-1] if metrics["loss"] else None,
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"All done. Summary written to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()


