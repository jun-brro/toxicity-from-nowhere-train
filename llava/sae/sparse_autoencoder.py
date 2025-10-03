import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Simple overcomplete Sparse Autoencoder.

    - Encoder: Linear -> ReLU (non-negative sparse code)
    - Decoder: Linear
    - Loss: MSE reconstruction + L1 sparsity on latent code
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Linear(latent_dim, input_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    @staticmethod
    def loss(recon: torch.Tensor, x: torch.Tensor, z: torch.Tensor, l1_coeff: float = 1e-3):
        """Compute SAE loss components.

        Args:
            recon: reconstructed input
            x: original input
            z: latent code
            l1_coeff: L1 sparsity weight
        Returns:
            total_loss, metrics_dict
        """
        mse = torch.mean((recon - x) ** 2)
        l1 = torch.mean(torch.abs(z))
        total = mse + l1_coeff * l1
        return total, {"mse": mse.detach().item(), "l1": l1.detach().item()}


