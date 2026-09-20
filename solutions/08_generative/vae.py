"""Generative history I: VAEs (2013) -- reparameterization + ELBO.

History: exact posterior inference is intractable -> maximize the ELBO with
the reparameterization trick (Kingma & Welling; Rezende et al., 2013).
The famous wrong turn: POSTERIOR COLLAPSE -- with a powerful decoder the KL
term goes to zero and z is ignored (Bowman et al. 2016). Remedies explored
since: KL annealing (beta-VAE schedules), free bits, lagging inference.
The `beta` knob below is exactly that annealing handle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma) || N(0, I)) per sample: -0.5*sum(1+lv-mu^2-e^lv)."""
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return mu + torch.randn_like(logvar) * (0.5 * logvar).exp()


class VAE(nn.Module):
    def __init__(self, x_dim: int, h_dim: int = 64, z_dim: int = 8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.to_mu = nn.Linear(h_dim, z_dim)
        self.to_logvar = nn.Linear(h_dim, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, x_dim))  # Bernoulli logits

    def elbo(self, x: torch.Tensor, beta: float = 1.0) -> dict:
        h = self.enc(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        z = reparameterize(mu, logvar)
        recon = F.binary_cross_entropy_with_logits(self.dec(z), x, reduction="none")
        recon = recon.sum(dim=-1).mean()
        kl = kl_normal(mu, logvar).mean()
        loss = recon + beta * kl
        return {"loss": loss, "recon": recon.detach(), "kl": kl.detach()}
