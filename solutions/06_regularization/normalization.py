"""Normalization beyond BatchNorm: WeightNorm, SpectralNorm, GradientPenalty.

History of wrong turns: BatchNorm's "internal covariate shift" story (2015)
did not survive scrutiny (Santurkar et al. 2018: it smooths the landscape).
WeightNorm (2016) reparameterizes w = g*v/||v|| to decouple norm/direction.
SpectralNorm (2018) divides by the top singular value (power iteration) to
enforce 1-Lipschitz critics -- replacing WGAN's weight clipping (2017),
which underused capacity and was superseded by gradient penalty (2017).
"""

import torch
import torch.nn as nn


class WeightNormLinear(nn.Module):
    """w = g * v / ||v||: learnable per-row norm g, direction v."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(out_features))
        self.v = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.v, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def weight(self) -> torch.Tensor:
        return self.g[:, None] * self.v / self.v.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight().t() + self.bias


class SpectralNormLinear(nn.Module):
    """W_sn = W / sigma_max(W); sigma estimated by power iteration on buffer u."""

    def __init__(self, in_features: int, out_features: int, n_power: int = 5):
        super().__init__()
        self.n_power = int(n_power)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("u", torch.randn(out_features))

    def sigma(self) -> torch.Tensor:
        u = self.u / self.u.norm().clamp_min(1e-12)
        with torch.no_grad():
            for _ in range(self.n_power if self.training else 1):
                v = (self.weight.t() @ u)
                v = v / v.norm().clamp_min(1e-12)
                u = (self.weight @ v)
                u = u / u.norm().clamp_min(1e-12)
            if self.training:
                self.u.copy_(u)
        return u @ self.weight @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ (self.weight / self.sigma().clamp_min(1e-12)).t() + self.bias


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """WGAN-GP penalty: mean ((||grad_x C(interp)||_2 - 1)^2)."""
    alpha = torch.rand(real.shape[0], *([1] * (real.dim() - 1)), device=real.device)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = critic(interp).sum()
    grads = torch.autograd.grad(scores, interp, create_graph=True)[0]
    return ((grads.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()
