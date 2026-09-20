"""EXERCISE 6.2 — WeightNorm, SpectralNorm, Gradient Penalty.

Goals: reparameterized row norms (||w_i|| must equal g_i exactly); power
iteration matching SVD's top singular value; GP exact on a linear critic.

Run: pytest tests/test_06_regularization.py -k "weightnorm or spectral or penalty or norm"
Reading: readings/06_regularization/normalization.md
"""

import torch
import torch.nn as nn


class WeightNormLinear(nn.Module):
    """w = g * v / ||v||."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # TODO: g/v/bias params with direction init on v.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")


class SpectralNormLinear(nn.Module):
    """Divide by sigma_max(W) via power iteration on buffer u."""

    def __init__(self, in_features: int, out_features: int, n_power: int = 5):
        super().__init__()
        # TODO: weight/bias params + u buffer.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    # TODO: mean((||grad interp||_2 - 1)^2) over random interpolations.
    raise NotImplementedError("# TODO: Implement this")
