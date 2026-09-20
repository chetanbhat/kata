"""EXERCISE 8.1 — VAE: reparameterization + ELBO with a beta (annealing) knob.

Goals: closed-form Gaussian KL; differentiable rsample; loss = recon + beta*KL;
10 training steps must lower the ELBO on synthetic binary data.

Run: pytest tests/test_08_generative.py -k "vae or kl or reparameterize or elbo"
"""

import torch
import torch.nn as nn


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # TODO: -0.5*sum(1 + lv - mu^2 - e^lv) per sample.
    raise NotImplementedError("# TODO: Implement this")


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # TODO: mu + eps * exp(0.5*logvar), eps ~ N(0, I).
    raise NotImplementedError("# TODO: Implement this")


class VAE(nn.Module):
    def __init__(self, x_dim: int, h_dim: int = 64, z_dim: int = 8):
        super().__init__()
        # TODO: Encoder MLP -> mu/logvar heads; decoder MLP to logits.
        raise NotImplementedError("# TODO: Implement this")

    def elbo(self, x: torch.Tensor, beta: float = 1.0) -> dict:
        # TODO: Return {'loss', 'recon', 'kl'}; loss = recon + beta*kl.
        raise NotImplementedError("# TODO: Implement this")
