"""Generative history II: GANs (2014) -- minimax, non-saturating, WGAN-GP.

History of wrong turns: the minimax generator loss saturates early (log(1-D)
flat when D is confident) -> non-saturating -log D(G) (same paper, pragmatic
fix). Mode collapse -> WGAN (2017) replaces JS with Wasserstein + weight
clipping (crude: cripples capacity) -> WGAN-GP (2017) penalizes gradient
norms instead. SpectralNorm discriminators (2018, cf. module 06) stabilize
further. This file keeps every variant as pure loss functions + a small SN
helper so each is exactly testable.
"""

import torch
import torch.nn.functional as F


def gan_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Minimax discriminator loss: -(log D(x) + log(1 - D(G(z))))."""
    return -(F.logsigmoid(real_logits) + F.logsigmoid(-fake_logits)).mean()


def gan_g_loss_minimax(fake_logits: torch.Tensor) -> torch.Tensor:
    """Original saturating generator loss: log(1 - D(G(z)))."""
    return F.logsigmoid(-fake_logits).mean()


def gan_g_loss_nonsaturating(fake_logits: torch.Tensor) -> torch.Tensor:
    """Practical fix: maximize log D(G(z)) instead."""
    return -F.logsigmoid(fake_logits).mean()


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    interp = real + torch.rand_like(real) * (fake - real)
    interp.requires_grad_(True)
    scores = critic(interp).sum()
    grads = torch.autograd.grad(scores, interp, create_graph=True)[0]
    return ((grads.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()


def wgan_gp_d_loss(critic, real: torch.Tensor, fake: torch.Tensor,
                   lam: float = 10.0) -> torch.Tensor:
    return critic(fake).mean() - critic(real).mean() + lam * gradient_penalty(critic, real, fake)


def wgan_g_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.mean()


def spectral_normalize(weight: torch.Tensor, u: torch.Tensor, steps: int = 5):
    """Power iteration; returns (W/sigma, updated u, sigma)."""
    with torch.no_grad():
        for _ in range(steps):
            v = (weight.t() @ u)
            v = v / v.norm().clamp_min(1e-12)
            u = (weight @ v)
            u = u / u.norm().clamp_min(1e-12)
    sigma = (u @ weight @ v).detach()
    return weight / sigma.clamp_min(1e-12), u, sigma
