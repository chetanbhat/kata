"""EXERCISE 8.2 — GAN losses: minimax vs non-saturating vs WGAN-GP + SN.

Goals: each loss as a pure function with exact hand-checkable values;
gradient penalty exact on a linear critic; power iteration matching SVD.

Run: pytest tests/test_08_generative.py -k "gan or wgan or spectral_normalize or penalty"
Reading: readings/08_generative/gan.md
"""

import torch


def gan_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    # TODO: -(log D(x) + log(1 - D(G(z)))).
    raise NotImplementedError("# TODO: Implement this")


def gan_g_loss_minimax(fake_logits: torch.Tensor):
    # TODO: Saturating original: log(1 - D(G(z))).
    raise NotImplementedError("# TODO: Implement this")


def gan_g_loss_nonsaturating(fake_logits: torch.Tensor):
    # TODO: Practical fix: -log D(G(z)).
    raise NotImplementedError("# TODO: Implement this")


def gradient_penalty(critic, real: torch.Tensor, fake: torch.Tensor):
    # TODO: mean((||grad||_2 - 1)^2) on random interpolations.
    raise NotImplementedError("# TODO: Implement this")


def wgan_gp_d_loss(critic, real, fake, lam=10.0):
    # TODO: E[fake] - E[real] + lam * GP.
    raise NotImplementedError("# TODO: Implement this")


def spectral_normalize(weight: torch.Tensor, u: torch.Tensor, steps: int = 5):
    # TODO: Power iteration; return (W/sigma, updated u, sigma).
    raise NotImplementedError("# TODO: Implement this")
