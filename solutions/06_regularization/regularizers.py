"""Regularization zoo: DropConnect, Stochastic Depth, Label Smoothing, Mixup.

History: dropout (2012/14) -> DropConnect (2013, mask weights not activations)
-> Stochastic Depth (2016, drop whole residual branches in very deep nets) ->
label smoothing (2016, Inception-v3: penalize overconfidence) -> Mixup (2017,
train on convex combos; breaks the i.i.d. dogma and works).
"""

import torch
import torch.nn.functional as F


def dropconnect(weight: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Mask the WEIGHT matrix (inverted scaling); identity when not training."""
    if not training or p == 0.0:
        return weight
    mask = (torch.rand_like(weight) > p).to(weight.dtype) / (1.0 - p)
    return weight * mask


def stochastic_depth(x: torch.Tensor, fx: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Drop the residual branch with prob p: x + Bernoulli(survival)/survival * Fx."""
    if not training or p == 0.0:
        return x + fx
    survival = 1.0 - p
    mask = (torch.rand(x.shape[0], *([1] * (x.dim() - 1)), device=x.device) < survival)
    return x + mask.to(x.dtype) * fx / survival


def label_smoothing_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """CE against smoothed targets: (1-eps) on truth + eps/K elsewhere."""
    log_p = F.log_softmax(logits, dim=-1)
    nll = -log_p.gather(1, targets.view(-1, 1)).squeeze(1)
    smooth = -log_p.mean(dim=-1)
    return ((1.0 - eps) * nll + eps * smooth).mean()


def mixup_data(x: torch.Tensor, y: torch.Tensor, lam: float):
    """Convex combo with a shuffled copy; returns (mixed_x, y_a, y_b, lam)."""
    idx = torch.randperm(x.shape[0])
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a, y_b, lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def sample_mixup_lambda(alpha: float = 1.0) -> float:
    return float(torch.distributions.Beta(alpha, alpha).sample())
