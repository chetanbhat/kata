"""EXERCISE 6.1 — Regularization zoo: DropConnect, Stochastic Depth, LS, Mixup.

Goals: mask weights (not activations); drop residual branches; smoothed
targets (eps=0 must equal plain CE); convex-combo training pairs.

Run: pytest tests/test_06_regularization.py -k "dropconnect or stochastic or smoothing or mixup"
"""

import torch


def dropconnect(weight: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: Inverted-scaled weight mask; identity when not training.
    raise NotImplementedError("# TODO: Implement this")


def stochastic_depth(x: torch.Tensor, fx: torch.Tensor, p: float, training: bool):
    # TODO: x + Bernoulli(survival)/survival * Fx; x + Fx at eval.
    raise NotImplementedError("# TODO: Implement this")


def label_smoothing_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 0.1):
    # TODO: (1-eps)*NLL + eps*mean(-log_softmax), averaged.
    raise NotImplementedError("# TODO: Implement this")


def mixup_data(x: torch.Tensor, y: torch.Tensor, lam: float):
    # TODO: Return (lam*x + (1-lam)*x_shuffled, y, y_shuffled, lam).
    raise NotImplementedError("# TODO: Implement this")


def mixup_criterion(criterion, pred: torch.Tensor, y_a, y_b, lam: float):
    # TODO: lam*crit(pred, y_a) + (1-lam)*crit(pred, y_b).
    raise NotImplementedError("# TODO: Implement this")
