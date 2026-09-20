"""EXERCISE 5.2 — Adaptive lineage: AdaGrad / RMSprop / AdamW / Lion / Lookahead.

Goals: feel WHY each fix exists -- AdaGrad's ever-shrinking lr, RMSprop's
moving average, AdamW's decoupled decay (exact: zero grad -> p *= 1-lr*wd),
Lion's sign step, Lookahead's slow/fast sync.

Run: pytest tests/test_05_optimization.py -k "adagrad or rmsprop or adamw or lion or lookahead"
"""

import torch


class AdaGrad:
    def __init__(self, params, lr: float = 0.01, eps: float = 1e-10):
        # TODO: Per-param sum of squared grads G.
        raise NotImplementedError("# TODO: Implement this")

    def zero_grad(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def step(self):
        # TODO: G += g^2; p -= lr*g/(sqrt(G)+eps).
        raise NotImplementedError("# TODO: Implement this")


class RMSprop:
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def zero_grad(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def step(self):
        # TODO: EMA of squared grads instead of AdaGrad's full sum.
        raise NotImplementedError("# TODO: Implement this")


class AdamW:
    """Adam + DECOUPLED decay: the adaptivity must NOT touch the wd term."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def zero_grad(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def step(self):
        # TODO: Bias-corrected Adam step, then p -= lr*wd*p separately.
        raise NotImplementedError("# TODO: Implement this")


class Lion:
    """Sign-based update with momentum (hint: first step from zero = -lr*sign(g))."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def zero_grad(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def step(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")


class Lookahead:
    """Wrap any optimizer with .params/.step()/.zero_grad(); sync every k steps."""

    def __init__(self, optimizer, alpha: float = 0.5, k: int = 6):
        # TODO: Slow-weight buffers + step counter.
        raise NotImplementedError("# TODO: Implement this")

    def zero_grad(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def step(self, *args, **kwargs):
        # TODO: Fast step; every k: slow += a*(fast-slow); fast <- slow.
        raise NotImplementedError("# TODO: Implement this")
