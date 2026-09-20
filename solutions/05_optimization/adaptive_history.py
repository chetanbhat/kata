"""Adaptive optimizers lineage: AdaGrad -> RMSprop -> Adam -> AdamW -> Lion.

History of the central wrong turn: AdaGrad (2011) accumulates ALL past
squared grads, so the effective lr decays to zero on long non-convex runs.
RMSprop (Hinton, lecture 6e, unpublished) fixes it with an exponential
moving average. Adam (2014) adds momentum + bias correction and wins by
default -- but couples weight decay into the adaptive denominator (the
subtle bug AdamW, 2017, fixes by decoupling). Lion (2023) shows a
sign-based update with less memory can match AdamW. Lookahead (2019) wraps
ANY optimizer with slow/fast weights.
"""

import torch


class AdaGrad:
    def __init__(self, params, lr: float = 0.01, eps: float = 1e-10):
        self.params = list(params)
        self.lr, self.eps = float(lr), float(eps)
        self.G = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p, G in zip(self.params, self.G):
            if p.grad is None:
                continue
            G.addcmul_(p.grad, p.grad)
            p.addcdiv_(p.grad, G.sqrt() + self.eps, value=-self.lr)


class RMSprop:
    def __init__(self, params, lr: float = 1e-3, alpha: float = 0.99, eps: float = 1e-8):
        self.params = list(params)
        self.lr, self.alpha, self.eps = float(lr), float(alpha), float(eps)
        self.G = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p, G in zip(self.params, self.G):
            if p.grad is None:
                continue
            G.mul_(self.alpha).addcmul_(p.grad, p.grad, value=1.0 - self.alpha)
            p.addcdiv_(p.grad, G.sqrt() + self.eps, value=-self.lr)


class AdamW:
    """Adam with DECOUPLED weight decay: p -= lr*wd*p (outside the adaptivity)."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = list(params)
        self.lr, self.eps = float(lr), float(eps)
        self.b1, self.b2 = betas
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1, bc2 = 1.0 - self.b1 ** self.t, 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            m.mul_(self.b1).add_(p.grad, alpha=1.0 - self.b1)
            v.mul_(self.b2).addcmul_(p.grad, p.grad, value=1.0 - self.b2)
            denom = (v / bc2).sqrt() + self.eps
            p.addcdiv_(m / bc1, denom, value=-self.lr)
            p.add_(p, alpha=-self.lr * self.weight_decay)  # decoupled


class Lion:
    """Evo-discovered sign update: c = b1*m + (1-b1)*g; p -= lr*(sign(c)+wd*p)."""

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99),
                 weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.weight_decay = float(weight_decay)
        self.m = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p, m in zip(self.params, self.m):
            if p.grad is None:
                continue
            c = self.b1 * m + (1.0 - self.b1) * p.grad
            p.add_(torch.sign(c) + self.weight_decay * p, alpha=-self.lr)
            m.mul_(self.b2).add_(p.grad, alpha=1.0 - self.b2)


class Lookahead:
    """Wraps any optimizer exposing .params/.step()/.zero_grad().

    k fast steps, then slow += alpha*(fast - slow); fast <- slow.
    """

    def __init__(self, optimizer, alpha: float = 0.5, k: int = 6):
        self.opt = optimizer
        self.params = optimizer.params
        self.alpha, self.k = float(alpha), int(k)
        self.slow = [p.detach().clone() for p in self.params]
        self._count = 0

    def zero_grad(self):
        self.opt.zero_grad()

    @torch.no_grad()
    def step(self, *args, **kwargs):
        self.opt.step(*args, **kwargs)
        self._count += 1
        if self._count % self.k == 0:
            for p, s in zip(self.params, self.slow):
                s.add_(p - s, alpha=self.alpha)
                p.copy_(s)
