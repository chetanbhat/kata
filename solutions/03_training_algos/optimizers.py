"""Optimizers from scratch: SGD (momentum / Nesterov) and Adam.

Both operate on torch tensors/parameters with a torch.optim-like API:
``zero_grad()`` clears ``.grad``; ``step()`` applies one update in
``torch.no_grad()``. Adam tracks first/second raw moments with bias
correction:  m_hat = m/(1-b1^t), v_hat = v/(1-b2^t).
"""

import torch


class SGD:
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0,
                 nesterov: bool = False, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = nesterov
        self.weight_decay = float(weight_decay)
        self.velocities = [torch.zeros_like(p) for p in self.params]
        if nesterov and momentum == 0.0:
            raise ValueError("Nesterov requires momentum > 0")

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.velocities):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p
            v.mul_(self.momentum).add_(g)
            p.add_(v, alpha=-self.lr) if not self.nesterov else \
                p.add_(g + self.momentum * v, alpha=-self.lr)


class Adam:
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
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
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p
            m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
            m_hat, v_hat = m / bc1, v / bc2
            p.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)
