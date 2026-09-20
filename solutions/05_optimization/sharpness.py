"""Sharpness-aware minimization (SAM) and stochastic weight averaging (SWA).

History: flat minima generalize better (Hochreiter & Schmidhuber 1997, Keskar
2017 large-batch sharpness). SAM (2020) minimizes the worst-case loss in a
rho-ball via a two-step perturb-then-update; SWA (2017) averages SGD iterates
to land in flat basins. Both are cheap add-ons to any base optimizer.
"""

import torch


class SAM:
    """Wraps a base optimizer over the same params.

    Protocol per step: loss.backward(); first_step()  (ascend to w+eps);
    zero_grad(); loss_perturbed.backward(); second_step() (descend, restore w).
    """

    def __init__(self, base_optimizer, params, rho: float = 0.05):
        self.opt = base_optimizer
        self.params = list(params)
        self.rho = float(rho)
        self._eps: dict = {}

    def zero_grad(self):
        self.opt.zero_grad()

    @torch.no_grad()
    def first_step(self):
        norm = torch.sqrt(sum((p.grad ** 2).sum() for p in self.params if p.grad is not None))
        scale = self.rho / (norm + 1e-12)
        for p in self.params:
            if p.grad is None:
                continue
            e = p.grad * scale
            self._eps[id(p)] = e
            p.add_(e)

    @torch.no_grad()
    def second_step(self):
        self.opt.step()  # steps with the perturbed-point gradient
        for p in self.params:
            if id(p) in self._eps:
                p.sub_(self._eps[id(p)])
        self._eps = {}


class SWA:
    """Running average of parameters: avg += (p - avg) / n."""

    def __init__(self, params):
        self.params = list(params)
        self.avg = [p.detach().clone() for p in self.params]
        self.n = 0

    @torch.no_grad()
    def update(self):
        self.n += 1
        for a, p in zip(self.avg, self.params):
            a.add_(p - a, alpha=1.0 / self.n)

    @torch.no_grad()
    def copy_to(self, params):
        for p, a in zip(params, self.avg):
            p.copy_(a)
