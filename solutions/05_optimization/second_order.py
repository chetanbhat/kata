"""Second-order optimization: damped Newton and L-BFGS from scratch.

History: full Newton (1940s roots, 1980s NN revival) is infeasible at scale
(O(n^3) solves, indefinite Hessians) -> damping (Levenberg-Marquardt style)
-> quasi-Newton (BFGS, 1970) -> limited-memory L-BFGS (1989) storing only
recent (s, y) curvature pairs. These still matter: small scientific models,
last-layer fine-tuning, and neural PDE solvers.
"""

import torch


def _flatten(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def _shapes(params):
    return [p.shape for p in params]


def _unflatten(vec, shapes):
    out, i = [], 0
    for s in shapes:
        n = 1
        for d in s:
            n *= d
        out.append(vec[i:i + n].reshape(s))
        i += n
    return out


class DampedNewton:
    """One damped Newton step per call: solve (H + l*I) d = g, p -= lr*d.

    `step(closure)` where closure() returns the loss. Hessian is built
    exactly via autograd -- only viable for small problems (the point).
    """

    def __init__(self, params, damping: float = 1e-3, lr: float = 1.0):
        self.params = list(params)
        self.damping, self.lr = float(damping), float(lr)
        self.shapes = _shapes(self.params)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def _set_flat(self, vec):
        for p, v in zip(self.params, _unflatten(vec, self.shapes)):
            p.copy_(v)

    def step(self, closure):
        self.zero_grad()
        loss = closure()
        g = torch.autograd.grad(loss, self.params, create_graph=True)
        gf = _flatten(g)
        n = gf.numel()
        rows = []
        for i in range(n):
            gi = torch.autograd.grad(gf[i], self.params, retain_graph=(i < n - 1))
            rows.append(_flatten(gi).detach())
        H = torch.stack(rows)
        d = torch.linalg.solve(H + self.damping * torch.eye(n), gf.detach())
        self._set_flat(_flatten(self.params).detach() - self.lr * d)
        return loss


def lbfgs_direction(g: torch.Tensor, S: list, Y: list) -> torch.Tensor:
    """L-BFGS two-loop recursion; returns the descent direction (-H_k g)."""
    q = g.clone()
    alphas, rhos = [], []
    for s, y in zip(reversed(S), reversed(Y)):
        rho = 1.0 / float(y @ s)
        rhos.append(rho)
        a = rho * float(s @ q)
        alphas.append(a)
        q = q - a * y
    r = q  # H_0 = I
    for s, y, a, rho in zip(S, Y, reversed(alphas), reversed(rhos)):
        beta = rho * float(y @ r)
        r = r + s * (a - beta)
    return -r


class LBFGS:
    """Limited-memory BFGS with Armijo backtracking.

    `step(closure)` where closure() zeroes grads, forwards, backprops, and
    returns the loss (same protocol as torch.optim.LBFGS).
    """

    def __init__(self, params, memory: int = 10, c1: float = 1e-4):
        self.params = list(params)
        self.memory, self.c1 = int(memory), float(c1)
        self.shapes = _shapes(self.params)
        self.S: list = []
        self.Y: list = []

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def _set_flat(self, vec):
        for p, v in zip(self.params, _unflatten(vec, self.shapes)):
            p.copy_(v)

    def step(self, closure):
        prev_loss = closure()
        g = _flatten([p.grad for p in self.params]).detach()
        d = lbfgs_direction(g, self.S, self.Y)
        x = _flatten(self.params).detach()
        gtd = float(g @ d)
        t, loss, accepted = 1.0, prev_loss, False
        for _ in range(25):
            self._set_flat(x + t * d)
            loss = closure()
            if float(loss.detach()) <= float(prev_loss.detach()) + self.c1 * t * gtd:
                accepted = True
                break
            t *= 0.5
        if not accepted:  # fall back: stay, keep old curvature info
            self._set_flat(x)
            return closure()
        g_new = _flatten([p.grad for p in self.params]).detach()
        s, y = t * d, g_new - g
        if float(y @ s) > 1e-10:
            self.S.append(s)
            self.Y.append(y)
            self.S, self.Y = self.S[-self.memory:], self.Y[-self.memory:]
        return loss
