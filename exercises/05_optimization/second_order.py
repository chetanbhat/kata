"""EXERCISE 5.1 — Damped Newton + L-BFGS from scratch.

Goals: exact-Hessian damped Newton step via autograd; two-loop recursion;
Armijo backtracking L-BFGS with (s, y) curvature pairs.

Run: pytest tests/test_05_optimization.py -k "newton or lbfgs"
Reading: readings/05_optimization/second_order.md
"""

import torch


def lbfgs_direction(g: torch.Tensor, S: list, Y: list) -> torch.Tensor:
    # TODO: Two-loop recursion; return -H_k g (steepest descent if no history).
    raise NotImplementedError("# TODO: Implement this")


class DampedNewton:
    """Solve (H + l*I) d = g exactly; p -= lr*d. step() takes a loss closure."""

    def __init__(self, params, damping: float = 1e-3, lr: float = 1.0):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def zero_grad(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def step(self, closure):
        # TODO: grad + full Hessian via autograd, solve, update.
        raise NotImplementedError("# TODO: Implement this")


class LBFGS:
    """Limited-memory BFGS; closure() zeroes grads, forwards, backprops."""

    def __init__(self, params, memory: int = 10, c1: float = 1e-4):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def step(self, closure):
        # TODO: Direction + Armijo line search + curvature-pair update.
        raise NotImplementedError("# TODO: Implement this")
