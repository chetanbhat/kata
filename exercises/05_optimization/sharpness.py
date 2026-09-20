"""EXERCISE 5.3 — SAM (perturb-then-update) and SWA (iterate averaging).

Goals: SAM first_step ascends to w+eps (loss must RISE), second_step descends
with the perturbed gradient and restores w; SWA running average is exact.

Run: pytest tests/test_05_optimization.py -k "sam or swa or sharpness"
"""

import torch


class SAM:
    def __init__(self, base_optimizer, params, rho: float = 0.05):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def zero_grad(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def first_step(self):
        # TODO: p += rho * g / ||g|| (store eps per param).
        raise NotImplementedError("# TODO: Implement this")

    def second_step(self):
        # TODO: base opt step with perturbed grads, then subtract stored eps.
        raise NotImplementedError("# TODO: Implement this")


class SWA:
    def __init__(self, params):
        # TODO: Average buffers + counter.
        raise NotImplementedError("# TODO: Implement this")

    def update(self):
        # TODO: avg += (p - avg) / n.
        raise NotImplementedError("# TODO: Implement this")

    def copy_to(self, params):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")
