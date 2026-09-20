"""EXERCISE 3.2 — LR schedulers: cosine warm restarts, warmup-stable-decay.

Goals: per-step lr computation; cycle restart with T_mult growth; linear
warmup -> stable -> cosine decay phases.

Run: pytest tests/test_03_training.py -k "scheduler or cosine or warmup"
"""


class CosineAnnealingWarmRestarts:
    """eta_min + 0.5*(base-eta_min)*(1+cos(pi*T_cur/T_i)); restart per cycle."""

    def __init__(self, base_lr: float, T_0: int, T_mult: int = 1, eta_min: float = 0.0):
        # TODO: Store schedule config + step counter.
        raise NotImplementedError("# TODO: Implement this")

    def get_lr(self) -> float:
        # TODO: Find the current cycle, return the cosine-interpolated lr.
        raise NotImplementedError("# TODO: Implement this")

    def step(self) -> float:
        # TODO: Return get_lr(), then advance the counter.
        raise NotImplementedError("# TODO: Implement this")


class WarmupStableDecay:
    """Linear warmup -> constant -> cosine decay to min_lr."""

    def __init__(self, base_lr: float, warmup_steps: int, stable_steps: int,
                 decay_steps: int, min_lr: float = 0.0):
        # TODO: Store schedule config + step counter.
        raise NotImplementedError("# TODO: Implement this")

    def get_lr(self) -> float:
        # TODO: Piecewise lr by phase.
        raise NotImplementedError("# TODO: Implement this")

    def step(self) -> float:
        # TODO: Return get_lr(), then advance the counter.
        raise NotImplementedError("# TODO: Implement this")
