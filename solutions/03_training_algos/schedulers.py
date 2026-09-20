"""Learning-rate schedulers as step-counted schedules.

Both schedulers expose ``step() -> float`` (advance one step, return the new
lr) and ``get_lr()``. Call ``attach(optimizer)`` to also write the lr into a
torch-style optimizer's ``param_groups`` on every step.
"""

import math


class _Schedule:
    def __init__(self, base_lr: float):
        self.base_lr = float(base_lr)
        self.step_count = 0

    def get_lr(self) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def attach(self, optimizer) -> "._Schedule":
        self._optimizer = optimizer
        return self

    def _push(self, lr: float) -> float:
        if hasattr(self, "_optimizer"):
            for group in self._optimizer.param_groups:
                group["lr"] = lr
        return lr


class CosineAnnealingWarmRestarts(_Schedule):
    """lr = eta_min + 0.5*(base-eta_min)*(1 + cos(pi*T_cur/T_i)); on cycle end
    restart at base_lr with period T_i *= T_mult."""

    def __init__(self, base_lr: float, T_0: int, T_mult: int = 1, eta_min: float = 0.0):
        super().__init__(base_lr)
        self.T_0, self.T_mult, self.eta_min = int(T_0), int(T_mult), float(eta_min)

    def get_lr(self) -> float:
        t, period = self.step_count, self.T_0
        while t >= period:  # find current cycle (supports T_mult growth)
            t -= period
            period *= self.T_mult
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1.0 + math.cos(math.pi * t / period))

    def step(self) -> float:
        lr = self.get_lr()
        self.step_count += 1
        return self._push(lr)


class WarmupStableDecay(_Schedule):
    """Linear warmup (0 -> base) over `warmup_steps`, constant `stable_steps`,
    then cosine decay to `min_lr` over `decay_steps`."""

    def __init__(self, base_lr: float, warmup_steps: int, stable_steps: int,
                 decay_steps: int, min_lr: float = 0.0):
        super().__init__(base_lr)
        self.warmup_steps, self.stable_steps = int(warmup_steps), int(stable_steps)
        self.decay_steps, self.min_lr = int(decay_steps), float(min_lr)

    def get_lr(self) -> float:
        t = self.step_count
        if t < self.warmup_steps:
            return self.base_lr * (t + 1) / max(1, self.warmup_steps)
        t -= self.warmup_steps
        if t < self.stable_steps:
            return self.base_lr
        t -= self.stable_steps
        if self.decay_steps <= 0:
            return self.min_lr
        progress = min(1.0, (t + 1) / self.decay_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress))

    def step(self) -> float:
        lr = self.get_lr()
        self.step_count += 1
        return self._push(lr)
