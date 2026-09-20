"""EXERCISE 3.1 — Optimizers from scratch: SGD (+momentum/Nesterov), Adam.

Goals: velocity buffers; Nesterov lookahead; Adam raw moments m/v with bias
correction m/(1-b1^t), v/(1-b2^t); torch.optim-like zero_grad/step API.

Run: pytest tests/test_03_training.py -k "sgd or adam or optimizer"
Reading: readings/03_training_algos/optimizers.md
"""

import torch


class SGD:
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0,
                 nesterov: bool = False, weight_decay: float = 0.0):
        # TODO: Store params + hyperparams; init one velocity buffer per param.
        raise NotImplementedError("# TODO: Implement this")

    def zero_grad(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def step(self):
        # TODO: v = mu*v + g ; p -= lr*v (Nesterov: p -= lr*(g + mu*v)).
        raise NotImplementedError("# TODO: Implement this")


class Adam:
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        # TODO: Store params; init m/v buffers and step counter t.
        raise NotImplementedError("# TODO: Implement this")

    def zero_grad(self):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def step(self):
        # TODO: Update biased moments, bias-correct, p -= lr*m_hat/(sqrt(v_hat)+eps).
        raise NotImplementedError("# TODO: Implement this")
