"""EXERCISE 3.3 — Modular training loop.

Goals: mini-batching, grad clipping by norm, accumulation steps, AMP hooks
that no-op on CPU, optional scheduler stepping; return loss history.

Run: pytest tests/test_03_training.py -k "train_loop or clipping or accumulation"
Reading: readings/03_training_algos/training_loop.md
"""

import torch
import torch.nn as nn


def train_loop(model: nn.Module, loss_fn, optimizer, X: torch.Tensor, y: torch.Tensor,
               epochs: int = 10, batch_size: int | None = None, max_grad_norm=None,
               accumulation_steps: int = 1, use_amp: bool = False, scheduler=None,
               device: str = "cpu") -> dict:
    """Train `model` on (X, y); return {'loss': [per-epoch mean loss]}."""
    # TODO: Implement this.
    raise NotImplementedError("# TODO: Implement this")
