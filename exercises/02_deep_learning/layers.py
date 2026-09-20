"""EXERCISE 2.2 — Custom layers: Linear, BatchNorm1d, LayerNorm, Dropout.

Goals: custom init (He/Xavier), running-stat BatchNorm with train/eval
behavior, LayerNorm over trailing dims, inverted Dropout.

Run: pytest tests/test_02_deep_learning.py -k "layer or norm or dropout or linear"
"""

import torch
import torch.nn as nn


class Linear(nn.Module):
    """Dense layer with selectable init: he / xavier / zeros."""

    def __init__(self, in_features: int, out_features: int, init: str = "he"):
        super().__init__()
        # TODO: Create weight/bias parameters and apply the chosen init.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")


class BatchNorm1d(nn.Module):
    """Batch norm with running stats; batch stats in train, running in eval."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        # TODO: weight/bias params + running_mean/running_var buffers.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")


class LayerNorm(nn.Module):
    """Layer norm over the last len(normalized_shape) dims, with affine."""

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")


class Dropout(nn.Module):
    """Inverted dropout: scale by 1/(1-p) in train, identity in eval."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")
