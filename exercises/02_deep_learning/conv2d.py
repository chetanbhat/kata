"""EXERCISE 2.3 — Manual Conv2D via unfold + matmul.

Goals: unfold input to patches, multiply by flattened kernels, reshape;
support stride/padding/dilation; grads flow via torch autograd.

Run: pytest tests/test_02_deep_learning.py -k conv
Reading: readings/02_deep_learning/conv2d.md
"""

import torch
import torch.nn as nn


class Conv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1):
        super().__init__()
        # TODO: Store conv hyperparams; create weight/bias parameters.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: F.unfold -> matmul with flattened weight -> reshape to (B,C,H,W).
        raise NotImplementedError("# TODO: Implement this")
