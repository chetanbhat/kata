"""EXERCISE 4.1 — Residual blocks (ResNet basic + bottleneck).

Goals: post-activation basic block; bottleneck reduce/expand; 1x1 projection
shortcut on channel/stride mismatch; optional pre-activation ordering.

Run: pytest tests/test_04_architectures.py -k "residual or bottleneck"
Reading: readings/04_architectures/resnet.md
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Post-activation basic block: Conv-BN-ReLU-Conv-BN + shortcut, then ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # TODO: Two conv-bn stages + projection shortcut when shapes mismatch.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")


class BottleneckBlock(nn.Module):
    """1x1 reduce -> 3x3 -> 1x1 expand; optionally pre-activation."""

    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int,
                 stride: int = 1, pre_activation: bool = False):
        super().__init__()
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")
