"""Residual blocks: post-activation basic block and pre-activation bottleneck.

Shortcut projection (1x1 conv + BN) is applied whenever stride != 1 or the
channel count changes, so F(x) + shortcut(x) always align.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Post-activation basic block: Conv-BN-ReLU-Conv-BN + shortcut, then ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class BottleneckBlock(nn.Module):
    """1x1 reduce -> 3x3 -> 1x1 expand; optionally pre-activation
    (BN-ReLU-Conv ordering, identity shortcut path kept clean)."""

    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int,
                 stride: int = 1, pre_activation: bool = False):
        super().__init__()
        self.pre_activation = pre_activation
        b = bottleneck_channels
        self.conv1 = nn.Conv2d(in_channels, b, 1, bias=False)
        self.conv2 = nn.Conv2d(b, b, 3, stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(b, out_channels, 1, bias=False)
        # One norm per conv; sized to the tensor each norm actually sees so
        # every parameter participates in both pre- and post-activation modes.
        sizes = (in_channels, b, b) if pre_activation else (b, b, out_channels)
        self.norms = nn.ModuleList(nn.BatchNorm2d(s) for s in sizes)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def _block(self, x):
        n0, n1, n2 = self.norms
        if self.pre_activation:
            out = self.conv1(self.relu(n0(x)))
            out = self.conv2(self.relu(n1(out)))
            return self.conv3(self.relu(n2(out)))
        out = self.relu(n0(self.conv1(x)))
        out = self.relu(n1(self.conv2(out)))
        return n2(self.conv3(out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._block(x) + self.shortcut(x)
        return out if self.pre_activation else self.relu(out)
