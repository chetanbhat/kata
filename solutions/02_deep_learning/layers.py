"""Custom neural-network layers in PyTorch.

Covers: Linear with Xavier/Glorot + He/Kaiming init, BatchNorm1d with
running statistics, LayerNorm, and Dropout with train/eval scaling.
"""

import math

import torch
import torch.nn as nn


class Linear(nn.Module):
    """Dense layer y = xW' + b with selectable init: he / xavier / zeros."""

    def __init__(self, in_features: int, out_features: int, init: str = "he"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters(init)

    def reset_parameters(self, init: str = "he"):
        if init == "he":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        elif init == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif init == "zeros":
            nn.init.zeros_(self.weight)
        else:
            raise ValueError(f"unknown init: {init}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t() + self.bias


class BatchNorm1d(nn.Module):
    """Batch norm tracking running stats; uses batch stats in train mode."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_hat + self.bias


class LayerNorm(nn.Module):
    """Layer norm over the last `len(normalized_shape)` dims (with affine)."""

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class Dropout(nn.Module):
    """Inverted dropout: scales by 1/(1-p) in training, identity in eval."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "dropout prob must be in [0, 1)"
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) > self.p).to(x.dtype) / (1.0 - self.p)
        return x * mask
