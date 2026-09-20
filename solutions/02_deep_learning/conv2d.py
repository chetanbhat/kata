"""Manual 2D convolution via unfold + matrix multiply.

Forward: unfold input to patches, multiply by flattened kernels, fold back.
Backward: free via torch autograd (all ops differentiable), so parameter
grads flow. Handles stride / padding / dilation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1):
        super().__init__()
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        kh, kw = self.weight.shape[2], self.weight.shape[3]
        patches = F.unfold(x, (kh, kw), dilation=self.dilation,
                           padding=self.padding, stride=self.stride)  # (B, C*kh*kw, L)
        out = self.weight.view(self.weight.shape[0], -1) @ patches + self.bias.view(-1, 1)
        h_out = (H + 2 * self.padding[0] - self.dilation[0] * (kh - 1) - 1) // self.stride[0] + 1
        w_out = (W + 2 * self.padding[1] - self.dilation[1] * (kw - 1) - 1) // self.stride[1] + 1
        return out.view(B, -1, h_out, w_out)
