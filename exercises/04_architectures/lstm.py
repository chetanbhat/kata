"""EXERCISE 4.2 — LSTM cell from raw tensor gates.

Goals: fused input/hidden matmuls split into f/i/o/g; forget/input/output
sigmoids + tanh candidate; c_t, h_t updates; unrolled LSTM layer.

Run: pytest tests/test_04_architectures.py -k lstm
Reading: readings/04_architectures/lstm.md
"""

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # TODO: Fused W_ih/W_hh/b_ih/b_hh params; forget-bias init to 1.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor, state=None):
        # TODO: Compute gates, return (h_new, (h_new, c_new)).
        raise NotImplementedError("# TODO: Implement this")


class LSTM(nn.Module):
    """Unrolls LSTMCell over time; returns stacked outputs + final state."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor, state=None):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")
