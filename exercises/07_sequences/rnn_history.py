"""EXERCISE 7.1 — Elman RNN, GRU, and the vanishing-gradient probe.

Goals: tanh recurrence; z/r/n GRU gates; grad_norm_vs_time must SHOW decay
for small recurrent weights (the 1990s wrong turn, measured not memorized).

Run: pytest tests/test_07_sequences.py -k "rnn or gru or vanishing"
Reading: readings/07_sequences/rnn_history.md
"""

import torch
import torch.nn as nn


class ElmanRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, recurrent_scale: float = 1.0):
        super().__init__()
        # TODO: W_ih/W_hh/b params; scale the recurrent matrix.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor, h=None):
        # TODO: tanh(x@W_ih' + h@W_hh' + b).
        raise NotImplementedError("# TODO: Implement this")


class GRUCell(nn.Module):
    """z update, r reset, n candidate: h = (1-z)*n + z*h_prev."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # TODO: Six fused-style matrices + 3-part bias.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor, h=None):
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")


def grad_norm_vs_time(cell, seq_len: int, batch=4, input_size=6, seed=0):
    # TODO: Forward a random seq (retain grads), backward from last hidden,
    # return [||d(sum h_T)/d h_t||] per step.
    raise NotImplementedError("# TODO: Implement this")
