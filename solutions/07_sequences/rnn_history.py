"""Sequence history I: Elman RNN -> LSTM/GRU, and WHY (vanishing grads).

History: Elman RNNs (1990) cannot learn long dependencies -- backprop
multiplies Jacobians, whose norms shrink geometrically when the recurrent
spectral radius < 1 (Bengio 1994). LSTM (1997) fixes it with an additive
cell highway gated by sigmoids; GRU (2014) compresses f+i into one update
gate. This file makes the wrong turn MEASURABLE: grad-norm-vs-time probes.
"""

import torch
import torch.nn as nn


class ElmanRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, recurrent_scale: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        with torch.no_grad():
            self.W_hh.mul_(recurrent_scale)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        return torch.tanh(x @ self.W_ih.t() + h @ self.W_hh.t() + self.b)


class GRUCell(nn.Module):
    """z update gate, r reset gate, n candidate: h = (1-z)*n + z*h_prev."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_in = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(3 * hidden_size))
        for w in (self.W_iz, self.W_hz, self.W_ir, self.W_hr, self.W_in, self.W_hn):
            nn.init.xavier_uniform_(w)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        bz, br, bn = self.b.chunk(3)
        z = torch.sigmoid(x @ self.W_iz.t() + h @ self.W_hz.t() + bz)
        r = torch.sigmoid(x @ self.W_ir.t() + h @ self.W_hr.t() + br)
        n = torch.tanh(x @ self.W_in.t() + (r * h) @ self.W_hn.t() + bn)
        return (1 - z) * n + z * h


def grad_norm_vs_time(cell: nn.Module, seq_len: int, batch: int = 4,
                      input_size: int = 6, seed: int = 0) -> list[float]:
    """||d(sum h_T)/d h_t|| for t = 0..T-1. Vanishing => early norms ~ 0."""
    torch.manual_seed(seed)
    hiddens, h = [], None
    for _ in range(seq_len):
        x = torch.randn(batch, input_size)
        h = cell(x, h)
        h.retain_grad()
        hiddens.append(h)
    hiddens[-1].sum().backward()
    return [float(h.grad.norm()) for h in hiddens]
