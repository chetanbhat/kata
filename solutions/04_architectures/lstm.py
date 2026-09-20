"""LSTM cell built step-by-step from raw tensor gates.

Gate equations (fused matmuls, split into four):
  f_t = sigmoid(x@W_if + h@W_hf + b_f)   (forget)
  i_t = sigmoid(x@W_ii + h@W_hi + b_i)   (input)
  o_t = sigmoid(x@W_io + h@W_ho + b_o)   (output)
  g_t = tanh(x@W_ig + h@W_hg + b_g)      (cell candidate)
  c_t = f_t*c_{t-1} + i_t*g_t ;  h_t = o_t*tanh(c_t)
"""

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        # Forget-gate bias trick: start biased toward remembering.
        with torch.no_grad():
            self.b_ih[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, x: torch.Tensor, state=None):
        B = x.shape[0]
        if state is None:
            h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = state
        gates = x @ self.W_ih.t() + self.b_ih + h @ self.W_hh.t() + self.b_hh
        f, i, o, g = gates.chunk(4, dim=-1)
        f, i, o = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)


class LSTM(nn.Module):
    """Unrolls LSTMCell over time; returns last hidden + stacked outputs."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, state=None):
        outs, st = [], state
        for t in range(x.shape[1]):
            h, st = self.cell(x[:, t], st)
            outs.append(h)
        return torch.stack(outs, dim=1), st
