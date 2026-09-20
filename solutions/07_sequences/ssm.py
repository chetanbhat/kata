"""Sequence history III: state-space models (S4 -> Mamba).

History: attention's O(T^2) wall revived SSMs -- S4 (2022) discretizes a
continuous linear system (A, B, C) with HiPPO init for long-range memory;
Mamba (2023) makes the scan SELECTIVE (dt, B, C depend on input) while
keeping linear time. Here: exact ZOH discretization of a diagonal system +
a Mamba-lite selective scan loop.
"""

import torch
import torch.nn as nn


def discretize_diag(A: torch.Tensor, B: torch.Tensor, dt: torch.Tensor):
    """ZOH discretization of diagonal dA/dt system: A_d = e^{dt*A}.

    Returns (A_d, B_d) with B_d = (A_d - 1)/A * B (limit dt*B as A -> 0).
    All elementwise; A, B, dt broadcastable to the same shape.
    """
    A_d = torch.exp(dt * A)
    B_d = torch.where(A.abs() < 1e-8, dt * B, (A_d - 1.0) / A * B)
    return A_d, B_d


class SelectiveScan(nn.Module):
    """Mamba-lite: per-step dt/B/C from input; scan h = A_d*h + B_d*u."""

    def __init__(self, d_model: int, d_state: int = 8):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model) * 15.0 + 1.0))
        self.W_dt = nn.Parameter(torch.empty(d_model, d_model))
        self.b_dt = nn.Parameter(torch.zeros(d_model))
        self.W_B = nn.Parameter(torch.empty(d_state, d_model))
        self.W_C = nn.Parameter(torch.empty(d_state, d_model))
        self.D = nn.Parameter(torch.ones(d_model))
        nn.init.xavier_uniform_(self.W_dt)
        nn.init.xavier_uniform_(self.W_B)
        nn.init.xavier_uniform_(self.W_C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = self.d_state
        A = -torch.exp(self.A_log)  # (D,) negative diagonal: stable
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            u = x[:, t]                                          # (B, D)
            dt = torch.nn.functional.softplus(u @ self.W_dt.t() + self.b_dt)  # (B, D)+
            Bt = u @ self.W_B.t()                                # (B, N)
            Ct = u @ self.W_C.t()                                # (B, N)
            A_d, B_d = discretize_diag(A[None, :, None],
                                       Bt[:, None, :] * torch.ones(1, D, 1, device=x.device),
                                       dt[:, :, None])
            h = h * A_d + u[:, :, None] * B_d
            outs.append((h * Ct[:, None, :]).sum(-1) + self.D * u)
        return torch.stack(outs, dim=1)
