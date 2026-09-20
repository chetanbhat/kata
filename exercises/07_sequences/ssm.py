"""EXERCISE 7.3 — Diagonal SSM discretization + Mamba-lite selective scan.

Goals: exact ZOH discretization (A_d = e^{dt*A}, with the A->0 limit);
input-dependent dt/B/C scan loop with a stable negative diagonal A.

Run: pytest tests/test_07_sequences.py -k "ssm or discretize or selective"
Reading: readings/07_sequences/ssm.md
"""

import torch
import torch.nn as nn


def discretize_diag(A: torch.Tensor, B: torch.Tensor, dt: torch.Tensor):
    # TODO: Return (A_d, B_d); handle |A| < 1e-8 via the dt*B limit.
    raise NotImplementedError("# TODO: Implement this")


class SelectiveScan(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8):
        super().__init__()
        # TODO: A_log (positive -> A negative), dt/B/C projections, D skip.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Per-step dt=softplus, Bt, Ct; h = A_d*h + B_d*u; y = h.C + D*u.
        raise NotImplementedError("# TODO: Implement this")
