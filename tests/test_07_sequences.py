"""Module 07 tests: RNN history, attention variations, selective SSM."""

from importlib import import_module

import torch
import torch.nn as nn
import pytest

rh = import_module("solutions.07_sequences.rnn_history")
ae = import_module("solutions.07_sequences.attention_evolution")
ss = import_module("solutions.07_sequences.ssm")


def assert_grads_alive(module: nn.Module):
    for p in [p for p in module.parameters() if p.requires_grad]:
        assert p.grad is not None and torch.any(p.grad != 0)


# --------------------------------------------------------------- RNN history
def test_elman_and_gru_shapes_and_grads():
    torch.manual_seed(0)
    rnn, gru = rh.ElmanRNNCell(6, 8), rh.GRUCell(6, 8)
    x = torch.randn(3, 6)
    assert rnn(x).shape == (3, 8) and gru(x).shape == (3, 8)
    # Unroll 3 steps: a single step from h=0 legitimately gives zero grads
    # to matrices that only see the (zero) recurrent input.
    for cell in (rnn, gru):
        cell.zero_grad()
        h = None
        for _ in range(3):
            h = cell(x, h)
        h.sum().backward()
        assert_grads_alive(cell)


def test_vanishing_gradient_measured_not_memorized():
    norms = rh.grad_norm_vs_time(rh.ElmanRNNCell(6, 10, recurrent_scale=0.3),
                                 seq_len=12, seed=0)
    assert len(norms) == 12
    assert all(v >= 0 for v in norms)
    assert norms[0] < norms[-1], f"expected decay, got {norms[0]:.2e} vs {norms[-1]:.2e}"
    gru_norms = rh.grad_norm_vs_time(rh.GRUCell(6, 10), seq_len=12, seed=0)
    assert gru_norms[0] > norms[0], "gating should preserve early gradients better"


def test_gru_10_step_convergence():
    torch.manual_seed(0)
    gru = rh.GRUCell(4, 8)
    head = nn.Linear(8, 1)
    opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=1e-2)
    X = torch.randn(16, 5, 4)
    y = X.sum(dim=(1, 2,)) .unsqueeze(-1) * 0.1
    losses = []
    for _ in range(10):
        opt.zero_grad()
        h = None
        for t in range(5):
            h = gru(X[:, t], h)
        loss = ((head(h) - y) ** 2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]


# -------------------------------------------------------- attention evolution
def test_alibi_slopes_and_bias_shapes():
    slopes = ae.alibi_slopes(8)
    assert torch.allclose(slopes, torch.pow(2.0, -torch.arange(1, 9).float()))
    bias = ae.alibi_bias(6, slopes)
    assert bias.shape == (8, 6, 6)
    assert torch.all(bias[:, torch.arange(6), torch.arange(6)] == 0)
    assert torch.all(torch.isneginf(bias[:, 0, 1:]))  # future is masked
    row = bias[0, 4, :5]  # past positions: nearer (larger j) = less penalty
    assert bool((row[1:] >= row[:-1]).all())


def test_sliding_window_mask_exact():
    expected = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(ae.sliding_window_mask(4, 1), expected)


@pytest.mark.parametrize("n_kv", [4, 1])
def test_gqa_shapes_and_grads(n_kv):
    torch.manual_seed(0)
    attn = ae.GroupedQueryAttention(d_model=16, n_heads=4, n_kv_heads=n_kv)
    x = torch.randn(2, 7, 16)
    out = attn(x, causal=True)
    assert out.shape == (2, 7, 16)
    out.sum().backward()
    assert_grads_alive(attn)


def test_gqa_causal_no_future_leak():
    torch.manual_seed(0)
    attn = ae.GroupedQueryAttention(d_model=8, n_heads=2, n_kv_heads=1)
    attn.eval()
    x = torch.randn(1, 6, 8)
    out1 = attn(x)
    x2 = x.clone()
    x2[:, 4:] *= 100
    out2 = attn(x2)
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-4)


def test_gqa_10_step_convergence():
    torch.manual_seed(0)
    attn = ae.GroupedQueryAttention(d_model=16, n_heads=4, n_kv_heads=2)
    opt = torch.optim.Adam(attn.parameters(), lr=3e-3)
    x = torch.randn(2, 8, 16)
    target = torch.randn(2, 8, 16)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = ((attn(x) - target) ** 2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]


# --------------------------------------------------------------------- SSM
def test_discretize_diag_exact_and_zero_limit():
    Ad, Bd = ss.discretize_diag(torch.tensor([-1.0]), torch.tensor([2.0]), torch.tensor([0.5]))
    import math
    assert float(Ad) == pytest.approx(math.exp(-0.5))
    assert float(Bd) == pytest.approx((math.exp(-0.5) - 1.0) / -1.0 * 2.0)
    _, Bd0 = ss.discretize_diag(torch.tensor([0.0]), torch.tensor([3.0]), torch.tensor([2.0]))
    assert float(Bd0) == pytest.approx(6.0)


def test_selective_scan_shape_grad_convergence():
    torch.manual_seed(0)
    scan = ss.SelectiveScan(d_model=8, d_state=4)
    x = torch.randn(2, 9, 8)
    out = scan(x)
    assert out.shape == (2, 9, 8)
    out.sum().backward()
    assert_grads_alive(scan)
    opt = torch.optim.Adam(scan.parameters(), lr=1e-2)
    target = torch.randn(2, 9, 8)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = ((scan(x) - target) ** 2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]
