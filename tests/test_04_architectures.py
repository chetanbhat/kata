"""Module 4 tests: ResNet blocks, LSTM cell, Transformer blocks."""

from importlib import import_module

import torch
import torch.nn as nn
import pytest

rn = import_module("solutions.04_architectures.resnet")
ls = import_module("solutions.04_architectures.lstm")
tr = import_module("solutions.04_architectures.transformer")


def assert_grads_alive(module: nn.Module):
    params = [p for p in module.parameters() if p.requires_grad]
    assert params, "module has no trainable parameters"
    for p in params:
        assert p.grad is not None, "dead graph: grad is None"
        assert torch.any(p.grad != 0), "dead graph: grad is all zero"


# -------------------------------------------------------------------- resnet
@pytest.mark.parametrize("in_c,out_c,stride,expected",
                         [(8, 8, 1, (2, 8, 8, 8)), (3, 8, 2, (2, 8, 4, 4))])
def test_residual_block_shape(in_c, out_c, stride, expected):
    torch.manual_seed(0)
    block = rn.ResidualBlock(in_c, out_c, stride=stride)
    x = torch.randn(2, in_c, 8, 8)
    assert block(x).shape == expected


@pytest.mark.parametrize("pre", [False, True])
def test_bottleneck_shape_and_grad(pre):
    torch.manual_seed(0)
    block = rn.BottleneckBlock(16, 4, 32, stride=2, pre_activation=pre)
    x = torch.randn(2, 16, 8, 8)
    out = block(x)
    assert out.shape == (2, 32, 4, 4)
    out.sum().backward()
    assert_grads_alive(block)


def test_residual_block_10_step_convergence():
    torch.manual_seed(0)
    block = rn.ResidualBlock(4, 4).eval()  # eval: BN uses running stats
    opt = torch.optim.Adam(block.parameters(), lr=1e-2)
    X = torch.randn(4, 4, 8, 8)
    target = torch.randn(4, 4, 8, 8)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = ((block(X) - target) ** 2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------- lstm
def test_lstm_cell_and_layer_shapes():
    torch.manual_seed(0)
    cell = ls.LSTMCell(input_size=6, hidden_size=8)
    h, (hn, cn) = cell(torch.randn(3, 6))
    assert h.shape == (3, 8) and hn.shape == (3, 8) and cn.shape == (3, 8)
    layer = ls.LSTM(input_size=6, hidden_size=8)
    outs, (hn2, cn2) = layer(torch.randn(3, 5, 6))
    assert outs.shape == (3, 5, 8) and hn2.shape == (3, 8)


def test_lstm_grad_flow():
    torch.manual_seed(0)
    layer = ls.LSTM(input_size=4, hidden_size=8)
    outs, _ = layer(torch.randn(2, 7, 4))
    outs.sum().backward()
    assert_grads_alive(layer)


def test_lstm_10_step_convergence():
    torch.manual_seed(0)
    model = nn.Sequential(ls.LSTM(3, 8), _LastStep(), nn.Linear(8, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    X = torch.randn(16, 6, 3)
    y = X[:, -1, :1] * 2.0  # predictable function of the last step
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = ((model(X) - y) ** 2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]


class _LastStep(nn.Module):
    def forward(self, x):
        outs, _ = x if isinstance(x, tuple) else (x, None)
        return outs[:, -1]


# --------------------------------------------------------------- transformer
def test_sdpa_shape_and_causal_masking():
    torch.manual_seed(0)
    B, T, d = 2, 6, 8
    Q = K = V = torch.randn(B, T, d)
    out = tr.scaled_dot_product_attention(Q, K, V)
    assert out.shape == (B, T, d)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    out1 = tr.scaled_dot_product_attention(Q, K, V, causal)
    V2 = V.clone()
    V2[:, 3:] *= 100  # corrupt the future
    out2 = tr.scaled_dot_product_attention(Q, K, V2, causal)
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5), \
        "causal mask leaked future info"


def test_mha_shape_and_grad():
    torch.manual_seed(0)
    attn = tr.MultiHeadSelfAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 7, 16)
    out = attn(x, causal=True)
    assert out.shape == (2, 7, 16)
    out.sum().backward()
    assert_grads_alive(attn)


def test_rmsnorm_unit_scale():
    torch.manual_seed(0)
    norm = tr.RMSNorm(12)
    x = torch.randn(3, 5, 12) * 4 + 2
    out = norm(x)
    assert out.shape == (3, 5, 12)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rope_preserves_shape_and_norm():
    torch.manual_seed(0)
    rope = tr.RotaryEmbedding(dim=16, max_seq_len=32)
    x = torch.randn(2, 10, 16)
    out = rope(x)
    assert out.shape == (2, 10, 16)
    # Rotation preserves per-position norms.
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    # ...but actually moves the vectors (position-dependent rotation).
    assert not torch.allclose(out, x)


def test_decoder_block_shape_grad_convergence():
    torch.manual_seed(0)
    block = tr.DecoderBlock(d_model=16, n_heads=4, d_ff=32)
    x = torch.randn(2, 8, 16)
    assert block(x).shape == (2, 8, 16)
    block.zero_grad()
    block(x).sum().backward()
    assert_grads_alive(block)

    opt = torch.optim.Adam(block.parameters(), lr=3e-3)
    target = torch.randn(2, 8, 16)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = ((block(x) - target) ** 2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]
