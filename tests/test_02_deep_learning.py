"""Module 2 tests: autograd engine, custom layers, manual Conv2D."""

from importlib import import_module

import torch
import torch.nn as nn
import pytest

ag = import_module("solutions.02_deep_learning.autograd")
ly = import_module("solutions.02_deep_learning.layers")
cv = import_module("solutions.02_deep_learning.conv2d")

Value = ag.Value


def assert_grads_alive(module: nn.Module):
    params = [p for p in module.parameters() if p.requires_grad]
    assert params, "module has no trainable parameters"
    for p in params:
        assert p.grad is not None, "dead graph: grad is None"
        assert torch.any(p.grad != 0), "dead graph: grad is all zero"


# ----------------------------------------------------------------- autograd
def test_autograd_known_gradients():
    # Micrograd-style graph: d = a*b + c ; e = d.relu? (use tanh branch too).
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b + c          # -6 + 10 = 4
    e = d.tanh()           # tanh(4)
    e.backward()
    t = 4.0
    dt = 1.0 - __import__("math").tanh(t) ** 2
    assert a.grad == pytest.approx(b.data * dt)
    assert b.grad == pytest.approx(a.data * dt)
    assert c.grad == pytest.approx(dt)


def test_autograd_pow_div_exp_sigmoid_relu():
    x = Value(3.0)
    y = (x ** 2 / Value(2.0) + x.exp() * Value(0.0) + x.sigmoid() + (-x).relu())
    # relu(-3)=0 branch contributes 0; check total analytically instead:
    z = Value(3.0)
    w = z ** 2 + z.sigmoid()
    w.backward()
    import math
    s = 1 / (1 + math.exp(-3.0))
    assert z.grad == pytest.approx(6.0 + s * (1 - s))
    assert y.data == pytest.approx(9.0 / 2.0 + s + 0.0)


def test_autograd_10_step_scalar_descent():
    w = Value(0.0)
    losses = []
    for _ in range(10):
        for v in [w]:
            v.grad = 0.0
        loss = (w + Value(-3.0)) ** 2  # (w-3)^2
        losses.append(loss.data)
        loss.backward()
        w.data -= 0.5 * w.grad
    assert losses[-1] < losses[0]
    assert w.data == pytest.approx(3.0, abs=0.1)


# ------------------------------------------------------------------- layers
@pytest.mark.parametrize("init", ["he", "xavier", "zeros"])
def test_linear_shape_and_grad(init):
    torch.manual_seed(0)
    layer = ly.Linear(6, 4, init=init)
    x = torch.randn(5, 6)
    out = layer(x)
    assert out.shape == (5, 4)
    out.sum().backward()
    assert_grads_alive(layer)


def test_linear_10_step_convergence():
    torch.manual_seed(0)
    layer = ly.Linear(4, 1)
    X = torch.randn(32, 4)
    y = X @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]]) + 0.5
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = ((layer(X) - y) ** 2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]


def test_batchnorm_train_eval_and_stats():
    torch.manual_seed(0)
    bn = ly.BatchNorm1d(4)
    x = torch.randn(16, 4) * 5 + 3
    before = bn.running_mean.clone()
    out = bn(x)
    assert out.shape == (16, 4)
    assert not torch.allclose(bn.running_mean, before), "running stats not tracked"
    bn.eval()
    with torch.no_grad():
        out_eval = bn(torch.randn(2, 4))
    assert out_eval.shape == (2, 4)
    bn.train()
    bn.zero_grad()
    bn(x).sum().backward()
    assert_grads_alive(bn)


def test_layernorm_normalizes():
    torch.manual_seed(0)
    ln = ly.LayerNorm(8)
    x = torch.randn(4, 8) * 10 + 5
    out = ln(x)
    assert out.shape == (4, 8)
    assert torch.allclose(out.mean(-1), torch.zeros(4), atol=1e-5)
    assert torch.allclose(out.var(-1, unbiased=False), torch.ones(4), atol=1e-5)
    out.sum().backward()
    assert_grads_alive(ln)


def test_dropout_eval_identity_and_train_shape():
    torch.manual_seed(0)
    do = ly.Dropout(p=0.5)
    x = torch.ones(4, 8)
    do.eval()
    assert torch.equal(do(x), x)
    do.train()
    out = do(x)
    assert out.shape == (4, 8)
    # Inverted scaling keeps the mean at ~1 over many draws.
    torch.manual_seed(1)
    big = do(torch.ones(20000, 8))
    assert float(big.mean()) == pytest.approx(1.0, abs=0.05)


# -------------------------------------------------------------------- conv2d
def test_conv2d_matches_nn_conv2d_and_shape():
    torch.manual_seed(0)
    conv = cv.Conv2D(3, 4, kernel_size=3, stride=2, padding=1)
    ref = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
    with torch.no_grad():
        ref.weight.copy_(conv.weight)
        ref.bias.copy_(conv.bias)
    x = torch.randn(2, 3, 8, 8)
    out = conv(x)
    assert out.shape == (2, 4, 4, 4)
    assert torch.allclose(out, ref(x), atol=1e-5)


def test_conv2d_dilation_shape_and_grad():
    torch.manual_seed(0)
    conv = cv.Conv2D(2, 5, kernel_size=3, padding=2, dilation=2)
    x = torch.randn(1, 2, 10, 10, requires_grad=False)
    out = conv(x)
    assert out.shape == (1, 5, 10, 10)
    out.sum().backward()
    assert_grads_alive(conv)


def test_conv2d_10_step_convergence():
    torch.manual_seed(0)
    conv = cv.Conv2D(2, 2, kernel_size=3, padding=1)
    opt = torch.optim.SGD(conv.parameters(), lr=0.05)
    X = torch.randn(4, 2, 6, 6)
    target = torch.randn(4, 2, 6, 6)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = ((conv(X) - target) ** 2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]
