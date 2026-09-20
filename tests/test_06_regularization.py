"""Module 06 tests: regularization zoo + WeightNorm/SpectralNorm/GP."""

from importlib import import_module

import torch
import torch.nn.functional as F
import pytest

rg = import_module("solutions.06_regularization.regularizers")
nm = import_module("solutions.06_regularization.normalization")


def test_dropconnect_eval_identity_and_train_scale():
    w = torch.ones(4, 8)
    assert torch.equal(rg.dropconnect(w, 0.5, training=False), w)
    torch.manual_seed(0)
    out = rg.dropconnect(torch.ones(20000, 8), 0.5, training=True)
    assert out.shape == (20000, 8)
    assert float(out.mean()) == pytest.approx(1.0, abs=0.05)


def test_stochastic_depth_eval_and_shapes():
    torch.manual_seed(0)
    x, fx = torch.randn(4, 8), torch.randn(4, 8)
    assert torch.equal(rg.stochastic_depth(x, fx, 0.5, training=False), x + fx)
    assert rg.stochastic_depth(x, fx, 0.0, training=True).shape == (4, 8)
    out = rg.stochastic_depth(x, fx, 0.5, training=True)
    assert out.shape == (4, 8)


def test_label_smoothing_matches_ce_at_zero():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    assert float(rg.label_smoothing_loss(logits, targets, eps=0.0)) == pytest.approx(
        float(F.cross_entropy(logits, targets)), rel=1e-6)
    assert torch.isfinite(rg.label_smoothing_loss(logits, targets, eps=0.1))
    logits2 = logits.detach().requires_grad_(True)
    rg.label_smoothing_loss(logits2, targets, eps=0.1).backward()
    assert logits2.grad is not None and torch.any(logits2.grad != 0)


def test_mixup_endpoints_shapes_and_grad():
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    y = torch.arange(8)
    mx, ya, yb, lam = rg.mixup_data(x, y, lam=1.0)
    assert torch.equal(mx, x) and torch.equal(ya, y)
    mx, ya, yb, lam = rg.mixup_data(x, y, lam=0.3)
    assert mx.shape == (8, 4) and lam == pytest.approx(0.3)
    pred = torch.randn(8, 5, requires_grad=True)
    loss = rg.mixup_criterion(F.cross_entropy, pred, ya % 5, yb % 5, lam)
    loss.backward()
    assert torch.any(pred.grad != 0)


def test_weightnorm_row_norms_exact_and_convergence():
    torch.manual_seed(0)
    layer = nm.WeightNormLinear(4, 3)
    x = torch.randn(6, 4)
    assert layer(x).shape == (6, 3)
    norms = layer.weight().norm(dim=1)
    assert torch.allclose(norms, layer.g.detach(), atol=1e-6)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    y = torch.randn(6, 3)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = ((layer(x) - y) ** 2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]
    for p in layer.parameters():
        assert p.grad is not None and torch.any(p.grad != 0)


def test_spectral_norm_matches_svd():
    torch.manual_seed(0)
    layer = nm.SpectralNormLinear(8, 6, n_power=10)
    layer.train()
    x = torch.randn(16, 8)
    for _ in range(5):
        layer(x).sum().backward()
        layer.zero_grad()
    sigma_est = float(layer.sigma().detach())
    sigma_true = float(torch.linalg.svdvals(layer.weight.detach())[0])
    assert sigma_est == pytest.approx(sigma_true, rel=0.06)
    wsn = layer.weight / layer.sigma().detach()
    assert float(torch.linalg.svdvals(wsn.detach())[0]) == pytest.approx(1.0, rel=0.06)


def test_gradient_penalty_exact_on_linear_critic():
    torch.manual_seed(0)
    critic = lambda t: t.sum(dim=1, keepdim=True)  # grad norm = sqrt(d)
    real = torch.randn(8, 4)
    fake = torch.randn(8, 4)
    gp = nm.gradient_penalty(critic, real, fake)
    assert float(gp) == pytest.approx((4 ** 0.5 - 1.0) ** 2, rel=1e-4)
    assert float(gp) >= 0
