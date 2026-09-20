"""Module 3 tests: hand-rolled optimizers, schedulers, training loop."""

from tests.impl import load, ref

import torch
import torch.nn as nn
import pytest

op = load("03_training_algos.optimizers")
sc = load("03_training_algos.schedulers")
tl = load("03_training_algos.training_loop")


def _quadratic_run(opt_cls, opt_kwargs, steps=10, lr=0.1, seed=0):
    torch.manual_seed(seed)
    w = torch.nn.Parameter(torch.tensor([4.0, -3.0]))
    target = torch.tensor([1.0, 2.0])
    opt = opt_cls([w], lr=lr, **opt_kwargs)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((w - target) ** 2).sum()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    return w, losses


# ------------------------------------------------------------------ SGD/Adam
def test_sgd_exact_step_and_shape_preserved():
    p = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    p.grad = torch.tensor([[0.5, -0.5], [1.0, 0.0]])
    opt = op.SGD([p], lr=0.1)
    opt.step()
    assert p.shape == (2, 2)
    assert torch.allclose(p, torch.tensor([[0.95, 2.05], [2.9, 4.0]]))


def test_sgd_10_step_convergence():
    _, losses = _quadratic_run(op.SGD, {})
    assert len(losses) == 10
    assert losses[-1] < losses[0]


def test_sgd_momentum_and_nesterov_converge():
    for kwargs in ({"momentum": 0.9}, {"momentum": 0.9, "nesterov": True}):
        _, losses = _quadratic_run(op.SGD, kwargs)
        assert losses[-1] < losses[0]


def test_adam_bias_correction_single_step():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.tensor([2.0])
    opt = op.Adam([p], lr=0.01)
    opt.step()
    # m_hat = 2, v_hat = 4 -> update = lr * 2 / (2 + eps) ~= lr.
    assert float(p.detach()) == pytest.approx(1.0 - 0.01, abs=1e-6)


def test_adam_10_step_convergence():
    _, losses = _quadratic_run(op.Adam, {}, lr=0.1)
    assert losses[-1] < losses[0]


def test_zero_grad_clears():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p.grad = torch.ones(2)
    opt = op.Adam([p])
    opt.zero_grad()
    assert torch.all(p.grad == 0)


# ---------------------------------------------------------------- schedulers
def test_cosine_warm_restarts_shape_of_schedule():
    sched = sc.CosineAnnealingWarmRestarts(base_lr=0.1, T_0=4, T_mult=1)
    lrs = [sched.step() for _ in range(9)]
    assert lrs[0] == pytest.approx(0.1)          # cycle start at base_lr
    assert lrs[1] > lrs[2] > lrs[3]              # cosine decay within cycle
    assert lrs[4] == pytest.approx(0.1)          # restart jumps back up
    assert lrs[4] > lrs[3]


def test_cosine_restart_period_growth():
    sched = sc.CosineAnnealingWarmRestarts(base_lr=0.1, T_0=2, T_mult=2)
    lrs = [sched.step() for _ in range(6)]
    assert lrs[2] == pytest.approx(0.1)          # second cycle, length 4
    assert lrs[2] > lrs[1]


def test_warmup_stable_decay_phases():
    sched = sc.WarmupStableDecay(base_lr=0.2, warmup_steps=3, stable_steps=2,
                                 decay_steps=4, min_lr=0.0)
    lrs = [sched.step() for _ in range(9)]
    assert lrs[0] < lrs[1] < lrs[2]              # warmup rises to base
    assert lrs[2] == pytest.approx(0.2)
    assert lrs[3] == pytest.approx(0.2)          # stable plateau
    assert lrs[4] > lrs[5] > lrs[6]              # decay falls
    assert lrs[-1] == pytest.approx(0.0, abs=1e-9)


# -------------------------------------------------------------- training loop
def test_train_loop_10_step_convergence_with_all_mechanics():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    X = torch.randn(40, 4)
    y = X.sum(-1, keepdim=True)
    opt = op.Adam(list(model.parameters()), lr=0.05)
    sched = sc.CosineAnnealingWarmRestarts(base_lr=0.05, T_0=10)
    hist = tl.train_loop(model, nn.MSELoss(), opt, X, y, epochs=10, batch_size=8,
                         max_grad_norm=1.0, accumulation_steps=2, use_amp=False,
                         scheduler=sched)
    assert len(hist["loss"]) == 10
    assert hist["loss"][-1] < hist["loss"][0]


def test_train_loop_clipping_bounds_grad_norm():
    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    X = torch.randn(16, 2) * 100  # huge inputs -> huge grads
    y = torch.randn(16, 1)
    opt = op.SGD(list(model.parameters()), lr=1e-4)
    hist = tl.train_loop(model, nn.MSELoss(), opt, X, y, epochs=3,
                         max_grad_norm=0.5)
    assert all(v == v and abs(v) != float("inf") for v in hist["loss"])
