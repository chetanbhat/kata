"""Module 05 tests: second-order methods, adaptive lineage, sharpness."""

from importlib import import_module

import torch
import pytest

so = import_module("solutions.05_optimization.second_order")
ah = import_module("solutions.05_optimization.adaptive_history")
sh = import_module("solutions.05_optimization.sharpness")


def _quadratic(Q, t):
    w = torch.nn.Parameter(torch.zeros(Q.shape[0]))
    def closure(backward=True):
        loss = 0.5 * ((w - t) @ Q @ (w - t))
        if backward:
            for p in [w]:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            loss.backward()
        return loss
    return w, closure


# ------------------------------------------------------------ Newton / LBFGS
def test_damped_newton_one_step_quadratic():
    Q = torch.diag(torch.tensor([1.0, 10.0]))
    w, closure = _quadratic(Q, torch.tensor([1.0, 2.0]))
    opt = so.DampedNewton([w], damping=1e-3)

    def fwd():
        return 0.5 * ((w - torch.tensor([1.0, 2.0])) @ Q @ (w - torch.tensor([1.0, 2.0])))
    opt.step(fwd)
    assert float(fwd().detach()) < 1e-4
    assert w.shape == (2,)


def test_lbfgs_direction_empty_history_is_steepest():
    g = torch.tensor([3.0, -4.0])
    assert torch.allclose(so.lbfgs_direction(g, [], []), -g)


def test_lbfgs_10_step_ill_conditioned_convergence():
    Q = torch.diag(torch.tensor([1.0, 100.0]))
    w, closure = _quadratic(Q, torch.tensor([3.0, -2.0]))
    opt = so.LBFGS([w], memory=5)
    losses = []
    for _ in range(10):
        loss = opt.step(closure)
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert losses[-1] < 1e-4


# ------------------------------------------------------- adaptive lineage
def _opt_run(opt_cls, kwargs, steps=10, lr=0.1):
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.tensor([4.0, -3.0]))
    target = torch.tensor([1.0, 2.0])
    opt = opt_cls([w], lr=lr, **kwargs)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((w - target) ** 2).sum()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    return w, losses


@pytest.mark.parametrize("cls,kw,lr", [
    (ah.AdaGrad, {}, 0.5), (ah.RMSprop, {}, 0.1),
    (ah.AdamW, {"weight_decay": 0.0}, 0.1), (ah.Lion, {}, 0.1),
])
def test_adaptive_10_step_convergence(cls, kw, lr):
    _, losses = _opt_run(cls, kw, lr=lr)
    assert len(losses) == 10
    assert losses[-1] < losses[0]


def test_adamw_decoupled_decay_exact():
    p = torch.nn.Parameter(torch.tensor([2.0]))
    p.grad = torch.zeros(1)
    opt = ah.AdamW([p], lr=0.1, weight_decay=0.1)
    opt.step()
    assert float(p.detach()) == pytest.approx(2.0 * (1 - 0.1 * 0.1))


def test_lion_first_step_is_signed():
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    p.grad = torch.tensor([2.0, -3.0])
    opt = ah.Lion([p], lr=0.1)
    opt.step()
    assert torch.allclose(p.detach(), torch.tensor([0.9, 1.1]))


def test_lookahead_sync_exact():
    from importlib import import_module as im
    base_mod = im("solutions.03_training_algos.optimizers")
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.tensor([4.0, 0.0]))
    target = torch.tensor([1.0, 1.0])
    inner = base_mod.SGD([w], lr=0.1)
    opt = ah.Lookahead(inner, alpha=0.5, k=2)
    for _ in range(2):
        opt.zero_grad()
        loss = ((w - target) ** 2).sum()
        loss.backward()
        opt.step()
    # fast after 2 SGD steps: [2.92, 0.36]; slow = [4,0] + 0.5*(fast-[4,0]).
    assert torch.allclose(w.detach(), torch.tensor([3.46, 0.18]), atol=1e-6)


# ----------------------------------------------------------------- SAM/SWA
def test_sam_ascent_then_10_step_convergence():
    torch.manual_seed(0)
    from importlib import import_module as im
    base_mod = im("solutions.03_training_algos.optimizers")
    w = torch.nn.Parameter(torch.tensor([4.0, -3.0]))
    target = torch.tensor([1.0, 2.0])
    base = base_mod.SGD([w], lr=0.05)
    opt = sh.SAM(base, [w], rho=0.05)
    f = lambda: ((w - target) ** 2).sum()
    base_loss = float(f().detach())
    opt.zero_grad()
    f().backward()
    opt.first_step()
    assert float(f().detach()) >= base_loss  # ascent to the ball edge
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = f()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.first_step()
        opt.zero_grad()
        f().backward()
        opt.second_step()
    assert losses[-1] < losses[0]


def test_swa_average_exact():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    swa = sh.SWA([p])
    swa.update()
    with torch.no_grad():
        p.copy_(torch.tensor([3.0, 4.0]))
    swa.update()
    q = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    swa.copy_to([q])
    assert torch.allclose(q, torch.tensor([2.0, 3.0]))
