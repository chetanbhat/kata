"""Module 08 tests: VAE, GAN losses, diffusion schedules and steps."""

from tests.impl import load, ref

import torch
import torch.nn as nn
import pytest

va = load("08_generative.vae")
gn = load("08_generative.gan")
df = load("08_generative.diffusion")


# --------------------------------------------------------------------- VAE
def test_kl_normal_closed_form():
    kl = va.kl_normal(torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 0.0]]))
    assert kl.shape == (1,)
    assert float(kl[0]) == pytest.approx(0.5)


def test_reparameterize_shape_and_grad_path():
    torch.manual_seed(0)
    mu = torch.zeros(4, 3, requires_grad=True)
    lv = torch.zeros(4, 3, requires_grad=True)
    z = va.reparameterize(mu, lv)
    assert z.shape == (4, 3)
    z.sum().backward()
    assert mu.grad is not None and torch.all(mu.grad == 1.0)
    assert lv.grad is not None and torch.any(lv.grad != 0)


def test_vae_beta_zero_kills_kl_and_elbo_converges():
    torch.manual_seed(0)
    model = va.VAE(x_dim=16, h_dim=32, z_dim=4)
    X = (torch.rand(32, 16) > 0.5).float()
    out = model.elbo(X, beta=0.0)
    assert set(out) == {"loss", "recon", "kl"}
    assert float(out["loss"].detach()) == pytest.approx(float(out["recon"]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = model.elbo(X)["loss"]
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]


# --------------------------------------------------------------------- GAN
def test_gan_losses_exact_and_directional():
    assert float(gn.gan_d_loss(torch.tensor([0.0]), torch.tensor([0.0]))) == pytest.approx(
        2 * 0.6931471805599453, rel=1e-6)
    confident = gn.gan_d_loss(torch.tensor([5.0]), torch.tensor([-5.0]))
    unsure = gn.gan_d_loss(torch.tensor([0.0]), torch.tensor([0.0]))
    assert confident < unsure  # D loss rewards correct confidence
    assert gn.gan_g_loss_nonsaturating(torch.tensor([5.0])) < \
        gn.gan_g_loss_nonsaturating(torch.tensor([0.0]))
    # The original pathology, made explicit: when D confidently rejects
    # (fake logit << 0), the minimax G gradient vanishes while the
    # non-saturating gradient stays strong.
    f_sat = torch.tensor([-10.0], requires_grad=True)
    gn.gan_g_loss_minimax(f_sat).backward()
    f_ns = torch.tensor([-10.0], requires_grad=True)
    gn.gan_g_loss_nonsaturating(f_ns).backward()
    assert abs(float(f_sat.grad)) < 1e-3
    assert abs(float(f_ns.grad)) == pytest.approx(1.0, abs=1e-3)


def test_wgan_gp_components_exact():
    torch.manual_seed(0)
    critic = lambda t: t.sum(dim=1, keepdim=True)
    real, fake = torch.randn(8, 4), torch.randn(8, 4)
    gp = gn.gradient_penalty(critic, real, fake)
    assert float(gp) == pytest.approx((4 ** 0.5 - 1.0) ** 2, rel=1e-4)
    d = gn.wgan_gp_d_loss(lambda t: t.mean(dim=1, keepdim=True) * 0.0 + 1.0,
                          real, fake, lam=0.0)
    assert float(d) == pytest.approx(0.0, abs=1e-6)


def test_spectral_normalize_matches_svd():
    torch.manual_seed(0)
    W = torch.randn(6, 8)
    u = torch.randn(6)
    Wsn, _, sigma = gn.spectral_normalize(W, u, steps=200)
    true = float(torch.linalg.svdvals(W)[0])
    assert float(sigma) == pytest.approx(true, rel=0.02)
    assert float(torch.linalg.svdvals(Wsn)[0]) == pytest.approx(1.0, rel=0.02)


def test_discriminator_10_step_convergence():
    torch.manual_seed(0)
    disc = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(disc.parameters(), lr=1e-2)
    real = torch.randn(20, 2) + 2
    fake = torch.randn(20, 2) - 2
    losses = []
    for _ in range(10):
        opt.zero_grad()
        # Train D with YOUR minimax loss on separable blobs.
        loss = gn.gan_d_loss(disc(real).squeeze(1), disc(fake).squeeze(1))
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------- diffusion
def test_beta_schedules_shapes_and_ranges():
    lin = df.linear_beta_schedule(100)
    assert lin.shape == (100,)
    assert float(lin[0]) == pytest.approx(1e-4) and float(lin[-1]) == pytest.approx(0.02)
    assert bool((lin[1:] > lin[:-1]).all())
    cos = df.cosine_beta_schedule(100)
    assert cos.shape == (100,) and bool(((cos > 0) & (cos < 1)).all())


def test_q_sample_exact_spot_check():
    torch.manual_seed(0)
    sched = df.DiffusionSchedule(10, kind="linear")
    x0 = torch.tensor([[1.0, 2.0]])
    noise = torch.tensor([[0.5, -0.5]])
    t = torch.tensor([0])
    xt = sched.q_sample(x0, t, noise)
    import math
    sa, s1 = math.sqrt(1 - 1e-4), math.sqrt(1e-4)
    assert torch.allclose(xt, torch.tensor([[sa * 1.0 + s1 * 0.5, sa * 2.0 + s1 * -0.5]]))


def test_ddpm_ddim_shapes_and_determinism():
    torch.manual_seed(0)
    sched = df.DiffusionSchedule(10)
    x_t = torch.randn(2, 4)
    eps = torch.randn(2, 4)
    out = sched.ddpm_step(x_t, 0, eps, z=torch.zeros_like(x_t))
    assert out.shape == (2, 4)
    a = sched.ddim_step(x_t, 9, 5, eps, eta=0.0)
    b = sched.ddim_step(x_t, 9, 5, eps, eta=0.0)
    assert a.shape == (2, 4) and torch.equal(a, b)


def test_cfg_endpoints_and_mixing_exact():
    cond = torch.ones(2, 3)
    uncond = torch.zeros(2, 3)
    assert torch.equal(df.classifier_free_guidance(cond, uncond, w=0.0), uncond)
    assert torch.equal(df.classifier_free_guidance(cond, uncond, w=1.0), cond)
    assert torch.equal(df.classifier_free_guidance(cond, uncond, w=2.0), torch.full_like(cond, 2.0))


def test_epsilon_predictor_10_step_convergence():
    torch.manual_seed(0)
    sched = df.DiffusionSchedule(10)
    net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 4))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    x0 = torch.randn(16, 4)
    eps = torch.randn(16, 4)
    t = torch.full((16,), 5, dtype=torch.long)
    xt = sched.q_sample(x0, t, eps)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = ((net(xt) - eps) ** 2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]
