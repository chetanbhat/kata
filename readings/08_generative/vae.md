# 8.1 — VAE (ELBO + Reparameterization + the β Knob)

**Builds on:** 1.3 (CE/BCE, KL facts in guides/01 §5), 1.2 (MLP encoder/
decoder). **Builds toward:** 8.3 (diffusion also trains a denoiser with a
weighted ELBO — same skeleton).

## Introduction

Exact posterior inference `p(z|x)` is intractable, so VAEs maximize a lower
bound instead: `ELBO = E[log p(x|z)] − KL(q(z|x)‖p(z))` — reconstruction
minus a leash pulling the posterior toward the prior. The
**reparameterization trick** (`z = μ + σ·ε`, `ε∼N(0,I)`) moves randomness
*outside* the differentiable path so gradients flow to `μ, logσ²` (the
test asserts `μ.grad == 1` exactly). The famous pathology: **posterior
collapse** — a strong decoder + full KL weight drives KL→0 and `z` is
ignored (Bowman 2016). The `β` knob is the annealing handle (β-VAE
schedules, free bits); `β=0 ⇒ loss == recon` is pinned so the knob's
mechanics stay honest.

## The math that matters

- Closed-form Gaussian KL: `−½Σ(1 + logσ² − μ² − σ²)`. At `μ=0,σ=1` it is
  exactly 0 (prior matched); at `μ=1` it is 0.5 (the test's hand-check).
- Reconstruction for binary data = summed BCE from logits; total loss is a
  *mean over batch of sums over dims* — get the reduction order right or
  `β` means different things at different batch sizes.
- Sampling (`randn`) breaks determinism: tests assert gradient *shapes*,
  not values, around `reparameterize` — know which assertions survive
  randomness.

## Intuition check

Train with β=0 (autoencoder, sharp reconstructions, useless latents), then
β=1 (blurrier, structured latents). The trade *is* the ELBO tension;
interpolating latents only works in the second regime.

## Readings

- Required: Kingma & Welling, "Auto-Encoding Variational Bayes" —
  [arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
- Recommended: Bowman et al., "Generating Sentences from a Continuous
  Space" (posterior collapse documented) —
  [arxiv.org/abs/1511.06349](https://arxiv.org/abs/1511.06349)
- Suggested: Higgins et al., "β-VAE" (disentangling via the β knob) —
  [arxiv.org/abs/1804.03599](https://arxiv.org/abs/1804.03599)
