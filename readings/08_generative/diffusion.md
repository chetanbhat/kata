# 8.3 — Diffusion (Schedules, DDPM/DDIM Steps, Guidance)

**Builds on:** 8.1 (weighted-ELBO thinking), 1.3 (MSE on noise is the whole
training signal), 4.3 (the denoiser backbone). **Builds toward:** reading
modern generative stacks (schedules + samplers + guidance compose in every
codebase).

## Introduction

Forward: destroy data with Gaussian noise on a fixed schedule (`q(x_t|x_0)`
in closed form — no simulation needed). Reverse: train a network to predict
the *noise* `ε` from `x_t` (epsilon parameterization turns denoising into
plain MSE — the suite's 10-step convergence target). **DDPM** (2020) steps
back through the full Markov chain with posterior-mean coefficients.
**DDIM** (2020) observed the chain was optional: deterministic, strided
jumps (`η=0`, exact reproducibility in the tests) with near-identical
quality in ~50 steps instead of 1000. **Schedules** matter physically:
linear destroys low-res signal too fast; cosine (2021) paces it.
**Classifier-free guidance** steers sampling: `ε = ε_uncond + w(ε_cond −
ε_uncond)` (`w=0` → unconditional, `w=1` → conditional, pinned exactly).

## The math that matters

- `x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε`: one line, any `t`, no loop — the spot-check
  test verifies coefficient gathering (the `#1 sampler bug is indexing `ᾱ`).
- DDPM posterior mean mixes `x̂_0` (from ε-prediction) and `x_t` with
  closed-form coefficients; at `t=0` the mean *is* the answer (no noise).
- DDIM `η` interpolates deterministic ↔ stochastic: `η=0` same input same
  output (test equality), `η=1` ≈ DDPM.
- Guidance scale trades fidelity for diversity (`w≈7.5` folk default);
  too high saturates/warps — visible in one sampling run.

## Intuition check

Predict noise at fixed `t` on 16 points: MSE must fall in 10 steps (it
does — pure regression). Then vary `t` per sample: harder, the network must
condition on time. That gap motivates time embeddings in real denoisers.

## Readings

- Required: Ho et al., "Denoising Diffusion Probabilistic Models" —
  [arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Recommended: Song et al., "DDIM" —
  [arxiv.org/abs/2010.02502](https://arxiv.org/abs/2010.02502)
- Suggested: Ho & Salimans, "Classifier-Free Guidance" —
  [arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)
