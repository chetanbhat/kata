# 6.2 — WeightNorm, SpectralNorm, Gradient Penalty

**Builds on:** 2.2 (BatchNorm mechanics — now the *story* gets questioned).
**Builds toward:** 8.2 (these are GAN stabilizers), Module 09 (norm choices
change trainability, not just accuracy).

## Introduction

Three answers to "control the function's scale." **WeightNorm** (2016)
reparameterizes each row `w = g·v/‖v‖`: learn norm (`g`) and direction
(`v`) separately — the test's exact `‖wᵢ‖ = gᵢ` pins the decoupling, and
optimization speeds up because length/angle stop fighting. **SpectralNorm**
(2018) divides by the top singular value (power iteration on buffer `u`),
enforcing 1-Lipschitz layers — born for GAN discriminators, now standard
wherever sensitivity must be bounded. **Gradient penalty** (WGAN-GP, 2017)
enforces Lipschitz *where it matters*: random interpolations between real
and fake, penalizing `(‖∇C‖₂−1)²`. And the history lesson: BatchNorm's
"internal covariate shift" explanation didn't survive scrutiny (Santurkar
2018: it smooths the landscape) — keep techniques, distrust first stories.

## The math that matters

- Power iteration converges to the top singular pair geometrically
  (rate = ratio of top two singular values); a few steps per forward pass
  suffices because `u` persists across steps (warm-started).
- A linear critic `C(x) = Σx` has gradient norm `√d` everywhere → GP equals
  `(√d−1)²` exactly (the test's hand-check at `d=4` gives 1.0).
- WGAN's weight *clipping* also enforced Lipschitz — crudely, crippling
  capacity. Penalty > constraint is a recurring upgrade pattern.

## Intuition check

Track σ(W) during training with and without spectral norm: unconstrained,
it drifts; constrained, the layer's Lipschitz constant stays 1 while
directions stay free. Scale control without capacity loss is the whole game.

## Readings

- Required: Ioffe & Szegedy, "Batch Normalization" —
  [arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
- Recommended: Santurkar et al., "How Does Batch Normalization Help
  Optimization" — [arxiv.org/abs/1805.11604](https://arxiv.org/abs/1805.11604)
- Suggested: Miyato et al., "Spectral Normalization for GANs" —
  [arxiv.org/abs/1802.05957](https://arxiv.org/abs/1802.05957)
