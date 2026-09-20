# 8.2 — GAN Losses (Minimax → Non-saturating → WGAN-GP + SpectralNorm)

**Builds on:** 1.3 (BCE from logits), 6.2 (spectral norm, gradient penalty
— now their *purpose*). **Builds toward:** 8.3 (diffusion won by being
stabler than this; know what it replaced).

## Introduction

The GAN game: D maximizes `log D(x) + log(1−D(G(z)))`, G minimizes the
second term. Early in training D confidently rejects (`fake logit ≪ 0`),
and there the original G loss is *flat* — `log(1−D)` saturates, gradients
vanish, G never learns (the suite exhibits this as a gradient test at
logit −10: minimax grad ≈ 0, non-saturating grad ≈ 1). The same paper's
pragmatic fix: maximize `log D(G(z))` instead — wrong game, working
gradients. **Mode collapse** (G emits one plausible sample forever) comes
from the Jensen–Shannon dynamics under disjoint supports. **WGAN** swaps JS
for Wasserstein distance (meaningful curves even disjoint); **WGAN-GP**
replaces capacity-crippling weight *clipping* with the gradient penalty
from 6.2. **SpectralNorm** discriminators bound sensitivity architecturally.

## The math that matters

- `D_loss = −(logσ(real) + logσ(−fake))`; at all-zero logits it is exactly
  `2·0.6931` (hand-check in the tests).
- `G_nonsat = −logσ(fake)`; `G_minimax = logσ(−fake)` — same optimum at
  infinity, opposite early-training behavior. Direction of the *gradient*,
  not the loss value, is the pathology.
- WGAN-GP critic loss: `E[fake] − E[real] + λ·GP`. No logs, no sigmoids —
  the critic is a *function*, not a classifier.

## Intuition check

Plot both G losses vs fake logit from −10 to +10: minimax flat on the
left, steep on the right; nonsaturating the reverse. Training lives on the
left early — now you see *why* the original GAN stalls and the fix unstalls.

## Readings

- Required: Goodfellow et al., "Generative Adversarial Nets" —
  [arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661)
- Recommended: Gulrajani et al., "Improved Training of Wasserstein GANs" —
  [arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)
- Suggested: Karras et al., "StyleGAN" (what stabilized GANs can do) —
  [arxiv.org/abs/1812.04948](https://arxiv.org/abs/1812.04948)
