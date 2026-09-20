# 6.1 — Regularization Zoo (DropConnect, Stochastic Depth, Smoothing, Mixup)

**Builds on:** 2.2 (dropout mechanics), 1.3 (CE targets get *softened*
here). **Builds toward:** 8.2 (regularized discriminators stabilize GANs),
Module 09 (these move val curves, never train curves — the diagnostic).

## Introduction

Regularization is controlled damage. **DropConnect** (2013) masks *weights*
instead of activations — finer-grained noise, same inverted scaling.
**Stochastic Depth** (2016) drops *entire residual branches* during
training: effectively trains an ensemble of depths, tests at full depth
(the eval identity in the tests is the point). **Label smoothing** replaces
the one-hot target with `(1−ε)` on truth + `ε/K` elsewhere — penalizes
overconfidence, and `ε=0` must *exactly* equal plain CE (pinned test).
**Mixup** (2017) trains on convex combos of pairs *and* labels — data
outside the manifold, behaving better on it. Each method answers "what
should the model *not* rely on": single activations, single weights,
single layers, hard labels, single examples.

## The math that matters

- Inverted scaling (`÷(1−p)`) preserves expectations everywhere in this
  file — eval stays the identity *by construction*.
- Smoothed loss = `(1−ε)·NLL + ε·mean(−log p)`: the second term punishes
  peaky distributions (a crude entropy bonus).
- Mixup criterion: `λ·L(pred, y_a) + (1−λ)·L(pred, y_b)` with `λ ∼ Beta(α,α)`.
  `α=1` is uniform; larger α concentrates near ½ (stronger mixing).

## Intuition check

Train the capstone twice: once plain (overfits), once with smoothing +
Mixup. The train curve gets *worse* and val gets *better* — regularization
working as intended feels like losing. Beginners regularize underfit
models; check train error first (guides/01 §4).

## Readings

- Required: Srivastava et al., "Dropout" (arxiv version) —
  [arxiv.org/abs/1207.0580](https://arxiv.org/abs/1207.0580)
- Recommended: Zhang et al., "mixup" —
  [arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412)
- Suggested: Szegedy et al., "Rethinking the Inception Architecture"
  (label smoothing's origin) —
  [arxiv.org/abs/1512.00567](https://arxiv.org/abs/1512.00567)
