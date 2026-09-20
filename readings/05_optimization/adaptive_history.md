# 5.2 — Adaptive Lineage (AdaGrad → RMSprop → AdamW → Lion → Lookahead)

**Builds on:** 3.1 (SGD/Adam mechanics). **Builds toward:** 5.3 (flatness
add-ons wrap these), Module 09 (you compare them on curves).

## Introduction

Each optimizer here fixes one predecessor's sin. **AdaGrad** (2011):
per-coordinate scaling by all past squared grads — elegant, but the
denominator grows forever and learning halts on long runs. **RMSprop**
(Hinton, unpublished lecture): replace the sum with an exponential moving
average — non-stationary allowed. **Adam** (2014): add momentum + bias
correction — the default that ate the world. **AdamW** (2017): Adam's dirty
secret was folding weight decay *into* the adaptive denominator (no longer
L2); decoupling restores honest regularization — the zero-grad exact test
(`p ×= 1−lr·wd`) is the whole paper in one line. **Lion** (2023):
evolution-discovered sign updates match AdamW with less memory — the design
space isn't closed. **Lookahead** (2019): slow/fast weights around *any*
optimizer — exploration with a leash.

## The math that matters

- AdaGrad's effective lr decays `~1/√t` *per coordinate, monotonically* —
  plot it once and you'll never trust unbounded accumulation again.
- Adam's bias correction matters most at `t < 100`; most "Adam diverges
  early" stories are correction bugs or missing warmup (Module 3.2).
- Lion's first step from zero momentum is exactly `−lr·sign(g)` (test) —
  the `β₁` interpolation only matters once memory accumulates.
- Lookahead sync (`slow += α(fast−slow)` every k) is SWA's cousin (5.3).

## Intuition check

Run all four adaptive methods on the same quadratic and plot trajectories:
AdaGrad stalls, RMSprop/Adam converge, AdamW with wd≠0 converges *to a
different point* (shrinkage). Same loss, different minima — optimizers are
implicit regularizers.

## Readings

- Required: Kingma & Ba, "Adam" —
  [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- Recommended: Loshchilov & Hutter, "Decoupled Weight Decay (AdamW)" —
  [arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
- Suggested: Reddi et al., "On the Convergence of Adam and Beyond"
  (when Adam provably fails) —
  [arxiv.org/abs/1804.05571](https://arxiv.org/abs/1804.05571)
