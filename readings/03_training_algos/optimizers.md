# 3.1 — Optimizers from Scratch (SGD, Momentum, Nesterov, Adam)

**Builds on:** 1.1 (the logistic GD loop), 2.1 (`.grad` consumption).
**Builds toward:** 5.2 (the full adaptive lineage), Module 09 (you will
compare these on real curves).

## Introduction

Plain SGD follows the gradient; momentum adds a velocity that damps
oscillation along steep directions and accelerates along consistent ones
(visualize a ravine: SGD bounces wall-to-wall, momentum rolls down the
middle). Nesterov evaluates the gradient *after* the momentum step — a
lookahead that corrects overshoot. Adam keeps per-parameter step sizes from
moment estimates: `m` (direction, like momentum) and `v` (scale, like
RMSprop), with bias correction dividing out the zero-initialization
transient (`m/(1−β₁ᵗ)`). The bias-correction test is exact for a reason:
it is the one line everyone re-derives wrong.

## The math that matters

- Momentum: `v ← μv + g; p ← p − lr·v`. Effective step grows `1/(1−μ)×`
  along persistent directions — which is why momentum needs a *smaller* lr
  than intuition suggests.
- Nesterov: `p ← p − lr·(g + μv)` — gradient at the lookahead point
  approximates the future; provably optimal for smooth convex problems.
- Adam: `m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)`, `p ← p − lr·m̂/(√v̂+ε)`. Early
  steps (`t` small) are dominated by the correction — without it, step
  sizes start near zero and warm up spuriously.

## Intuition check

One Adam step on `p=1, g=2`: `m̂=2, v̂=4`, update `≈ lr·2/(2+ε) ≈ lr`.
The test pins this. If your implementation moves any other amount, the
correction or the denominator is wrong.

## Readings

- Required: Ruder, "An overview of gradient descent optimization
  algorithms" — [arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
- Recommended: Kingma & Ba, "Adam" —
  [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- Suggested: Sutskever et al., "On the importance of initialization and
  momentum" — [proceedings.mlr.press/v28/sutskever13](https://proceedings.mlr.press/v28/sutskever13.html)
