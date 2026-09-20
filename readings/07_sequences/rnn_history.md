# 7.1 — Elman RNN, GRU, and the Vanishing Probe

**Builds on:** 4.2 (LSTM gates — GRU merges two of them), 2.1 (Jacobian
products are backprop unrolled). **Builds toward:** 7.3 (SSMs answer the
same memory problem linearly), 4.x reading (every modern block bypasses
this failure).

## Introduction

The Elman RNN (`h = tanh(xW + hU + b)`) looks complete and can't remember:
backprop through T steps multiplies T Jacobians, and with spectral radius
< 1 the product dies geometrically (Bengio 1994). The exercise makes this
a *measurement*: `grad_norm_vs_time` backprops from the last hidden state
and records `‖∂L/∂h_t‖` per step — early norms near zero *is* the 1990s.
The GRU (2014) compresses LSTM's three gates into two: update `z` (blend
old/new, merging forget+input) and reset `r` (drop history for the
candidate). Fewer parameters, same highway idea — and the probe shows
early gradients surviving where Elman's died.

## The math that matters

- `∂h_T/∂h_t = Π diag(1−tanh²)·U`: T matrix products. Norm < 1 per step →
  exponential decay; norm > 1 → explosion (the same equation gives both
  RNN pathologies; clipping treats the symptom).
- GRU: `h = (1−z)·n + z·h_prev`. With `z ≈ 1`, `∂h/∂h_prev ≈ I` — an
  *adaptive* identity path, vs ResNet's fixed one (4.1) and LSTM's cell (4.2).
- `recurrent_scale` in the exercise directly sets the spectral radius knob
  history argues about: 0.3 vanishes on cue.

## Intuition check

Vary `recurrent_scale` 0.3 → 1.5 and plot the norm curves: decay → flat →
growth. You have now reproduced, in one plot, thirty years of RNN
optimization literature.

## Readings

- Required: Pascanu et al., "On the difficulty of training RNNs"
  (vanishing + explosion + clipping) —
  [arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063)
- Recommended: Olah, "Understanding LSTM Networks" (gates visually) —
  [colah.github.io/posts/2015-08-Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Suggested: Lipton et al., "A Critical Review of RNNs" —
  [arxiv.org/abs/1506.00019](https://arxiv.org/abs/1506.00019)
