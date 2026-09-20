# 1.2 — Perceptron & MLP (manual backprop)

**Builds on:** 1.1 (linear units, sigmoid), guides/01 §2 (chain rule).
**Builds toward:** 2.1 (autograd mechanizes this file), 4.2/4.3 (gates and
attention reuse these exact backward patterns).

## Introduction

The perceptron (Rosenblatt, 1958) learns only what a hyperplane can
express — Minsky & Papert's XOR critique killed it in 1969. The MLP
answers by stacking: linear → nonlinearity → linear. Depth buys
*compositional* expressivity (each layer builds features from the last),
and backprop (1986) buys a training procedure. The stub walks you through
both in five ordered steps with a hand-checkable micro-example
(`dW1 = 8, dW2 = 24` — reproduce it before coding).

## The math that matters

- Forward cache per layer: `(Z, A_prev)`. Memory for speed — the defining
  trade of backprop (revisited as gradient checkpointing in large models).
- Output grad for mean-MSE: `dA = 2(out − y)/N`. Hidden: `dZ = dA·φ'(Z)`;
  `dW = A_prev'·dZ`; `dB = ΣdZ`; `dA = dZ·W'`. Learn this loop cold.
- Init scale matters: He `√(2/fan_in)` keeps ReLU variances stable;
  too-large init saturates sigmoids on step one (dead network, Story 3).
- GeLU is smooth ReLU (`x·Φ(x)`); its tanh approximation exists because
  `erf` was historically expensive — know *why* the formula looks odd.

## Intuition check

Perturb one weight by `1e-5`: if the loss doesn't move, that parameter is
decorative. The test suite's finite-difference probe is this habit,
automated.

## Readings

- Required: Goodfellow et al., *Deep Learning*, Ch. 6 (MLPs) —
  [deeplearningbook.org/contents/mlp.html](https://www.deeplearningbook.org/contents/mlp.html)
- Recommended: Karpathy, "Zero to Hero" lecture 1 + micrograd build —
  [github.com/karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
- Suggested: Rumelhart, Hinton & Williams (1986), "Learning representations
  by back-propagating errors," *Nature* (the paper that restarted the field).
