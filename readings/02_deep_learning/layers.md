# 2.2 — Custom Layers: Linear, BatchNorm, LayerNorm, Dropout

**Builds on:** 1.2 (activations, init scale), 1.3 (stability habits).
**Builds toward:** 4.x (these compose into ResNet/LSTM/Transformer), 6.2
(normalization history re-examines BatchNorm).

## Introduction

Four primitives, four lessons. **Linear + init**: random weights must
preserve activation variance across depth or signals die/explode — He
(`√(2/n)`, for ReLU's halved variance) and Xavier (`√(1/n)`, symmetric
activations) are variance bookkeeping, not magic. **BatchNorm**: normalize
by batch statistics in train mode, frozen running stats in eval — the
train/eval asymmetry is its #1 beginner trap (and why eval uses different
code paths). **LayerNorm**: normalize over features, not batch — batch-size
independent, hence the Transformer default. **Dropout**: inverted scaling
(`÷(1−p)` in train) keeps expectations matched so eval is the identity.

## The math that matters

- Variance propagation: `Var(Wx) = fan_in·Var(w)·Var(x)` ⇒ set
  `Var(w) = 1/fan_in` (Xavier) or `2/fan_in` (He, ReLU kills half).
- BatchNorm: `γ·(x−μ_B)/√(σ²_B+ε) + β`, with running `μ, σ²` updated by
  momentum — learnable affine (`γ, β`) restores representational power.
- LayerNorm output has exactly mean 0 / variance 1 per row at init (the
  test asserts this) — then affine shifts it.
- Inverted dropout preserves the *mean* activation (test: mean ≈ 1 over
  20k draws), which is why no eval-time rescaling is needed.

## Intuition check

Train a Linear layer and watch running stats move (BatchNorm) vs stay
(layer norms have no running stats). If eval accuracy craters while train
is fine, suspect the running stats (tiny batches) — Module 09's version of
this is forgetting `model.eval()`.

## Readings

- Required: He et al., "Delving Deep into Rectifiers" (He init) —
  [arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
- Recommended: Ba et al., "Layer Normalization" —
  [arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
- Suggested: Srivastava et al., "Dropout" (arxiv version) —
  [arxiv.org/abs/1207.0580](https://arxiv.org/abs/1207.0580)
