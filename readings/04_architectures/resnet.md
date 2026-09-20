# 4.1 — Residual Blocks (Basic + Bottleneck, Pre/Post Activation)

**Builds on:** 2.2–2.3 (conv + BN + ReLU are the Lego bricks). **Builds
toward:** 6.1 (stochastic depth drops these branches), 7.x (highways
reappear as gates and attention bypasses).

## Introduction

Plain nets degrade past ~20 layers: not overfitting — *optimization*
fails, gradients vanish through long chains. The residual fix is almost
insultingly simple: learn `F(x)` and output `F(x) + x`. If identity is
optimal, the layer drives `F → 0` (easy); gradients get a clean shortcut
path backwards (the `+x` term differentiates to 1). Dimension mismatch
(stride/channel change) needs a 1×1 projection shortcut. Bottlenecks
(1×1 reduce → 3×3 → 1×1 expand) cut FLOPs so depth can scale to 100+.
Pre-activation (BN–ReLU–Conv ordering) keeps the shortcut path *clean* —
every norm in the exercise is sized so all parameters participate.

## The math that matters

- Backward through `y = F(x) + x`: `dy/dx = dF/dx + I`. The identity term
  guarantees gradient flow no matter how broken `F` is.
- Projection shortcut: 1×1 conv matching channels/stride — used *only* on
  mismatch; identity elsewhere (projections everywhere would defeat the
  purpose and add parameters pointlessly).
- Bottleneck economics: two 1×1 convs around one 3×3 cost ~`1/4` of two
  3×3s at the same width — ResNet-50+ is bottleneck-only for this reason.

## Intuition check

Zero-initialize the last BN of a block: the block starts as identity and
training *adds* residual corrections. If your block can't learn identity
initially, depth will hurt instead of help.

## Readings

- Required: He et al., "Deep Residual Learning" —
  [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- Recommended: He et al., "Identity Mappings in Deep Residual Networks"
  (why pre-activation) —
  [arxiv.org/abs/1603.05027](https://arxiv.org/abs/1603.05027)
- Suggested: Huang et al., "Densely Connected Networks" (skip → dense) —
  [arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)
