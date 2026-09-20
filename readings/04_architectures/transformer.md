# 4.3 — Attention, RoPE, and the GPT Decoder Block

**Builds on:** 1.3 (softmax over scores), 2.2 (norms, dropout), 4.2
(sequences without recurrence). **Builds toward:** 7.2 (ALiBi, GQA,
windows vary this block), 8.3 (diffusion backbones inherit it).

## Introduction

Attention replaces recurrence with routing: each position queries all
others (`softmax(QK'/√d)V`). Scaling by `√d` isn't cosmetic — dot products
grow with dimension, saturating softmax and killing gradients (same
saturation story as Module 8's GAN). Multi-head runs several routing
subspaces in parallel. Causal masking (lower-triangular `-inf`) enforces
autoregression: position `t` never sees the future — the corruption test in
the suite *proves* it by scrambling future values. RoPE rotates query/key
pairs by position-dependent angles, baking *relative* position into the dot
product (absolute positions don't extrapolate; relative ones do). RMSNorm
drops mean-centering (cheaper, works as well). The GPT block: pre-norm →
masked MHA → residual → pre-norm → MLP → residual.

## The math that matters

- `Attention = softmax(QK'/√d + mask)V`. Forget the `√d` and training dies
  at large widths — scale is load-bearing.
- RoPE: rotating pairs `(x₁,x₂)` by angle `m·θ` makes `q·m k` depend only
  on relative distance — rotation preserves norms (the test asserts this).
- Pre-norm (norm *inside* the residual branch) trains deeper stacks than
  post-norm; the original Transformer used post-norm and later models
  quietly switched.
- Causal test design: perturb `V[:, 3:]`, assert outputs `[:, :3]`
  unchanged — *behavioral* masking proof, stronger than inspecting weights.

## Intuition check

With uniform attention weights, output = mean of values. Sanity-check any
attention implementation against this degenerate case before trusting the
masking tests.

## Readings

- Required: Vaswani et al., "Attention Is All You Need" —
  [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Recommended: Alammar, "The Illustrated Transformer" —
  [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
- Suggested: Harvard NLP, "The Annotated Transformer" (code + paper) —
  [nlp.seas.harvard.edu/annotated-transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
