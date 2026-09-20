# 3.3 — Training Loop (Clipping, Accumulation, AMP)

**Builds on:** 3.1–3.2 (optimizer + scheduler plug in here). **Builds
toward:** Module 09 (this loop *becomes* the capstone harness, plus data
and validation).

## Introduction

The loop is where theory meets hardware limits. **Gradient clipping**
(rescale if global norm > max) prevents one bad batch from detonating the
run — essential for RNNs/Transformers where attention logits can spike.
**Accumulation** simulates big batches on small GPUs: sum K micro-batch
gradients, step once (remember to divide the loss by K). **Mixed precision**
does matmuls in fp16 with a loss *scaler* that shifts gradients into
representable range, unscaling before clipping. On CPU the AMP hooks in
this exercise safely no-op — the structure is what you're learning.

## The math that matters

- Clipping by norm: `g ← g·min(1, max_norm/‖g‖)`. Direction preserved,
  magnitude capped — unlike value-clipping, it never changes *where* you go.
- Accumulation equivalence: `Σ(Loss_i/K)` backwarded K times = one batch
  of size K (BatchNorm statistics excepted — the fine print).
- Loss scaling: fp16 underflows grads below ~6e-5; scaling the loss by S
  scales grads by S, then unscale before the optimizer step. Overflow →
  skip the step and shrink S (dynamic scaling).

## Intuition check

Feed the loop 100×-scaled inputs with `max_grad_norm=0.5`: loss must stay
finite. Then remove clipping and watch it diverge — that contrast *is* the
lesson. The test asserts finiteness, not convergence.

## Readings

- Required: Micikevicius et al., "Mixed Precision Training" —
  [arxiv.org/abs/1710.03741](https://arxiv.org/abs/1710.03741)
- Recommended: Karpathy, "A Recipe for Training Neural Networks"
  (the loop in its natural habitat) —
  [karpathy.github.io/2019/04/25/recipe](https://karpathy.github.io/2019/04/25/recipe/)
- Suggested: Pascanu et al., "On the difficulty of training RNNs"
  (gradient clipping's origin) —
  [arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063)
