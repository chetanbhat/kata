# 1.3 — Losses & Numerical Stability

**Builds on:** 1.1 (BCE inside logistic GD), guides/01 §3 (softmax by hand).
**Builds toward:** every module after this — these three functions are the
objectives of Modules 3, 5, 8, and 09.

## Introduction

A loss function is a *contract*: MSE says errors cost quadratically (and
punishes outliers brutally); BCE says "match these probabilities";
cross-entropy from logits says "put mass on the right class." The exercise
looks trivial and isn't: naive implementations overflow (`e^1000`), underflow
(`log 0`), or silently average over the wrong axis. The log-sum-exp trick —
subtract the row max before exponentiating — is the single most reused
numerical idiom in deep learning (it reappears in attention, Module 4).

## The math that matters

- `softmax(z)_i = e^(z_i−m)/Σe^(z_j−m)`, `m = max(z)`: identical math,
  finite floats. Always stabilize *before* exponentiating.
- `CE = −log p_true`; gradient w.r.t. logits is `p − one_hot` (sums to 0 —
  classes compete; guides/01 §3 worked example).
- BCE needs clipped probabilities `[ε, 1−ε]`; logits-based BCE
  (`BCEWithLogitsLoss`) fuses sigmoid+loss for the same stability reason.

## Intuition check

Logits `[1000, 1001]`: your code must return a finite ~0.31 for class 1.
If it returns `nan`, you exponentiated before shifting. The stability tests
are exactly these adversarial inputs.

## Readings

- Required: Goodfellow et al., *Deep Learning*, §5.5–5.6 (likelihood losses)
  — [deeplearningbook.org](https://www.deeplearningbook.org)
- Recommended: PyTorch `CrossEntropyLoss` docs (fused, weighted, ignore_index
  semantics you'll meet in Module 09) —
  [pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- Suggested: Blanchard et al., "Log-sum-exp trick" references in
  Murphy, *Probabilistic ML* §2 (any library copy).
