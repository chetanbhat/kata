# 2.1 — Mini-Autograd Engine (`Value` + `.backward()`)

**Builds on:** 1.2 (you hand-derived these backward passes; now systematize
them). **Builds toward:** everything in PyTorch — `loss.backward()` stops
being magic; Module 5's Hessian construction reuses the graph idea.

## Introduction

An autograd engine does two things: (1) *record* every op into a graph
while running forward, (2) *replay* local derivatives in reverse
topological order, multiplying by the chain rule. Each `Value` carries
`data`, `grad`, children, and a `_backward` closure capturing its local
derivative (e.g. for `mul`: `self.grad += other.data × out.grad`).
Seeding the output with `grad = 1` and walking backwards *is*
backpropagation — Module 1.2's algebra, mechanized.

## The math that matters

- Reverse-mode AD computes the full gradient in *one* backward pass for
  scalar outputs — why it's unbeatable for ML (many params, one loss).
  Forward-mode would need one pass *per parameter*.
- Topological order guarantees children accumulate before parents consume;
  forgetting it double-counts shared subgraphs (the classic first bug).
- Each op's `_backward` is just its partial derivatives: `add` copies,
  `mul` swaps, `pow` brings down the exponent, `tanh` gives `1−t²`.

## Intuition check

Build `(a·b + c).tanh()` with `a=2, b=−3, c=10` (the test's graph): forward
`4 → tanh(4)`; backward, `c`'s grad must equal `a`'s grad divided by `b`'s
data. If your numbers disagree, print every node's grad — the faulty op is
the first one whose grad is wrong walking back from the output.

## Readings

- Required: Karpathy, micrograd README + `engine.py` (the reference this
  exercise mirrors) —
  [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
- Recommended: Goodfellow et al., *Deep Learning*, §6.5 (backprop as
  reverse-mode AD) — [deeplearningbook.org](https://www.deeplearningbook.org)
- Suggested: Baydin et al., "Automatic Differentiation in Machine
  Learning: a Survey" — [arxiv.org/abs/1502.05767](https://arxiv.org/abs/1502.05767)
