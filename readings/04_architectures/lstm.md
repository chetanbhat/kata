# 4.2 — LSTM Cell from Raw Gates

**Builds on:** 1.2 (sigmoid/tanh by hand), 2.1 (every gate's backward is
these patterns). **Builds toward:** 7.1 (GRU merges gates; vanishing probe
quantifies *why* gates exist).

## Introduction

RNNs fail at long memory because backprop multiplies the *same* recurrent
Jacobian every step — norms shrink geometrically (Module 7 measures this).
The LSTM's answer is an additive cell state: `c_t = f·c_{t−1} + i·g`. When
the forget gate `f ≈ 1`, gradient flows back through time *unchanged* —
a gradient highway with learned on/off ramps. Input gate `i` writes,
output gate `o` reads (`h = o·tanh(c)`). The forget-bias-to-1 trick in the
exercise is a real folk discovery: initialize remembering, learn to forget.

## The math that matters

- All four gates come from one fused matmul, `chunk`ed — one big GEMM
  beats four small ones on GPUs (implementation detail with teeth).
- `∂c_t/∂c_{t−1} = f_t` (elementwise): the network *controls its own
  gradient decay*. Compare with Elman's fixed `W_hh` (Module 7).
- Candidate `g` is tanh (signed write); gates are sigmoid (0–1 valves).
  Swapping them is the canonical beginner bug — shapes still match, which
  is why the convergence test (not just shapes) matters.

## Intuition check

Clamp `f = 1, i = 0` manually: the cell becomes perfect memory (`c`
frozen). Your implementation should reproduce this in two lines — if not,
the gate wiring is wrong.

## Readings

- Required: Olah, "Understanding LSTM Networks" (best visual build-up) —
  [colah.github.io/posts/2015-08-Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Recommended: Goodfellow et al., *Deep Learning*, §10.10 (gated RNNs) —
  [deeplearningbook.org](https://www.deeplearningbook.org)
- Suggested: Greff et al., "LSTM: A Search Space Odyssey" (which parts
  actually matter) — [arxiv.org/abs/1503.04069](https://arxiv.org/abs/1503.04069)
