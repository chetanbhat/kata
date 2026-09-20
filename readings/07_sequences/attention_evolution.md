# 7.2 — Attention Variations (ALiBi, Windows, MQA/GQA)

**Builds on:** 4.3 (MHA + causal masking — now vary positions and heads).
**Builds toward:** reading real LLM code (GQA + RoPE/ALiBi + windows =
  LLaMA/Mistral inference stacks).

## Introduction

Full attention has two taxes: **positions** (absolute embeddings don't
extrapolate past training length) and **memory** (the KV cache grows with
heads × length, bottlenecking inference). **ALiBi** (2022) answers positions
with almost nothing: subtract `slope·distance` from scores — a linear
penalty, zero parameters, and models extrapolate to 2–4× training length.
Slopes are geometric (`2^(−8/n·i)`), giving heads different effective
windows. **Sliding windows** hard-cap the past (Mistral): O(T·w) instead of
O(T²), exact mask in the exercise. **MQA → GQA** (2019→2023) answers memory:
share KV heads across query groups (MQA = one KV head; GQA = e.g. 8 Q
heads per KV head), repeating KV to match. Quality barely moves; cache
shrinks ~8×. Architecture follows the bottleneck — here, memory bandwidth.

## The math that matters

- ALiBi bias `[i,j] = −m·max(0, i−j)`, `-inf` above diagonal: recency as an
  additive prior, causal by construction.
- GQA `repeat_interleave(H/KV)`: the *only* code change vs MHA — grouping
  is a reshape, not a new operation (read the forward pass with that lens).
- Sliding window + causal: attend `j ≤ i`, `i−j ≤ w`. Composing two
  constraints in one mask is the exercise's real skill.

## Intuition check

ALiBi slope `m=½` at distance 10 penalizes `e^−5 ≈ 1/148` — effectively a
soft window per head, hard-coded. Compare with the sliding mask: same
spirit, zero vs learned flexibility. Extrapolation tests (train 1k, eval
4k) separate position schemes the way the suite's corruption test
separates masks.

## Readings

- Required: Press et al., "Train Short, Test Long (ALiBi)" —
  [arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409)
- Recommended: Ainslie et al., "GQA: Training Generalized Multi-Query
  Transformers" — [arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)
- Suggested: Dao et al., "FlashAttention" (the IO-aware complement:
  exact attention, tiled) —
  [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
