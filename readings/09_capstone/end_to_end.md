# 9 — Capstone: Two-Moons End to End

**Builds on:** everything, especially 1.x (losses), 2.2 (MLP), 3.3 (loop
pattern), guides §2 (curves). **Builds toward:** independent projects —
swap moons for sklearn digits, then CIFAR, keeping this harness.

## Introduction

Exercises 1–8 are scales; this is the recital. The pipeline order is load-
bearing: **split first** (shuffled, disjoint, covering — indices returned
so you can *prove* it), **fit preprocessing on train only** (pooled stats
leak test information into training; the test compares your scaler's mean
against both), **monitor val while training** (curves, not vibes),
**early-stop + checkpoint the best** (the checkpoint *is* a regularizer),
**evaluate test once** (test-set continence — every re-peek invalidates
it). The width experiment (8 vs 32 vs 128) picks by *val*, reports test —
model selection done right, in miniature.

## The craft that matters

- Overfit deliberately first (small noisy set, big net): train ≫ val gap
  *diagnoses* capacity. Can't overfit 4 samples? Optimization is broken
  (guides §2, Story 4) — don't touch regularization yet.
- Early stopping needs a task that bottoms out: on clean moons val keeps
  improving (nothing to stop); on small+noisy it degrades (patience fires).
  The test uses both regimes — know which yours is in.
- Checkpoint reload equality (`argmax` predictions identical) is the
  minimum reproducibility contract. PNG curves exist so future-you can
  *see* what happened without rerunning.
- dtype discipline: scaler emits float32 into a float32 model (a real
  pipeline bug this exercise already caught once — Double vs Float matmul).

## Intuition check

Before running: predict which width wins val and by how much. After: if
128 wins *test* too, say why (smooth boundary, enough data); if 8 wins,
say why (noise punishes capacity). Wrong predictions beat no predictions —
calibration is the skill.

## Readings

- Required: Karpathy, "A Recipe for Training Neural Networks" —
  [karpathy.github.io/2019/04/25/recipe](https://karpathy.github.io/2019/04/25/recipe/)
- Recommended: ESL Ch. 7 (assessment & selection) —
  [web.stanford.edu/~hastie/ElemStatLearn](https://web.stanford.edu/~hastie/ElemStatLearn/)
- Suggested: Prechelt, "Early Stopping — But When?" —
  [arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
