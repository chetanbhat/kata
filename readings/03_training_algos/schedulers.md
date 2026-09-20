# 3.2 — LR Schedulers (Cosine Restarts, Warmup-Stable-Decay)

**Builds on:** 3.1 (the `lr` these schedules drive). **Builds toward:**
5.3 (flat basins love restarts + averaging), Module 09 (curves respond to
schedule shape).

## Introduction

A constant lr is a compromise: too big to settle, too small to travel.
Cosine annealing decays smoothly to `η_min`; *restarts* snap back to base
lr, letting the run escape sharp basins (each cycle can land somewhere
flatter — snapshot ensembling exploits exactly this). `T_mult` grows
successive cycles (explore early, refine late). Warmup-Stable-Decay is the
LLM-era answer: linearly ramp from 0 (early gradients are noisy and large
— starting at full lr can kill a run on step one), hold steady, then cosine
down. The test pins each phase's shape: rising, flat, falling.

## The math that matters

- Cosine: `η_min + ½(base−η_min)(1+cos(π·T_cur/T_i))`. At `T_cur=0` you get
  base; at `T_cur=T_i` you get `η_min`. Restart = reset `T_cur`.
- Warmup is *linear* `base·(t+1)/warmup` — gentle enough to survive
  initialization transients, short enough not to waste budget.
- The schedule is a pure function of step count (no gradients involved) —
  implement it standalone, then `attach()` it to an optimizer.

## Intuition check

Plot your schedule before training (10 lines of matplotlib). If the curve
doesn't look like the description — monotone decay within a cycle, jump at
restart — the bug is in cycle bookkeeping (`T_mult` growth is the usual
suspect), not in training.

## Readings

- Required: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with
  Warm Restarts" — [arxiv.org/abs/1608.03983](https://arxiv.org/abs/1608.03983)
- Recommended: Goyal et al., "Accurate, Large Minibatch SGD" (warmup for
  large batches) — [arxiv.org/abs/1706.02677](https://arxiv.org/abs/1706.02677)
- Suggested: Smith, "Cyclical Learning Rates" —
  [arxiv.org/abs/1506.01186](https://arxiv.org/abs/1506.01186)
