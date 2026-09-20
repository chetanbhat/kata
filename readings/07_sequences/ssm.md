# 7.3 — Diagonal SSMs & Mamba-lite Selective Scan

**Builds on:** 7.1 (the memory problem, answered without attention), 1.2
(discretization is applied calculus). **Builds toward:** linear-time
sequence modeling literacy (Mamba, Griffin, RWKV-adjacent ideas).

## Introduction

State-space models are control theory's answer to sequences: a continuous
linear system `dh/dt = Ah + Bx`, `y = Ch`, discretized to a recurrence.
**S4** (2022) showed that with the right `A` init (HiPPO: memorize history
optimally) a *linear* recurrence rivals attention on long-range benchmarks
— no quadratic cost. **Mamba** (2023) adds *selection*: make `Δ, B, C`
input-dependent so the scan keeps what's relevant and forgets the rest
(gating returns, wearing a trench coat). The exercise implements both
halves: exact ZOH discretization of a diagonal system (`A_d = e^{ΔA}`,
with the `A→0` limit handled), then the selective loop with a stable
negative diagonal `A` and skip connection `D`.

## The math that matters

- ZOH: `A_d = e^{ΔA}`, `B_d = (A_d−1)/A·B` — exact under piecewise-constant
  inputs. As `A→0`: `B_d → ΔB` (use the limit; the test pins it).
- Negative diagonal `A` ⇒ `|A_d| < 1` ⇒ bounded memory, no explosion by
  construction (compare Elman's prayers in 7.1).
- Selectivity = time-variance: fixed `(A,B,C)` is a convolution (parallel
  training); input-dependent breaks that — Mamba's hardware-aware scan
  recovers speed. The loop here is the honest O(T) version.

## Intuition check

Set `Δ` tiny: `A_d ≈ I`, memory persists (slow forgetting). Set it large:
`A_d ≈ 0`, memory resets (attend to now). Selection learns this knob per
step — plot `Δ` over a sequence and you can *read* what the model keeps.

## Readings

- Required: Gu et al., "Efficiently Modeling Long Sequences with S4" —
  [arxiv.org/abs/2111.00396](https://arxiv.org/abs/2111.00396)
- Recommended: Gu & Dao, "Mamba" —
  [arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)
- Suggested: Gu et al., "HiPPO" (the memory theory behind A's init) —
  [arxiv.org/abs/2008.07669](https://arxiv.org/abs/2008.07669)
