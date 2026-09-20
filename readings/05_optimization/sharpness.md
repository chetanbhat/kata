# 5.3 — SAM & SWA (Seeking Flat Minima)

**Builds on:** 3.1/5.2 (both wrap a base optimizer), guides/01 §4
(bias–variance: flatness is the mechanism). **Builds toward:** Module 09
(checkpoint *selection* by val is SWA's poor cousin).

## Introduction

Large-batch training found sharp minima that generalize poorly (Keskar
2017) — same train loss, worse test. Two responses. **SAM** (2020):
minimize the *worst* loss in a ρ-ball around w — ascend to `w+ε`
(`ε = ρ·g/‖g‖`), compute gradients *there*, step, restore. Costs 2× grads,
buys flatter basins. The ascent check in the tests (`first_step` must
*raise* the loss) is the implementation's honesty pledge. **SWA** (2017):
average SGD iterates — the mean of a basin's rim sits in its flat middle.
Nearly free, surprisingly strong, and the conceptual parent of model soups
and EMA teacher models.

## The math that matters

- SAM's ε is the steepest-ascent direction *normalized*: fixed radius ρ,
  adaptive direction. ρ too big optimizes paranoia; ~0.05 is the folk default.
- SWA's running average `avg += (p−avg)/n` needs a cyclical or high
  constant lr to *collect diverse* rim points — averaging one trajectory's
  tail does little.
- Flatness ↔ generalization is empirical, not theorem — know the evidence
  (Keskar) and the skeptics (reparameterization can sharpen any minimum
  without changing the function).

## Intuition check

On a 1-D double-well loss, SAM's perturbation pushes iterates out of the
sharp well; SWA averages two rim points into the flat one's interior. Draw
it — both methods are geometric before they're algebraic.

## Readings

- Required: Foret et al., "Sharpness-Aware Minimization" —
  [arxiv.org/abs/2010.01412](https://arxiv.org/abs/2010.01412)
- Recommended: Izmailov et al., "Averaging Weights Leads to Wider Optima
  (SWA)" — [arxiv.org/abs/1803.05407](https://arxiv.org/abs/1803.05407)
- Suggested: Keskar et al., "On Large-Batch Training" (the motivating
  observation) — [arxiv.org/abs/1611.03530](https://arxiv.org/abs/1611.03530)
