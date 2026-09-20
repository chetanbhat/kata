# Module 05 — Optimization History & Second-Order Methods

## Where this sits

Module 03 taught the working defaults; this module asks *why they are the
defaults* by walking the graveyard: full Newton (correct, unaffordable),
AdaGrad (elegant, self-strangling), coupled Adam (subtly wrong). Plus the
modern add-ons that buy generalization: flatness-seeking (SAM) and
iterate-averaging (SWA).

## Builds on

- Module 03 (SGD/Adam mechanics — now compare their failure modes).
- Module 01 (quadratics as the laboratory: exact Newton in one step).

## Builds toward

- Research taste: optimizer papers are mostly "one assumption relaxed" —
  after this module you can read new ones critically (see Lion notes).
- Module 09 (you will *feel* Adam vs SGD on learning curves).

## The arc

Second-order methods ruled small problems but never scaled (O(n³) solves,
indefinite Hessians) → quasi-Newton kept the idea alive for batch settings
→ stochastic first-order won deep learning → adaptive methods fixed scaling
→ AdamW fixed Adam's regularization bug → sign-based Lion questioned the
whole edifice. Pendulum, not ladder.

## Exercises

| Note | Exercise | Question it answers |
|------|----------|---------------------|
| [second_order.md](second_order.md) | `exercises/05_optimization/second_order.py` | Newton, damping, L-BFGS two-loop |
| [adaptive_history.md](adaptive_history.md) | `exercises/05_optimization/adaptive_history.py` | AdaGrad→RMSprop→AdamW→Lion→Lookahead |
| [sharpness.md](sharpness.md) | `exercises/05_optimization/sharpness.py` | Flat minima: SAM perturb, SWA average |

## Section readings

- Required: Nocedal & Wright, *Numerical Optimization*, Ch. 3, 6–7 (the
  canon; any library copy).
- Recommended: Martens, "Deep learning via Hessian-free optimization" —
  [arxiv.org/abs/1004.1630](https://arxiv.org/abs/1004.1630)
- Suggested: Foret et al., "Sharpness-Aware Minimization" —
  [arxiv.org/abs/2010.01412](https://arxiv.org/abs/2010.01412)
