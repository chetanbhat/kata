# 5.1 — Damped Newton & L-BFGS

**Builds on:** 1.1 (quadratics: Newton solves them in one step), 2.1
(Hessian via nested autograd). **Builds toward:** 5.2 (why first-order won
anyway), research literacy for second-order papers (K-FAC, Shampoo).

## Introduction

Newton's method multiplies the gradient by the *inverse Hessian* — optimal
step for quadratics, one-step convergence in the exercise. Two problems
killed it for deep learning: (1) O(n³) solves at millions of parameters,
(2) indefinite Hessians (saddle directions get *maximized*). Damping
(`H + λI`) fixes (2) locally — Levenberg–Marquardt thinking. L-BFGS fixes
(1) *approximately*: never form H, just keep recent `(s, y)` curvature
pairs and replay them through the two-loop recursion, with Armijo
backtracking guaranteeing sufficient decrease. Empty history = steepest
descent — the algorithm degrades gracefully, a design virtue worth copying.

## The math that matters

- Newton step: `Δ = −(H+λI)⁻¹g`. On `½(w−t)'Q(w−t)`, one step lands within
  damping error of `t` (test asserts `< 1e-4`).
- Two-loop recursion applies an implicit inverse-Hessian approximation in
  O(m·n) for memory m — read it as "steepest descent, bent by experience."
- Curvature condition `y's > 0` guards positive-definiteness; skip the
  update otherwise (non-convex landscapes violate it routinely).
- Armijo: accept step `t` if `f ≤ f₀ + c₁·t·g'd`. Theory in one line.

## Intuition check

L-BFGS on a κ=100 quadratic must crush gradient descent's zigzag — watch
the loss hit `1e-4` in 10 steps where SGD would need hundreds. That gap is
the entire historical argument for curvature, and its irrelevance at batch
scale is the argument against.

## Readings

- Required: Nocedal & Wright, *Numerical Optimization*, Ch. 3 (line
  search), 6–7 (Newton, quasi-Newton) — the canon; library copy.
- Recommended: Martens, "Deep learning via Hessian-free optimization" —
  [arxiv.org/abs/1004.1630](https://arxiv.org/abs/1004.1630)
- Suggested: Martens & Grosse, "Kronecker-factored Approximate Curvature"
  — [arxiv.org/abs/1503.05671](https://arxiv.org/abs/1503.05671)
