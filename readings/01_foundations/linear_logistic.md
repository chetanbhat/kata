# 1.1 — Linear & Logistic Regression (NumPy, zero loops)

**Builds on:** guides/01 §1 (matmul shapes). **Builds toward:** 1.2 (these
units stack into MLPs), 3.1 (this GD loop generalizes to all optimizers).

## Introduction

Two answers to "fit parameters to data." Least squares has a closed form:
set the gradient to zero and solve the normal equations
`w = (X'X)⁻¹X'y` (plus `λI` for ridge — the bias column stays unpenalized,
a detail beginners get wrong). Logistic regression has no closed form, so
you iterate: full-batch gradient descent on the Bernoulli negative
log-likelihood, i.e. cross-entropy with a sigmoid. Same data flow, two
optimization regimes — the whole field is variations on this split.

## The math that matters

- Ridge: `(X'X + λI)⁻¹X'y`. `λ` shrinks weights toward zero (see the
  weight-recovery test: small noise, near-exact recovery).
- Sigmoid must be branched (`z ≥ 0` vs `z < 0`) or `exp(1000)` overflows —
  your first encounter with *numerical analysis as a survival skill*.
- L1 adds `sign(w)` subgradients (sparsity); L2 adds `2λw` (shrinkage).
  Different penalties, different solutions, one loop.

## Intuition check

If classes are Gaussian blobs (the test data), the logistic boundary is
the hyperplane where `w·x + b = 0` — draw it. If accuracy stalls below
~95% there, your gradient (not the model) is wrong.

## Readings (digested here; originals linked)

- Required: ESL §3.1–3.2 (linear methods) and §4.4 (logistic) —
  [web.stanford.edu/~hastie/ElemStatLearn](https://web.stanford.edu/~hastie/ElemStatLearn/)
- Recommended: NumPy `linalg.solve` vs `inv` docs (never invert explicitly)
  — [numpy.org/doc/stable/reference/routines.linalg.html](https://numpy.org/doc/stable/reference/routines.linalg.html)
- Suggested: Hoerl & Kennard (1970), "Ridge regression" (the original
  shrinkage argument; any library copy).
