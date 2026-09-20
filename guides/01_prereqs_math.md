# Math prerequisites, worked by hand (45 min read)

You need surprisingly little math, but you need it *fluently*. Each section
ends with a one-line check you should be able to do on paper.

## 1. Vectors, matrices, broadcasting

`X @ w` with `X: (n, d)`, `w: (d,)` gives `(n,)` — each row dotted with `w`.
Broadcasting aligns trailing dims: `(n, 1) + (n, d)` works; `(n,) + (n, d)`
works (treated as `(1, d)`… no — as `(d,)` broadcast across rows). Rule of
thumb: **write the shapes first, compute second.**

Check: `A: (4, 3)`, `B: (3,)` → `A @ B` is `(4,)`. `A + B` broadcasts `B`
across the 4 rows.

## 2. The only calculus you need: the chain rule

If `L = f(g(x))`, then `dL/dx = f'(g) · g'(x)`. Backprop is this applied
mechanically, layer by layer, reusing intermediate values (the "caches").

Worked example — one neuron, MSE loss, single sample:
`z = w·x + b = 2·3 + 1 = 7`, `ŷ = ReLU(7) = 7`, `L = (7 − 4)² = 9`, target 4.
- `dL/dŷ = 2(7−4) = 6`
- `dŷ/dz = 1` (ReLU, z > 0)
- `dz/dw = x = 3`, `dz/db = 1`
- So `dw = 6·1·3 = 18`, `db = 6`. Gradient descent: `w ← 2 − lr·18`.

Check: if the target were 7 instead, every gradient above is exactly 0 —
*loss zero ⇒ no learning signal*, the base case of all debugging.

## 3. Softmax + cross-entropy, by hand

Logits `[2.0, 1.0, 0.1]`, true class 0. Subtract the max (the log-sum-exp
trick — do this *before* exponentiating or `e^1000` overflows):
`[0, −1, −1.9]` → exps `[1, 0.368, 0.150]` → sum `1.518` →
probs `[0.659, 0.242, 0.099]`. Loss `= −log(0.659) ≈ 0.417`.

The beautiful gradient: `dL/dlogits = probs − one_hot(target)`.
Here `[-0.341, 0.242, 0.099]` — note it sums to 0 (raising one logit must
lower the others; softmax outputs compete).

Check: with logits `[0, 0, 0]` and class 1, loss `= log(3) ≈ 1.099` and the
gradient is `[-1/3, 2/3, -1/3]`… verify: probs are uniform `1/3`,
minus one-hot `[0,1,0]` gives `[1/3, −2/3, 1/3]`? Sign check: loss
*decreases* when the true logit rises, so `dL/dlogit_true < 0`:
`[1/3, −2/3, 1/3]`. Always sanity-check the sign.

## 4. Bias–variance in one paragraph

Low train error + high val error = **overfit** (memorizing; add data,
regularize, shrink the model). High train error = **underfit** (model too
weak or optimization broken — fix the model/lr before regularizing).
Regularizing an underfit model is the classic beginner trap.

## 5. Probability facts used in Modules 5–8

- Bayes: `p(z|x) ∝ p(x|z)p(z)` — VAEs approximate the intractable left side.
- KL `N(μ,σ)‖N(0,I) = −½Σ(1 + logσ² − μ² − σ²)` — closed form, Module 8.
- Expectation linearity: `E[aX + b] = aE[X] + b` — Mixup, SWA averaging.
- `Var(mean of n i.i.d.) = σ²/n` — why bigger batches = cleaner grads.
