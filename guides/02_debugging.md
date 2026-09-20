# Debugging guide: read the failure, not the code

Every failure in this gym is one of four stories. Identify the story first.

## Story 1: shape mismatch (S invariant)

Symptom: `matmul` / broadcasting `RuntimeError` with two shapes.
Response: **draw the diagram.** For an MLP layer write, on paper:

```
X: (B, d_in)  @  W.T: (d_in, d_out)  +  b: (d_out,)  ->  (B, d_out)
```

90% of shape bugs are a missing transpose or a bias shaped `(d_out, 1)`
instead of `(d_out,)`. Fix the diagram, then the code. `einops`-style
comments (`# (B, T, H, d)`) above each line prevent recurrence.

## Story 2: NaN / inf loss

Checklist in order:
1. **Log of zero:** any `log(p)` with `p = 0`? Clip probs, use
   log-sum-exp (subtract the max first — guides/01, §3).
2. **lr too high:** loss explodes after a few steps? Divide lr by 10; if
   the explosion point moves later, it was the lr.
3. **Softmax overflow:** raw logits in the hundreds? Normalize inputs
   (StandardScaler on train stats only — Module 09).
4. Bisect: run 3 steps printing loss each step; the step where it jumps
   tells you which op overflowed.

## Story 3: grad is None or all-zero (G invariant)

- `None`: the tensor is detached from the graph. Culprits: `.detach()`,
  `torch.no_grad()` around the forward pass, in-place ops on needed
  tensors, or building the loss from `.data`/numpy conversions.
- All-zero: a dead path. Classic: a weight matrix whose only input is an
  all-zero hidden state (first RNN step from `h = 0`), or ReLU killing
  every activation (dead neurons — check activation stats, lower the lr).
- Forgot `optimizer.zero_grad()`? Then grads *accumulate* — loss falls
  then oscillates. If your loop has no `zero_grad`, add it.

Finite-difference check (the NumPy analogue): perturb one parameter by
`1e-5`; if the loss doesn't move, that parameter is decorative.

## Story 4: loss won't decrease (C invariant)

1. Is the loss connected to the parameters at all? (Story 3 first.)
2. Is the gradient *correct*? Compare against finite differences on a
   batch of 4. Wrong sign = bug in the backward pass.
3. Is the lr sane? Try `lr × 10` and `lr / 10` for 10 steps each; one of
   them should move faster. If neither moves, the bug is in 1–2.
4. Is the task learnable at this scale? Overfit ONE batch first (loss
   should hit ~0). If it can't memorize 4 samples, nothing else matters.

## Reading learning curves (capstone skill)

```
loss
 ^   train ___
 |           \___
 |               \____  <- healthy: both fall, small gap
 |   val    ___
 |             \___  ___
 +--------------------------> epoch
```

- **Gap widens, train keeps falling:** overfitting. More data, dropout /
  weight decay, smaller model, early stopping.
- **Both flat and high:** underfitting or broken optimization — go back to
  Story 4, do NOT regularize.
- **Val rises while train falls:** stop where val was lowest (early
  stopping); the checkpoint *is* the regularizer.
- **Jagged:** batch too small or lr too high; smooth with larger batches
  or lr decay, not with a bigger model.

## The 10-minute rule

If a single test resists for 10 minutes, shrink the problem: batch of 4,
hidden size 2, one step, print everything. Bugs that survive scale rarely
survive shrinking.
