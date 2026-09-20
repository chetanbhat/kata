# Research Log — wrong turns, fixes, and open questions

## Meta — our own harness wrong turn (fixed)

- The first test suite imported `solutions.*`, so an untouched checkout
  reported all-green and learner code was never evaluated. Tests must run
  against the learner's implementation by default (`KATA_IMPL=exercises`),
  with the reference behind an explicit flag. A gym that can't see you
  can't coach you. Pinned by `tests/test_impl_selector.py`.

Cumulative journal for the advanced track (Modules 5–8). Each entry records
what was tried historically, why it failed or sufficed, and what replaced
it. Add dated entries as you work through the exercises.

## 05 — Optimization

- **Full Newton doesn't scale.** Exact Hessians cost O(n³) to invert and are
  indefinite away from minima. Damping (`H + λI`) rescues the step locally;
  still only viable for small problems — the kata keeps it for last-layer
  tuning and scientific models, not foundation training.
- **AdaGrad's fatal accumulation.** Summing *all* past squared grads drives
  the effective lr to zero on long runs. RMSprop's exponential moving
  average is the one-line fix; verify by comparing lr trajectories.
- **Adam's coupled weight decay.** `wd * p` passed through the adaptive
  denominator is *not* L2 regularization. AdamW decouples it; the kata's
  exact test (zero grad → `p *= 1 - lr·wd`) pins the difference.
- **Lion: less can match.** A sign-based update with two momentum buffers
  rivals AdamW — a reminder that the optimizer design space is not closed.
- **Flat minima generalize.** SAM pays 2× grads to minimize the worst case
  in a ρ-ball; SWA averages iterates into flat basins for free. The ascent
  check (`first_step` must *raise* loss) keeps the implementation honest.

## 06 — Regularization & normalization

- **BatchNorm's origin myth.** The "internal covariate shift" story did not
  survive (Santurkar et al. 2018: BN smooths the landscape). Lesson: keep
  the technique, distrust the first explanation.
- **Dropout → DropConnect → Stochastic Depth.** Same idea climbing levels:
  mask activations, mask weights, drop whole residual branches. Each step
  trades finer noise for coarser structural regularization.
- **WGAN clipping underused capacity.** Forcing weights into a box cripples
  the critic; the gradient penalty enforces 1-Lipschitz where it matters
  (interpolated samples). Spectral norm does it architecturally instead.
- **Label smoothing & Mixup break dogmas.** Penalizing confidence and
  training off-manifold both "shouldn't" work — both do. Calibration and
  robustness come from softening the target, not hardening the model.

## 07 — Sequences

- **Elman RNNs can't carry memory.** Backprop multiplies Jacobians; with
  spectral radius < 1 the gradient dies geometrically (Bengio 1994). The
  kata's `grad_norm_vs_time` probe makes this a measurement, not folklore.
- **Gates, then attention, then selection.** LSTM's additive cell highway →
  GRU's merged gate → attention sidestepping recurrence entirely → SSMs
  returning with input-dependent selection (Mamba). Each step re-asks: what
  must be remembered, and what may be forgotten?
- **Positions: absolute → relative → none + bias.** Sinusoids → RoPE/ALiBi
  (extrapolation by construction, not by hope). ALiBi's linear penalty is
  almost embarrassingly simple — and extrapolates.
- **KV-cache economics drive MQA/GQA.** Inference is memory-bound on the
  cache, so KV heads shrink (8→1) while Q heads stay. Architecture follows
  the bottleneck, not the benchmark.

## 08 — Generative models

- **VAE posterior collapse.** A strong decoder + full KL weight → `KL → 0`,
  `z` ignored (Bowman et al. 2016). The `beta` knob is the annealing
  handle; `beta=0 ⇒ loss == recon` is the kata's pinned invariant.
- **GAN minimax saturates.** `log(1−D)` is flat when D confidently rejects —
  the gradient test (saturated vs non-saturating at logit −10) exhibits the
  exact pathology Goodfellow et al. patched in the same paper.
- **Distances matter.** JS divergence → Wasserstein (meaningful curves
  under non-overlapping supports) → penalties over constraints. Mode
  collapse is the price of the original game, not a bug in your code.
- **Diffusion's chain was optional.** DDPM's Markov forward + learned
  reverse → DDIM's deterministic strided sampler → guidance trading
  conditional against unconditional. The cosine schedule fixes DDPM
  destroying low-res information too fast — schedules are hyperparameters
  with physical meaning.

## Open questions (for future modules)

- Scaling laws (Chinchilla-optimal token/param tradeoffs) as a fitting exercise.
- µP (maximal update parametrization) for hyperparameter transfer.
- RL foundations: REINFORCE → PPO clipping → DQN target networks.
- Distributed primitives: all-reduce simulation, ZeRO sharding, activation
  checkpointing cost models.
