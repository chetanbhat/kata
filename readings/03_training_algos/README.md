# Module 03 — Advanced Training Algorithms & Optimization

## Where this sits

Modules 01–02 built machines that *can* learn; this module is *how they
learn*: the update rule (optimizers), the step-size policy (schedulers),
and the loop that holds it together (clipping, accumulation, AMP). This is
the highest-leverage module for practitioners — most "model didn't train"
mysteries die here, not in architecture.

## Builds on

- Modules 01–02 (losses, `.grad` flow, layer mechanics).
- Guides §2, Story 4 (loss won't decrease).

## Builds toward

- Module 05 (the full optimizer lineage: why Adam exists, what AdamW fixed).
- Module 09 capstone (this loop becomes the training harness).

## The arc

SGD (1951, Robbins–Monro) → momentum (1964, Polyak) → adaptive methods
(AdaGrad 2011 → RMSprop → Adam 2014) → decoupled decay (AdamW 2017). Each
step relaxed an assumption: constant curvature, constant scale, honest
regularization. Schedulers tell the parallel story for step sizes: decay →
restarts → warmup (the large-batch era).

## Exercises

| Note | Exercise | Question it answers |
|------|----------|---------------------|
| [optimizers.md](optimizers.md) | `exercises/03_training_algos/optimizers.py` | Momentum, Nesterov, Adam's bias correction |
| [schedulers.md](schedulers.md) | `exercises/03_training_algos/schedulers.py` | Restarts and warmup-stable-decay |
| [training_loop.md](training_loop.md) | `exercises/03_training_algos/training_loop.py` | Clipping, accumulation, AMP in one loop |

## Section readings

- Required: Ruder, "An overview of gradient descent optimization
  algorithms" — [arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
- Recommended: Kingma & Ba, "Adam" — [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- Suggested: Micikevicius et al., "Mixed Precision Training" —
  [arxiv.org/abs/1710.03741](https://arxiv.org/abs/1710.03741)
