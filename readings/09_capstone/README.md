# Module 09 — Capstone: End-to-End Practice

## Where this sits

This is where the gym becomes the sport. Modules 01–08 taught isolated
skills; the capstone chains them: data → splits → preprocessing discipline
→ training with validation → diagnosis → checkpointing → one honest test
evaluation. Nobody hires "implements cross-entropy"; everybody needs
"ships a model whose val curve you can defend."

## Builds on

- Modules 01–03 (losses, MLP mechanics, the training loop pattern).
- Guides §2, Story 4 + curve reading (used here for real).
- Guides/01 §4 (bias–variance stops being abstract).

## Builds toward

- Independent practice: swap moons for a real dataset (sklearn digits,
  then CIFAR via torchvision) using the identical harness.
- Experiment discipline: change one thing, watch val, write it down
  (start your own RESEARCH_LOG entries here).

## The arc

There is no history here — there is *craft*: leakage discipline (fit on
train only), early stopping as regularization, test-set continence
(evaluate once). These norms come from hard collective experience, best
codified in Karpathy's training recipe (see readings).

## Exercises

| Note | Exercise | Question it answers |
|------|----------|---------------------|
| [end_to_end.md](end_to_end.md) | `exercises/09_capstone/end_to_end.py` | What DL work actually feels like |

## Section readings

- Required: Karpathy, "A Recipe for Training Neural Networks" —
  [karpathy.github.io/2019/04/25/recipe](https://karpathy.github.io/2019/04/25/recipe/)
- Recommended: Hastie et al., *Elements of Statistical Learning*, Ch. 7
  (model assessment & selection) —
  [web.stanford.edu/~hastie/ElemStatLearn](https://web.stanford.edu/~hastie/ElemStatLearn/)
- Suggested: Prechelt, "Early Stopping — But When?" —
  [arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
