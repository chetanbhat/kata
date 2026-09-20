# Module 01 — Foundations & Classical ML

## Where this sits

Everything above this module is an application of it. Regression is the
simplest learning machine (fit parameters to data); classification is
regression pushed through a squashing function plus a probabilistic loss;
perceptrons and MLPs stack those units into universal approximators; loss
functions decide what "wrong" means. If Modules 4–8 are the city, this is
the bedrock and the plumbing.

## Builds on

- `guides/01_prereqs_math.md` (matmul shapes, chain rule, softmax by hand).
- NumPy fluency: broadcasting, `einsum`-level comfort with axes.

## Builds toward

- Module 02 (autograd mechanizes the manual gradients you write here).
- Module 03 (optimizers generalize the GD loop in `LogisticRegression.fit`).
- Module 08 (cross-entropy returns as the VAE/GAN/diffusion workhorse).

## The arc

1958: the perceptron learns — then Minsky & Papert (1969) show a single
layer can't do XOR, freezing the field. 1986: backprop (Rumelhart, Hinton
& Williams) trains multi-layer nets and thaws it. The lesson: *representational
capacity means nothing without a training procedure* — a theme that recurs
in Modules 05, 07, and 08.

## Exercises

| Note | Exercise | Question it answers |
|------|----------|---------------------|
| [linear_logistic.md](linear_logistic.md) | `exercises/01_foundations/linear_logistic.py` | Closed form vs iteration; why probabilities need squashing |
| [perceptron_mlp.md](perceptron_mlp.md) | `exercises/01_foundations/perceptron_mlp.py` | What depth buys; how gradients flow by hand |
| [losses.md](losses.md) | `exercises/01_foundations/losses.py` | What "wrong" means; why naive code overflows |

## Section readings

- Required: Hastie, Tibshirani & Friedman, *Elements of Statistical
  Learning*, Ch. 3–4 — [web.stanford.edu/~hastie/ElemStatLearn](https://web.stanford.edu/~hastie/ElemStatLearn/)
- Recommended: Goodfellow, Bengio & Courville, *Deep Learning*, Ch. 5–6 —
  [deeplearningbook.org](https://www.deeplearningbook.org)
- Suggested: Karpathy, "Neural Networks: Zero to Hero" (builds the same
  objects from nothing) — [github.com/karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
