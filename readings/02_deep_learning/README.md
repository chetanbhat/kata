# Module 02 — Deep Learning Primitives & Mechanics

## Where this sits

Module 01 did calculus by hand; this module builds the *machinery* that
does it for you (autograd) and the *parts* real networks are assembled from
(layers, norms, convolution). After this module you can read any PyTorch
model definition without blinking.

## Builds on

- Module 01 (manual backprop → now mechanize it; activations reused).
- Guides §2 (shape diagrams become a daily habit here).

## Builds toward

- Module 03 (optimizers consume the `.grad`s produced here).
- Module 04 (ResNet/LSTM/Transformer are compositions of these parts).
- Module 06 (normalization history revisits BatchNorm's story).

## The arc

2012: AlexNet shows GPU-trained convnets crush ImageNet — but the real
enabler was software: Theano → TensorFlow → PyTorch made gradients free,
turning architecture design from calculus homework into Lego. The
counter-lesson (Module 05): free gradients made everyone stop thinking
about curvature for a decade.

## Exercises

| Note | Exercise | Question it answers |
|------|----------|---------------------|
| [autograd.md](autograd.md) | `exercises/02_deep_learning/autograd.py` | What `.backward()` actually does |
| [layers.md](layers.md) | `exercises/02_deep_learning/layers.py` | Init, normalization, dropout mechanics |
| [conv2d.md](conv2d.md) | `exercises/02_deep_learning/conv2d.py` | What convolution *is* (patches × kernels) |

## Section readings

- Required: Karpathy, micrograd (a complete autograd engine in ~200 lines)
  — [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
- Recommended: Goodfellow et al., *Deep Learning*, Ch. 6.5 (backprop) and
  Ch. 9 (convnets) — [deeplearningbook.org](https://www.deeplearningbook.org)
- Suggested: PyTorch autograd mechanics —
  [pytorch.org/docs/stable/notes/autograd.html](https://pytorch.org/docs/stable/notes/autograd.html)
