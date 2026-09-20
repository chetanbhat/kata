# 2.3 — Manual Conv2D via `unfold` + MatMul

**Builds on:** 2.2 (Linear as matmul — convolution is the same idea with
weight sharing), guides/01 §1 (shape bookkeeping). **Builds toward:** 4.1
(ResNet *is* these blocks + skip paths).

## Introduction

Convolution looks exotic but is mechanically simple: slide a window over
the input, dot each patch with the kernel. `unfold` materializes all
patches as columns (`(B, C·kH·kW, L)`), then one matrix multiply applies
every filter to every patch at once — exactly how libraries implement it
under the hood (im2col). Stride = step size, padding = border, dilation =
holes in the kernel (atrous: bigger receptive field, same parameters).
Backward is free: every op used is differentiable, so autograd handles it —
verify against `nn.Conv2d` with copied weights (the test demands exact
equality, not vibes).

## The math that matters

- Output size: `⌊(W + 2P − D(K−1) − 1)/S⌋ + 1` per axis. Derive it by
  counting window placements; the test's `(8,8)→(4,4)` (k=3,s=2,p=1) is
  your check.
- Weight sharing: a dense layer on 8×8×3 would need ~200k weights per
  output map; conv reuses `C·k²` everywhere — translation equivariance as
  parameter efficiency.
- Dilation composes receptive fields exponentially with depth — the
  mechanism behind WaveNet/DeepLab context (no extra code in this exercise,
  but know the formula generalizes).

## Intuition check

With weights copied from your layer, `nn.Conv2d` must agree to `1e-5`.
Off-by-one in the output-size formula shows up as a reshape error, not a
wrong number — read the reshape error as "my arithmetic, not my code."

## Readings

- Required: Goodfellow et al., *Deep Learning*, Ch. 9 (convolution) —
  [deeplearningbook.org](https://www.deeplearningbook.org)
- Recommended: CS231n, "Convolutional Networks" (best visual treatment) —
  [cs231n.github.io/convolutional-networks](https://cs231n.github.io/convolutional-networks/)
- Suggested: Dumoulin & Visin, "A guide to convolution arithmetic" —
  [arxiv.org/abs/1603.07285](https://arxiv.org/abs/1603.07285)
