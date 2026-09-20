# Module 06 — Regularization & Normalization History

## Where this sits

Training (Module 03) gets loss down; this module decides *which* low-loss
solution you keep. Two families: noise/constraints that prevent memorization
(dropout lineage, smoothing, Mixup) and reparameterizations that reshape the
landscape (norm lineage from BatchNorm to spectral control).

## Builds on

- Module 02 (BatchNorm mechanics — now question its origin story).
- Module 01 (L1/L2 penalties reappear inside AdamW and label smoothing).

## Builds toward

- Module 08 (spectral norm + gradient penalty are GAN stabilization).
- Module 09 (regularization choices move the val curve, not the train one).

## The arc

Dropout (2012) → DropConnect → Stochastic Depth: noise climbing from
activations to weights to whole layers. BatchNorm (2015) worked for reasons
its paper got wrong (landscape smoothing, not covariate shift — Santurkar
2018). GANs forced honest Lipschitz control: clipping (crude) → gradient
penalty → spectral norm.

## Exercises

| Note | Exercise | Question it answers |
|------|----------|---------------------|
| [regularizers.md](regularizers.md) | `exercises/06_regularization/regularizers.py` | Noise levels, smoothing, Mixup |
| [normalization.md](normalization.md) | `exercises/06_regularization/normalization.py` | WeightNorm, SpectralNorm, WGAN-GP |

## Section readings

- Required: Srivastava et al., "Dropout" —
  [arxiv.org/abs/1207.0580](https://arxiv.org/abs/1207.0580)
- Recommended: Santurkar et al., "How Does Batch Normalization Help
  Optimization" — [arxiv.org/abs/1805.11604](https://arxiv.org/abs/1805.11604)
- Suggested: Zhang et al., "mixup" —
  [arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412)
