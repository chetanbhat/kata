# Module 04 — Architectures & Modern Blocks

## Where this sits

Parts (Module 02) + training (Module 03) → complete organisms. Three
lineages in one module: depth for vision (residual paths), memory for
sequences (gating), and content-based routing (attention). The Transformer
block at the end is the atom of everything built after 2017.

## Builds on

- Module 02 (conv, norms, linear layers compose directly into these blocks).
- Module 01 (gates are just sigmoids; attention scores are softmax).

## Builds toward

- Module 07 (attention variations: ALiBi, GQA, sliding windows; RNN→SSM).
- Module 08 (diffusion models are trained Transformer/UNet backbones).

## The arc

Depth stalled at ~20 layers (vanishing gradients) until Highway (2015) and
ResNet (2015) made depth a *choice*; sequences went Elman → LSTM (1997) →
seq2seq → "throw away recurrence" (Transformer, 2017). Common thread:
*architectural bypasses* (skip connections, cell highways, attention) beat
fighting optimization head-on.

## Exercises

| Note | Exercise | Question it answers |
|------|----------|---------------------|
| [resnet.md](resnet.md) | `exercises/04_architectures/resnet.py` | Why adding identity paths unlocks depth |
| [lstm.md](lstm.md) | `exercises/04_architectures/lstm.py` | How gates protect a gradient highway |
| [transformer.md](transformer.md) | `exercises/04_architectures/transformer.py` | Attention, RoPE, and the GPT block |

## Section readings

- Required: Vaswani et al., "Attention Is All You Need" —
  [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Recommended: He et al., "Deep Residual Learning" —
  [arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- Suggested: Alammar, "The Illustrated Transformer" —
  [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
