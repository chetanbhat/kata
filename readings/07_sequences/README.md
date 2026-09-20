# Module 07 — Sequence Modeling History (RNN → Attention → SSM)

## Where this sits

Module 04 built one LSTM and one Transformer block; this module is the
*forty-year argument* around them: recurrence can't remember (fixed by
gates), attention remembers everything at quadratic cost (fixed by
positions, grouping, windows), and state-space models try to have both.
The full "wrong turns" museum in one hallway.

## Builds on

- Module 04 (LSTM cell, MHA mechanics — now vary them).
- Module 02 (norms/scans reuse the same tensor discipline).

## Builds toward

- Reading modern LLM codebases (GQA + RoPE + sliding windows = Mistral).
- Module 08 (diffusion backbones inherit these sequence choices).

## The arc

Elman (1990) → vanishing analysis (Bengio 1994) → LSTM (1997) → GRU (2014)
→ seq2seq bottleneck → attention (2014–17) → "what does position cost?"
(RoPE/ALiBi) → "what does the KV cache cost?" (MQA/GQA) → "can recurrence
come back?" (S4/Mamba). Each era optimizes what the previous era made
expensive.

## Exercises

| Note | Exercise | Question it answers |
|------|----------|---------------------|
| [rnn_history.md](rnn_history.md) | `exercises/07_sequences/rnn_history.py` | Vanishing, measured; GRU's merged gate |
| [attention_evolution.md](attention_evolution.md) | `exercises/07_sequences/attention_evolution.py` | ALiBi, windows, MQA/GQA |
| [ssm.md](ssm.md) | `exercises/07_sequences/ssm.py` | Discretization + selective scan |

## Section readings

- Required: Olah, "Understanding LSTM Networks" —
  [colah.github.io/posts/2015-08-Understanding-LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- Recommended: Gu et al., "Mamba" —
  [arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)
- Suggested: Lipton et al., "A Critical Review of RNNs" —
  [arxiv.org/abs/1506.00019](https://arxiv.org/abs/1506.00019)
