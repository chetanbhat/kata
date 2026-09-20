"""EXERCISE 7.2 — Attention variations: ALiBi, sliding window, GQA/MQA.

Goals: geometric ALiBi slopes; causal distance bias; window masks;
grouped-query attention (n_kv_heads=1 is MQA) with KV-head repetition.

Run: pytest tests/test_07_sequences.py -k "alibi or window or gqa or grouped"
"""

import torch
import torch.nn as nn


def alibi_slopes(n_heads: int) -> torch.Tensor:
    # TODO: 2^(-8/n * i) for i = 1..n.
    raise NotImplementedError("# TODO: Implement this")


def alibi_bias(seq_len: int, slopes: torch.Tensor) -> torch.Tensor:
    # TODO: (H, T, T) bias -slope*(i-j) for j<=i, -inf above diagonal.
    raise NotImplementedError("# TODO: Implement this")


def sliding_window_mask(seq_len: int, window: int) -> torch.Tensor:
    # TODO: (T, T) bool: j<=i and i-j <= window.
    raise NotImplementedError("# TODO: Implement this")


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = 1):
        super().__init__()
        # TODO: q/kv/o projections (KV heads < Q heads).
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x, causal=True, attn_bias=None):
        # TODO: Split heads, repeat KV heads, attend, merge, project.
        raise NotImplementedError("# TODO: Implement this")
