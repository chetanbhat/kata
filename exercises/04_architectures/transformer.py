"""EXERCISE 4.3 — Attention & decoder-only Transformer block.

Goals: scaled dot-product attention with masking; causal multi-head
self-attention; RMSNorm; RoPE rotation; full pre-norm DecoderBlock.

Run: pytest tests/test_04_architectures.py -k "attention or rope or rmsnorm or decoder"
Reading: readings/04_architectures/transformer.md
"""

import torch
import torch.nn as nn


def scaled_dot_product_attention(Q, K, V, mask=None) -> torch.Tensor:
    # TODO: scores/sqrt(d) + mask -> softmax -> @V.
    raise NotImplementedError("# TODO: Implement this")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO: Fused qkv proj + output proj.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        # TODO: Split heads, attend (causal mask), merge, project.
        raise NotImplementedError("# TODO: Implement this")


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x / rms(x) * weight.
        raise NotImplementedError("# TODO: Implement this")


class RotaryEmbedding(nn.Module):
    """RoPE: rotate each consecutive dim pair by a position-dependent angle."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        # TODO: Precompute cos/sin buffers of shape (max_T, dim/2).
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Apply pairwise rotation; preserve shape.
        raise NotImplementedError("# TODO: Implement this")


class DecoderBlock(nn.Module):
    """Pre-norm GPT block: x + MHA(RMSNorm(x)), then + MLP(RMSNorm(.))"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this.
        raise NotImplementedError("# TODO: Implement this")
