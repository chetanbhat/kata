"""Decoder-only Transformer block (GPT-style) components.

Includes: scaled dot-product attention with optional masking, multi-head
self-attention with causal masking, RMSNorm, Rotary Positional Embeddings
(RoPE), and a complete DecoderBlock (norm -> masked MHA -> norm -> MLP).
"""

import torch
import torch.nn as nn


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """Q/K/V: (..., T, d). Additive boolean/float mask broadcast over scores."""
    scores = Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask.bool() if mask.dtype == torch.bool else mask == 0,
                                    float("-inf"))
    return torch.softmax(scores, dim=-1) @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        split = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        mask = None
        if causal:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask)
        return self.proj(out.transpose(1, 2).reshape(B, T, -1))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class RotaryEmbedding(nn.Module):
    """RoPE: rotates each consecutive pair of dims by position-dependent angle."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_T, dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., T, dim) -> rotated tensor of same shape."""
        T = x.shape[-2]
        cos, sin = self.cos[:T], self.sin[:T]  # (T, dim/2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        # Broadcast (T, dim/2) against (..., T, dim/2).
        shape = (1,) * (x1.dim() - 2) + cos.shape
        cos, sin = cos.view(shape), sin.view(shape)
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)


class DecoderBlock(nn.Module):
    """Pre-norm GPT block: x + MHA(RMSNorm(x)) then + MLP(RMSNorm(.))"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=True)
        return x + self.mlp(self.norm2(x))
