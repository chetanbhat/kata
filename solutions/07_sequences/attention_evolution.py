"""Sequence history II: attention after "Attention Is All You Need".

Variations practitioners must know: absolute sinusoidal positions (2017) ->
relative biases/T5 (2019) -> RoPE (2021, in module 04) -> ALiBi (2022,
zero-shot length extrapolation by penalizing distance); full MHA -> MQA
(2019, one KV head) -> GQA (2023, groups) for KV-cache-bound inference;
sliding-window attention (Mistral 2023) for linear-ish long context.
"""

import torch
import torch.nn as nn


def alibi_slopes(n_heads: int) -> torch.Tensor:
    """Geometric slopes 2^(-8/n * i), i = 1..n (Press et al. 2022)."""
    return torch.pow(2.0, -8.0 / n_heads * torch.arange(1, n_heads + 1).float())


def alibi_bias(seq_len: int, slopes: torch.Tensor) -> torch.Tensor:
    """(H, T, T) bias: -slope*(i-j) for j<=i, -inf above diagonal."""
    pos = torch.arange(seq_len)
    dist = (pos[None, :] - pos[:, None]).clamp_min(0)  # (T, T)
    bias = -slopes[:, None, None] * dist[None]
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return torch.where(causal[None], bias, torch.tensor(float("-inf")))


def sliding_window_mask(seq_len: int, window: int) -> torch.Tensor:
    """(T, T) bool: attend to j<=i with i-j <= window."""
    pos = torch.arange(seq_len)
    return ((pos[:, None] - pos[None, :]).clamp_min(0) <= window) & torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool))


class GroupedQueryAttention(nn.Module):
    """Q heads = n_heads, KV heads = n_kv_heads (1 = MQA)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = 1):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv = n_heads, n_kv_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.d_head).transpose(1, 2)
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)
        if attn_bias is not None:
            scores = scores + attn_bias
        if causal:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, float("-inf"))
        return self.o_proj((torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(B, T, -1))
