"""
手撕练习 01 — Scaled Dot-Product / Multi-Head Attention

规则：
1. 不要看 lessons/01_attention.py
2. 只允许查 PyTorch API（matmul / view / transpose / masked_fill）
3. 写完后运行：python tests/01_attention.py
4. 对照 lessons/01_attention.py 自己改错

通过标准：
- shape 全对
- causal attn 上三角 ≈ 0
- 每行 attn 和 ≈ 1
"""

from __future__ import annotations

import math

import torch
from torch._numpy import bool_
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = attn @ v
    return attn, out



def scaled_dot_product_attention(q, k, v, mask=None):
    # TODO: scores = ? / sqrt(d_k)
    # TODO: mask 填 -inf
    # TODO: softmax + 乘 V
    raise NotImplementedError


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    # TODO: 返回 (seq, seq) 的 bool 上三角掩码（True=屏蔽）
    raise NotImplementedError


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # TODO: w_q, w_k, w_v, w_o
        raise NotImplementedError

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: (B, T, C) -> (B, H, T, D)
        raise NotImplementedError

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: (B, H, T, D) -> (B, T, C)
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask=None):
        # TODO
        raise NotImplementedError


def _self_check() -> None:
    torch.manual_seed(0)
    b, t, d_model, n_heads = 2, 5, 32, 4
    x = torch.randn(b, t, d_model)
    mha = MultiHeadAttention(d_model, n_heads)
    mask = causal_mask(t)
    out, attn = mha(x, mask=mask)

    assert out.shape == (b, t, d_model), out.shape
    assert attn.shape == (b, n_heads, t, t), attn.shape
    # 上三角应接近 0
    upper = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    assert attn[0, 0][upper].abs().max() < 1e-5
    # 行和 ≈ 1
    assert torch.allclose(attn[0, 0].sum(-1), torch.ones(t), atol=1e-4)
    print("PASS  Attention 手撕过关")


if __name__ == "__main__":
    seq_len = 5; d_k = 6
    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attn, out = attention(q, k, v, mask)
    print(attn, "\n", out)
    #_self_check()