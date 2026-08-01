"""
Lesson 01 — Scaled Dot-Product Attention & Multi-Head Attention
面试手撕 Transformer 的第一关（也是最高频）。

目标（今天必须能默写）：
1. Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
2. 每个张量的 shape 变化
3. 因果掩码（GPT 用）怎么加
4. Multi-Head 如何拆头、拼头

运行：
    python lessons/01_attention.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Scaled Dot-Product Attention（面试必默写）
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q: (..., seq_q, d_k)
        k: (..., seq_k, d_k)
        v: (..., seq_k, d_v)
        mask: 可广播到 (..., seq_q, seq_k)；True/1 表示「屏蔽」该位置
              （常见约定：mask==True 的位置填 -inf）

    Returns:
        out:  (..., seq_q, d_v)
        attn: (..., seq_q, seq_k)  — 注意力权重，面试常被要求返回便于 debug
    """
    d_k = q.size(-1)

    # (..., seq_q, d_k) @ (..., d_k, seq_k) -> (..., seq_q, seq_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 被 mask 的位置不能被 attend，置为 -inf，softmax 后 ≈ 0
        scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """下三角可见：位置 i 只能看 <= i。shape: (seq, seq)，True=屏蔽。"""
    # triu(..., diagonal=1) 得到严格上三角为 True
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


# ---------------------------------------------------------------------------
# 2. Multi-Head Attention（手撕标准答案结构）
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    标准面试写法：
      1) 线性投影得到 Q/K/V
      2) 拆成 n_heads
      3) 做 scaled_dot_product_attention
      4) 拼回头，再做输出投影

    输入 x: (batch, seq, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 面试里常见三种写法：
        #   A) 三个独立 Linear（下面这种，最清晰）
        #   B) 一个大 Linear 一次投影再 chunk
        #   C) 合并 QKV 成 (3, n_heads, d_head) —— 工程里更常见
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, n_heads, T, d_head)
        b, t, _ = x.shape
        x = x.view(b, t, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n_heads, T, d_head) -> (B, T, C)
        b, _, t, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(b, t, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        self-attn: kv=None，Q/K/V 都来自 x
        cross-attn: kv=encoder_out，Q 来自 x，K/V 来自 kv
        """
        kv = x if kv is None else kv

        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(kv))
        v = self._split_heads(self.w_v(kv))

        out, attn = scaled_dot_product_attention(q, k, v, mask=mask)
        out = self._merge_heads(out)
        out = self.w_o(self.dropout(out))
        return out, attn


# ---------------------------------------------------------------------------
# 3. 自测：shape + 因果掩码直觉
# ---------------------------------------------------------------------------
def _demo() -> None:
    torch.manual_seed(0)
    b, t, d_model, n_heads = 2, 5, 32, 4
    x = torch.randn(b, t, d_model)

    mha = MultiHeadAttention(d_model, n_heads)
    mask = causal_mask(t)

    out, attn = mha(x, mask=mask)

    print("=== Shape Check ===")
    print(f"x:    {tuple(x.shape)}")          # (2, 5, 32)
    print(f"out:  {tuple(out.shape)}")        # (2, 5, 32)
    print(f"attn: {tuple(attn.shape)}")       # (2, 4, 5, 5)  — 含 head 维

    print("\n=== Causal mask (True=blocked) ===")
    print(mask.to(torch.int))

    # 取 batch0, head0 的注意力：上三角应接近 0
    print("\n=== attn[0,0] (应近似下三角) ===")
    print(attn[0, 0].detach().numpy().round(3))

    # 行和 ≈ 1
    row_sum = attn[0, 0].sum(dim=-1)
    print("\n=== row sums (应≈1) ===")
    print(row_sum.detach().numpy().round(4))


if __name__ == "__main__":
    _demo()
