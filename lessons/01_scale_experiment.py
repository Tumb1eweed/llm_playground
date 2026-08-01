"""
实验：有没有除以 sqrt(d_k)，注意力分布差在哪？

运行：
    python lessons/01_scale_experiment.py
"""

from __future__ import annotations

import math
import unicodedata

import torch
import torch.nn.functional as F


def _disp_width(s: str) -> int:
    """终端显示宽度：中日韩全角计 2，其余计 1。"""
    w = 0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ("F", "W"):
            w += 2
        else:
            w += 1
    return w


def _pad(s: str, width: int, align: str = "<") -> str:
    gap = max(0, width - _disp_width(s))
    if align == ">":
        return " " * gap + s
    if align == "^":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def _hline(widths: list[int], left="+", mid="+", right="+", fill="-") -> str:
    parts = [fill * w for w in widths]
    return left + mid.join(parts) + right


def _row(cells: list[str], widths: list[int], aligns: list[str]) -> str:
    body = "|".join(_pad(c, w, a) for c, w, a in zip(cells, widths, aligns))
    return "|" + body + "|"


def entropy(p: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum())


def compare_once(d_k: int, n_keys: int = 8, seed: int = 0) -> None:
    torch.manual_seed(seed)

    q = torch.randn(d_k)
    K = torch.randn(n_keys, d_k)

    scores_raw = K @ q
    scores_scaled = scores_raw / math.sqrt(d_k)

    attn_raw = F.softmax(scores_raw, dim=-1)
    attn_scaled = F.softmax(scores_scaled, dim=-1)

    # 列宽：指标 | 未缩放 | 除以sqrt(d_k)
    widths = [16, 14, 14]
    aligns = ["<", ">", ">"]

    print()
    print(_hline(widths, "+", "+", "+", "="))
    title = f" d_k={d_k}  n_keys={n_keys}  seed={seed} "
    print("|" + _pad(title, sum(widths) + len(widths) - 1, "^") + "|")
    print(_hline(widths))
    print(_row(["metric", "no scale", "/sqrt(d_k)"], widths, ["<", "^", "^"]))
    print(_hline(widths))
    print(_row(["scores mean", f"{scores_raw.mean():.3f}", f"{scores_scaled.mean():.3f}"], widths, aligns))
    print(_row(["scores std", f"{scores_raw.std():.3f}", f"{scores_scaled.std():.3f}"], widths, aligns))
    print(_row(["attn max", f"{attn_raw.max():.4f}", f"{attn_scaled.max():.4f}"], widths, aligns))
    print(_row(["entropy", f"{entropy(attn_raw):.4f}", f"{entropy(attn_scaled):.4f}"], widths, aligns))
    print(_hline(widths))

    # 注意力权重：按位置对齐
    idx_w = 8
    cell_w = 8
    attn_widths = [idx_w] + [cell_w] * n_keys
    attn_aligns = ["<"] + [">"] * n_keys  # 首列标签，其余数值
    headers = ["kind"] + [str(i) for i in range(n_keys)]

    print(_row(headers, attn_widths, ["^"] * (n_keys + 1)))
    print(_hline(attn_widths))
    print(_row(["raw"] + [f"{x:.4f}" for x in attn_raw.tolist()], attn_widths, attn_aligns))
    print(_row(["scaled"] + [f"{x:.4f}" for x in attn_scaled.tolist()], attn_widths, attn_aligns))
    print(_hline(attn_widths, "+", "+", "+", "="))


def compare_across_dims() -> None:
    widths = [8, 12, 12, 12, 12]
    aligns = [">", ">", ">", ">", ">"]

    print()
    print(_hline(widths, "+", "+", "+", "="))
    title = " sweep d_k: does no-scale collapse to one-hot? "
    print("|" + _pad(title, sum(widths) + len(widths) - 1, "^") + "|")
    print(_hline(widths))
    print(
        _row(
            ["d_k", "raw max", "scaled max", "raw H", "scaled H"],
            widths,
            ["^"] * 5,
        )
    )
    print(_hline(widths))

    for d_k in [4, 8, 16, 32, 64, 128, 256, 512]:
        torch.manual_seed(0)
        q = torch.randn(d_k)
        K = torch.randn(8, d_k)
        raw = F.softmax(K @ q, dim=-1)
        scaled = F.softmax((K @ q) / math.sqrt(d_k), dim=-1)
        print(
            _row(
                [
                    str(d_k),
                    f"{raw.max():.4f}",
                    f"{scaled.max():.4f}",
                    f"{entropy(raw):.4f}",
                    f"{entropy(scaled):.4f}",
                ],
                widths,
                aligns,
            )
        )

    print(_hline(widths, "+", "+", "+", "="))


if __name__ == "__main__":
    compare_once(d_k=8, seed=0)
    compare_once(d_k=64, seed=0)
    compare_once(d_k=256, seed=0)
    compare_across_dims()

    print(
        """
how to read
-----------
- no scale: larger d_k -> larger scores std -> sharper attn (max up, entropy down)
- /sqrt(d_k): keeps scores std ~ O(1), distribution stays soft across d_k
"""
    )
