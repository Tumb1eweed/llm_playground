# 实验笔记：为什么要除以 √d_k

对应代码：`lessons/01_scale_experiment.py`

```bash
python lessons/01_scale_experiment.py
```

## 1. 问题

Scaled Dot-Product Attention 的公式是：

```text
Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V
```

已经有了 softmax，为什么还要除以 `√d_k`？为什么不是除以 `d_k`？

## 2. 实验设定

- 构造 1 个 query：`q ~ N(0,1)^{d_k}`
- 构造 8 个 key：`K ~ N(0,1)^{8 × d_k}`
- 对比两组分数：
  - **no scale**：`scores = K @ q`
  - **/√d_k**：`scores = (K @ q) / sqrt(d_k)`
- 再各自做 softmax，观察：
  - scores 的均值 / 标准差
  - 注意力最大权重（max）
  - 熵（entropy，越大越“软”）

## 3. 关键结果（seed=0）

### 3.1 d_k = 8（小维度，差别不大）

| metric | no scale | /√d_k |
|--------|----------|-------|
| scores std | 2.483 | 0.878 |
| attn max | 0.5317 | 0.2846 |
| entropy | 1.2076 | 1.8281 |

### 3.2 d_k = 64（常见 d_head）

| metric | no scale | /√d_k |
|--------|----------|-------|
| scores std | 6.481 | 0.810 |
| attn max | 0.8120 | 0.2728 |
| entropy | 0.5602 | 1.8572 |

### 3.3 d_k = 256（不缩放几乎 one-hot）

| metric | no scale | /√d_k |
|--------|----------|-------|
| scores std | 19.595 | 1.225 |
| attn max | **0.9996** | 0.2984 |
| entropy | **0.0037** | 1.7745 |

未缩放时 attn 近似：`[0, 0, 0.9996, 0, 0, 0, 0, 0]`  
缩放后仍是软分布，最大约 0.30。

### 3.4 扫不同 d_k

| d_k | raw max | scaled max | raw H | scaled H |
|-----|---------|------------|-------|----------|
| 4 | 0.4341 | 0.3136 | 1.1830 | 1.7071 |
| 8 | 0.5317 | 0.2846 | 1.2076 | 1.8281 |
| 16 | 0.9866 | 0.4811 | 0.0840 | 1.6083 |
| 32 | 0.9985 | 0.4907 | 0.0125 | 1.6037 |
| 64 | 0.8120 | 0.2728 | 0.5602 | 1.8572 |
| 128 | 0.9563 | 0.3607 | 0.1862 | 1.5538 |
| 256 | 0.9996 | 0.2984 | 0.0037 | 1.7745 |
| 512 | 1.0000 | 0.4096 | 0.0000 | 1.7288 |

趋势：

- **不缩放**：`d_k` 越大 → scores std 越大 → attn 越尖（max↑，熵↓）
- **除以 √d_k**：不同 `d_k` 下分布都更稳定

## 4. 结论

### 4.1 为什么要缩放

链路：

```text
不除 √d_k
  → 点积 scores 标准差随 d_k 变大
  → softmax 容易饱和成 one-hot
  → 注意力过“硬”，梯度变差
```

softmax 只负责归一化成概率，**不会**把过大的分数先拉回合理尺度。

### 4.2 为什么是 √d_k，不是 d_k

假设 `q, k` 各维独立、均值 0、方差 1：

```text
Var(q · k) = d_k
Std(q · k) = sqrt(d_k)
```

想让缩放后方差 ≈ 1：

```text
Var( (q · k) / s ) = d_k / s^2 = 1
  =>  s = sqrt(d_k)
```

如果除以 `d_k`：

```text
Var( (q · k) / d_k ) = 1 / d_k
```

分数会过小 → softmax 过于均匀，是另一个极端。

| 缩放方式 | scores 方差量级 | 倾向 |
|----------|-----------------|------|
| 不除 | `d_k`（太大） | one-hot |
| 除以 `√d_k` | ≈ 1 | 合适 |
| 除以 `d_k` | `1/d_k`（太小） | 过于均匀 |

### 4.3 面试一句话

> Softmax 管归一化；`√d_k` 管控制点积尺度。点积方差约等于 `d_k`，所以除以标准差 `√d_k`，避免维度升高后 softmax 饱和。

## 5. 代码里 shape 提醒

实验是玩具设定，不是完整 batch Attention：

```text
q: (d_k,)           # 1 个 query
K: (n_keys, d_k)    # n_keys 个 key
K @ q -> (n_keys,)  # n_keys 个点积分数
```

正式 Attention 里通常是：

```text
Q: (B, H, Tq, D)
K: (B, H, Tk, D)
scores = Q @ K^T / sqrt(D)  -> (B, H, Tq, Tk)
```
