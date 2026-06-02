# ADR-014: SQ-Format — Mixed-Precision Sparse-Quantized Bank Format

**日期**: 2026-05-15
**状态**: 设计中
**来源**: [SQ-format: A Unified Sparse-Quantized Hardware-friendly Data Format for LLMs](https://arxiv.org/abs/2512.05409) (Huang et al., 2025)
**涉及**: `FormatBase`、`GranularitySpec`、`QuantScheme`、`QuantConfig`、Calibration Pipeline

## 背景

ADR-012 实现了 BANK 粒度 + outlier_ratio 稀疏分离 + 可配置 outlier_format。在 `outlier_format='int8'` 时，outlier 组用 INT8、normal 组用 INT4——这已经实现了 mixed-precision。但 ADR-012 与 SQ-format 论文在以下关键维度存在差异：

1. **权重 mask 选择**：ADR-012 用纯 magnitude top-k；论文用 GPTQ Hessian importance
2. **激活 mask 选择**：ADR-012 用 per-element cross-sample voting；论文用 per-channel A·W product importance
3. **sparsity 固定**：ADR-012 的 outlier_ratio 不提供 per-bank 固定 sparsity 保证
4. **掩码表示**：ADR-012 用显式 bool tensor；论文用隐式 vmask sentinel
5. **存储布局**：论文有紧凑稀疏存储（W_high compact + W_low with vmask）

本文档记录 SQ-format 论文的完整算法定义，作为增量开发的依据。

## SQ-Format 定义

### 统一框架（Equation 1）

```
SQ-format(X) = ([X_quant], [S_quant], [m], h_high, h_low, b, s)
```

| 参数 | 含义 |
|------|------|
| `h_high` | 高精度格式（如 INT8） |
| `h_low` | 低精度格式（如 INT4） |
| `b` | bank size — 每 bank 包含 b 个元素（沿 bank 轴） |
| `s` | sparsity — 低精度元素占比。s=0.5 表示 50% 用 h_low，50% 用 h_high |
| `[m]` | 精度掩码 — 标记每个元素用高/低精度。隐式（vmask）或显式（vector） |

**核心约束**：
- SQ-format 只作用于 matmul 的**一个操作数**（权或激活），另一个用普通 uniform 格式
- **固定 sparsity per bank**：每 bank 内精确 (1−s)·b 个高精度元素，消除硬件负载不均衡
- 是 NVIDIA 2:4 半结构化稀疏的泛化（2:4 sparse = s=0.5, h_low=0, b=4）

### 等价比特数

```
y = (1-s) × h_high + s × h_low
```

| Config | s | y | 含义 |
|--------|---|----|------|
| B-(8/4)-0.5 | 0.5 | 6 bits | W(SQ6)A4 或 W4A(SQ6) |
| B-(8/4)-0.75 | 0.75 | 5 bits | W(SQ5)A4 或 W4A(SQ5) |
| B-(8/4)-0.875 | 0.875 | 4.5 bits | W(SQ4.5)A4 或 W4A(SQ4.5) |

### 命名约定

- `W4A(SQy)`：权重 INT4，激活 SQ-format（等效 y bits）
- `W(SQy)A4`：权重 SQ-format（等效 y bits），激活 INT4
- `B-(h_high/h_low)-s`：完整 SQ-format 参数，B=bank_size

---

## Algorithm 1：SQ-format on Weights

### 输入与输出

```
Input:  W ∈ R^{K×N}, calibration set D, sparsity s, bank_size b, h_high, h_low
Output: W_high [(1-s)K, N], W_low [K, N], S_high [K/b, N], S_low [K/b, N]
```

### Step 1 — SmoothQuant 平滑

```
W', H ← Smooth(W, D)
```

- `W' = diag(S) · W`：将激活 outlier 通过对角缩放迁移到权重侧
- `H`：标定数据上的 Hessian 对角矩阵（GPTQ 框架）

### Step 2 — GPTQ 重要性打分

```
I_{r,i} = (W'_{r,i})² / (H⁻¹_{i,i})²
```

Per-element 重要性度量，来自 Optimal Brain Surgeon 框架（论文原文："synthesizes the weight's own magnitude with the model's sensitivity to its perturbation"）：
- `(W')²`：绝对值大的权重量化扰动影响大
- `1 / (H⁻¹)²`：loss 曲率陡峭方向（Hessian 对角元素小 → H⁻¹ 大）上权重量化敏感度高

### Step 3 — Per-bank 分组量化

```
for each bank w of W' (共 K/b 个 bank):
    m_w ← top-(1-s) by I_w within each column of this bank
          # 论文原文："within each column" of each bank
          # 选 importance 最高的 (1-s)·b 个元素 → True（高精度）
          # 剩余 s·b 个元素 → False（低精度）

    (w'_high, s'_high) ← Quant(w' ⊙ m_w,   h_high)  # 稀疏高精度组
    (w'_low,  s'_low)  ← Quant(w' ⊙ ∼m_w, h_low)   # 稠密低精度组
```

- 每 bank 的**每列**独立选 top-k，k = (1-s) × b（论文："Within each weight bank, we rank the weights based on I and select the top (1 − s) fraction of weights **within each column**"）
- 两组各自量化，各得 per-bank per-column scale
- Quant 是 symmetric uniform quantization（论文使用）

### Step 4 — 紧凑存储

```
W_high:  [(1-s)K, N]  紧凑排列，仅含高精度元素，无空隙
W_low:   [K, N]        保持原始形状，高精度位置填 vmask sentinel
S_high:  [K/b, N]      per-bank per-column scale
S_low:   [K/b, N]      同上
```

### vmask 隐式掩码机制

用低精度格式的最大未用整数值作为 sentinel：
- `h_low = INT2`：合法值 {−1, 0, 1}，`vmask = 2`
- `h_low = UINT4`（无符号）：合法值 {0,...,14}，`vmask = 15`
- 硬件在低精度计算路径检测到 vmask → 从 W_high 取真值 → 通过 MUX 选对应激活行 → 送入高精度 Tensor Core

### 硬件执行模型（Figure 2a）

两条并行计算路径：

```
Path 1 (低精度, Dense):
    W_low [K, N] × A [K, M] → Low Result [N, M]
    低精度 Tensor Core，高吞吐

Path 2 (高精度, Sparse):
    检测 W_low 中的 vmask → MUX 选择对应 A 行
    W_high [(1-s)K, N] × A_selected [(1-s)K, M] → High Result [N, M]
    高精度 Tensor Core，因稀疏度高其时延被 Path 1 隐藏

Final: Y = Low Result + High Result
```

---

## Algorithm 2：SQ-format on Activations（Static Strategy）

### 输入与输出

```
Input:  W ∈ R^{K×N}, calibration set D, sparsity s, bank_size b, h_high, h_low
Output: W_quant, S_quant, m
```

### 关键差异

激活值是运行时动态产生的，不能预先量化。但激活的 outlier 呈现 **per-channel** 结构——某些通道整体幅度大，对这些通道使用高精度、其余用低精度。Mask 是 **per-channel** 的（不是 per-element）。

### Step 1 — SmoothQuant + 收集统计量

```
W', Ā ← Smooth(W, D)
```

`Ā_j`：标定集上第 j 个输入通道的平均激活幅度（对所有 sample 和 token 位置取均值）。

### Step 2 — Per-channel 重要性打分（A·W Product）

```
for each bank of W' [j_start : j_end] and Ā [j_start : j_end]:
    I_j = |Ā_j · Σ_i W'_{j,i}|    ∀j ∈ [j_start, j_end]
```

其中 `W'` 形状为 `[K, N]`（K = input channels, N = output features）：
- `Σ_i W'_{j,i}`：输入通道 j 与所有输出特征的连接权重之和（带符号，非绝对值。正负权重可抵消，反映该通道对输出的净贡献）
- `Ā_j`：通道 j 的平均激活幅度
- 乘积度量该通道对 dot product 的总贡献（外层 `|·|` 对整个乘积取绝对值）

**为什么不能只用 |Ā_j|？** 论文明确发现这会导致 "significant performance degradation"——激活大的通道如果乘以零权重，对结果无影响。必须同时考虑权重侧。

### Step 3 — Per-bank 选 mask + 量化

```
for each bank:
    m_w ← top-(1-s) channels by I_j   # mask 是 per-channel 的，在 bank 内选
    (w', s') ← Quant(w, h_high, h_low) # 量化权重
```

注意：这里 quant 的是**权重**，mask 是 channel-level。权重被量化为两个精度组以配合激活侧的通道分离。

### Step 4 — 权重行重排

```
Reorder rows of W' based on mask m.
将所有高精度通道对应的 W 行聚集到前面，低精度通道对应的行聚集到后面。
```

> 论文 pseudocode 写 "Reorder rows of W′"，正文写 "columns of the weight matrices can be reordered"。取决于 W 的 shape 约定（K=in_features 还是 out_features），实际含义是将输入通道按 mask 分组后使 W 的对应轴聚集。静态 mask 存储开销极小：Llama-3-70B 仅 5.94 MB。

### 推理时

```
A 到来:
    A_high ← A[mask_channels, :]   # 选中的高精度通道
    A_low  ← A[~mask_channels, :]  # 剩余低精度通道
    Result = W_high_part ⊗ A_high  (高精度 matmul)
           + W_low_part  ⊗ A_low   (低精度 matmul)
```

### 动态策略（对比）

不需要标定期 pre-compute mask。推理时对每个激活 tensor 做 per-bank TopK（按 |A| 绝对值），选 top-(1-s) 通道。需要专用流水线单元（Figure 4）来隐藏 TopK 开销。

---

## 与 ADR-012 的关键差异总结

| 维度 | ADR-012 (当前实现) | SQ-format (论文) |
|------|-------------------|-----------------|
| **权重 mask 选择** | `torch.topk(abs(x))` 纯幅度 | GPTQ Hessian: `I = W² / (H⁻¹)²` |
| **激活 mask 选择** | per-element cross-sample magnitude voting | per-channel `|Ā_j · Σ_i W_{j,i}|` |
| **sparsity 固定** | 不保证 per-bank 精确固定 | per-bank 精确固定 `(1-s)·b` 个 |
| **掩码存储** | 显式 bool tensor | 隐式 vmask sentinel（低精度格式） |
| **高精度存储** | 保持原位，mask 标记 | 紧凑稀疏排列 `[(1-s)K, N]` |
| **权重行重排** | 无 | Algorithm 2 Step 4 重排 W 行 |
| **等价比特** | 无此概念 | `y = (1-s)·h_high + s·h_low` |
| **硬件设计** | 纯软件 | 完整硬件方案 + RTL 综合 |

### 语义对照

| 论文概念 | ADR-012 对应 | 差异 |
|---------|-------------|------|
| `h_high` | `outlier_format`（如 'int8'）| 概念相同 |
| `h_low` | `format`（如 'int4'）| 概念相同 |
| `s` (sparsity) | `1 - outlier_ratio` | s = 低精度占比，outlier_ratio = 高精度占比 → s = 1 - outlier_ratio |
| `b` (bank_size) | `bank_size` (= `block_size` in QuantConfig) | 概念相同 |
| vmask | 显式 bool mask | 机制不同 |

---

## 论文实验发现（指导调参）

### Bank Size × Sparsity 关系（Section 5.1）

1. **Weights**：固定 sparsity 下存在最优 bank_size。**sparsity 越大 → 最优 bank_size 越大**（Figure 7）
2. **Activations**：bank_size 趋势不如 weights 明显。静态策略倾向更小的 bank_size（Figure 8）
3. 硬件建议：支持 ≥16× sparsity（s=0.9375）时 bank_size 至少 64

### 精度选择边界（Section 5.2）

1. `h_low = INT2`：仅在 B-(8/2)-0.5 下可用，更低的 h_low 几乎无法维持精度
2. `(8/3)` 与 `(8/4)` 趋势相似，存在 storage-performance tradeoff
3. sparsity 下限由硬件算力比决定：若 W8A8 Tensor Core 算力是 W4A4 的 4×，则 sparsity 至少 0.75（高精度 matmul 才能被低精度 matmul 隐藏时延）

---

## 实现原则

1. **不修改现有 `_quantize_per_bank*` 系列函数**：SQ-format 新增独立的量化路径
2. **新实现放在独立文件**（如 `src/formats/_sq_format.py`），通过 `FormatBase` dispatch 集成
3. **新增 `GranularityMode.SQ_FORMAT` 或通过 `QuantConfig` 参数开关**：设计待确定
4. **重要性分数计算独立封装**：`compute_hessian_importance()`（Algorithm 1）和 `compute_channel_product_importance()`（Algorithm 2）
5. **与 ADR-012 共享 bank 维度的 reshape/reduction 逻辑**，复用 `GranularitySpec.bank_size`
