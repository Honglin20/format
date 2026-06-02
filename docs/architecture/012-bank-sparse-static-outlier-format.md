# ADR-012: BANK 粒度 + Sparse 静态量化 + 可配置 Outlier Format

**日期**: 2026-05-14
**状态**: 已实施 ✅ (2026-05-15)
**涉及**: `GranularitySpec`、`FormatBase`、`QuantScheme`、`QuantConfig`、Calibration Pipeline

## 背景

当前量化系统有三轴（format × granularity × transform）+ sparse（outlier_ratio）。存在四个待补能力缺口：

1. **BANK 粒度缺失**：无按行切分的粗粒度 scale 分组模式
2. **Sparse 不支持静态量化**：`base.py:137-140` 直接 `raise NotImplementedError`
3. **Sparse 两组用相同格式**：无法为 outlier 组配置不同格式（如 outlier=int8, normal=int4）
4. **BANK 不支持 sparse**：新粒度需要原生支持 outlier 分离

## 决策

### 1. BANK 为独立 GranularityMode

BANK 不等于 PER_BLOCK(axis=0)。核心语义差异：

| | PER_BLOCK | BANK |
|---|---|---|
| 切分方式 | 沿 block_axis tile 成 blocks | 沿 bank_axis 切 contiguous banks |
| 细分程度 | M×N → M×(N/32) 个 block（跨 M 维细分） | M×N → N/16 个 bank（M 维不细分，整个 bank 共享一个 amax） |
| Scale 类型 | MX shared exponent（强制 POT） | 普通 amax（支持 fp32/pot） |
| Scale 数量 | M × (N/block_size) 个 | N/bank_size 个（与 M 无关） |

**BANK 的 reduction 语义**：每个 bank 覆盖该轴段内所有元素，不做跨 bank 维度的二次细分。

```
M×N tensor, bank_axis=-1, bank_size=16:
  reshape → (M, N/16, 16)
  amax = max(abs(x), dim=(0, -1))  → shape (N/16,)  ← 跨 M 维 + bank 内部做 reduction
  结果: N/16 个 scale，每个对应 M×16 个元素
```

**bank_axis 与激活值 batch 安全**：`bank_axis` 复用 `w_axis`/`a_axis` 的值（`_resolve_granularity` 中同步），激活值默认 `a_axis=-1`（特征维），不会切到 batch dim 0。

### 2. Sparse 静态量化：QSNR 驱动的 per-sample mask + 跨 sample 投票

动态 sparse 用 `torch.topk` 构造 mask——需要排序，推理时不可行。

静态 sparse 的 mask 在**校准期确定**，推理时直接复用，不做排序或 threshold 比较。

**核心流程**：

**Step 1 — Per-sample mask 构造（校准期）**：

对每个校准 sample 的 tensor `x_s`，在 granularity group 内部取 top-k magnitude 元素：

```python
# 每个 group 内部独立取 top-k
# mask_s 形状 = x_s.shape（per-element mask，与 granularity 无关）
k = outlier_ratio * group_size
_, top_indices = torch.topk(abs(x_group).flatten(), k)
mask_s = scatter_(top_indices)  # bool tensor, same shape as x_s
```

**Step 2 — 跨 sample 投票（校准期）**：

```python
# 所有 sample 的 per-element mask 求平均
mask_avg = mean(mask_1, mask_2, ..., mask_S)  # shape = tensor shape, values ∈ [0, 1]

# 取平均分最高的 k 个位置作为最终 mask
# k = outlier_ratio * total_elements（与 per-sample 的 group 内 k 一致）
final_mask = topk(mask_avg.flatten(), k)  # 固定，推理时直接使用
```

含义：某个元素位置如果在大多数 sample 中 magnitude 都突出，平均分就高，最终被选入 outlier 组。这是**位置级 voting**，直接优化平均效果。

**推理期**：
- 读取预存的 `final_mask`（固定 bool tensor）
- 读取预存的 `_input_scale_o`、`_input_scale_n`（两组 amax，校准期从 final_mask 计算并存储）
- 两组各自用对应 scale + mask 做量化

**Scale shapes（静态）**：

| Mode | amax_o shape | amax_n shape |
|------|-------------|-------------|
| PER_TENSOR | `()` | `()` |
| PER_CHANNEL | `(C, 1, ...)` | `(C, 1, ...)` |
| PER_BLOCK | `(..., B, 1)` | `(..., B, 1)` |
| BANK | `(B,)` | `(B,)` |

> mask 形状始终与 tensor 相同（per-element），不由 granularity 决定。

**mask 构造独立函数封装**：

`compute_sparse_mask()` 必须封装为独立函数，与 FormatBase 量化 dispatch 解耦。要求：

- **单一职责**：输入 calibration data + format + granularity + outlier_ratio → 输出固定 mask
- **逻辑清晰**：per-sample group 内 top-k → cross-sample voting → final mask，每一步可独立验证
- **可替换**：函数签名稳定，未来替换 QSNR 优化策略或 voting 方式时，调用方代码不变
- **位置**：`src/quantize/_sparse_mask.py` 或同等独立模块

```python
def compute_sparse_mask(
    x_calib: torch.Tensor,           # calibration data, shape (S, ...)
    fmt: FormatBase,
    granularity: GranularitySpec,
    outlier_ratio: float,
) -> torch.Tensor:
    """
    Compute a fixed sparse mask from calibration data via per-sample
    top-k + cross-sample voting.

    The returned mask has the same shape as a single sample tensor (not
    including the batch/S dimension). It is intended to be stored and
    reused at inference time without any per-sample computation.

    Args:
        x_calib: Calibration samples stacked along dim 0 (S, D1, D2, ...).
        fmt: Target format (used for any format-specific considerations).
        granularity: Granularity spec determining group boundaries for
                     per-sample top-k.
        outlier_ratio: Fraction of elements to mark as outliers.

    Returns:
        Boolean mask with shape (D1, D2, ...), True for outlier positions.
    """
```

### 3. outlier_format 放在 QuantScheme

`QuantScheme` 新增 `outlier_format: Optional[FormatBase] = None`：
- `None`：outlier 组用主格式 `self.format`（默认行为，完全向后兼容）
- 非 None：outlier 组用 `outlier_format`，normal 组继续用 `self.format`

`QuantConfig` 新增：
- `outlier_format: Optional[str] = None`：weight 和 activation 的 outlier 格式
- `a_outlier_format: Optional[str] = None`：activation 独立覆盖（None = 跟随 `outlier_format`）

沿用 `a_format` 的设计模式：一个主参数 + 一个 activation override。

### 4. BANK 纳入所有 granularity + sparse 组合

BANK 与 PER_TENSOR/PER_CHANNEL/PER_BLOCK 平权：均支持 `outlier_ratio > 0`，均支持静态量化，均支持 `outlier_format`。

## 备选方案（已拒绝）

**BANK = PER_BLOCK(axis=0)**：BANK 与 PER_BLOCK 的 scale 类型和 reduction 语义不同（BANK 不做跨 bank 轴的二次细分）。合并为一个 mode 会污染语义，且 PER_BLOCK 的 MX shared exponent 强制 POT 不适用于 BANK。

## 实现阶段

| Phase | 内容 | 优先级 | 状态 |
|-------|------|--------|------|
| P1 | BANK 粒度（无 sparse）| P1 | ✅ 已实施 |
| P2 | `compute_sparse_mask()` 独立函数 + 跨 sample 投票 | P1 | ✅ 已实施 |
| P3 | 各 granularity mode 接入静态 mask 路径（per_tensor/per_channel/per_block/bank）| P1 | ✅ 已实施 |
| P4 | 可配置 outlier_format | P2 | ✅ 已实施 |

> P3 备注: PER_BLOCK 静态 sparse 当前 raise NotImplementedError — 动态路径通过 _quantize_outlier_bank 正常工作。
> 实施 commit: 40494bc → 14d8a54 (9 commits, 详见 CHANGELOG)

## 开发优先级分析

推荐顺序：**BANK → compute_sparse_mask → static(all modes) → outlier_format**

理由：
- BANK（~100行）是最自包含的改动，先做可以验证 GranularitySpec 扩展
- `compute_sparse_mask()` 独立封装后，所有 mode 的静态路径共享同一套 mask 逻辑
- 各 mode 接入 static 时改动最小——只需调用 `compute_sparse_mask()` 存 mask，推理时复用
- outlier_format 在所有 sparse 路径稳定后做，改动最小
