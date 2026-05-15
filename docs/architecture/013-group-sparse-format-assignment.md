# ADR-013: 按粒度对齐的分组格式分配（Group Sparse）

**日期**: 2026-05-15
**状态**: 已实现 (P1-P6)
**涉及**: `QuantScheme`、`FormatBase`、`QuantConfig`、Calibration Pipeline、`compute_group_mask()`

## 背景

ADR-012 实现了 per-element 的 sparse 量化：每个元素独立决定自己是 outlier(H) 还是 normal(L)，mask 形状始终与 tensor 相同。Granularity 只影响 scale 的 reduction 范围和 per-sample top-k 的 group 边界——但 mask 本身始终是 per-element 的。

在实际部署中存在另一种需求：**格式分配在 granularity group 边界上进行**。例如，一个权重 tensor 的某些 channel 整体更重要（动态范围更大），希望整个 channel 统一用高精度格式（如 int8），其他 channel 用低精度格式（如 int4）。这比 per-element sparse 更结构化、更利于硬件加速（无需 per-element mask 索引）。

### 与 ADR-012 的本质差异

```
ADR-012 (element sparse):         ADR-013 (group sparse):
mask: per-element                  mask: per-granularity-group
┌──────────────────┐               ┌──────────────────┐
│ H  L  L  H  L  H │               │ H  H  H  L  L  L │  ← channel 0 全 H
│ L  H  L  L  H  L │               │ H  H  H  L  L  L │
│ H  L  H  L  L  H │               │ L  L  L  L  L  L │  ← channel 2 全 L
└──────────────────┘               └──────────────────┘
 一个 channel 内混用 H/L             一个 channel 内统一格式
```

核心区别：
- **Mask 形状**：per-group（如 `(C,)` 表示 C 个 channel 各有 1 个 bool），而非 per-element（`(M, C, ...)`）
- **选择单位**：整个 granularity group（channel / block / bank），而非单个元素
- **组内一致性**：一个 group 内所有元素用同一个格式量化

## 决策

### 1. 新增 `group_format` + `group_ratio` 到 QuantScheme

`QuantScheme` 新增两个字段，与 ADR-012 的 `outlier_format` 并列、互斥：

```python
@dataclass(frozen=True)
class QuantScheme:
    format: FormatBase                    # 主格式（L，默认格式）
    granularity: GranularitySpec
    transform: TransformBase
    ...
    outlier_format: Optional[FormatBase] = None   # ADR-012，不变
    group_format: Optional[FormatBase] = None     # [NEW] H 格式
    group_ratio: float = 0.0                      # [NEW] H group 占比 ∈ [0, 1]
```

**语义**：
- `group_format=None` → group sparse 关闭，行为完全不变
- `group_format=int8, format=int4, group_ratio=0.3` → 30% 的 group 用 int8（H），70% 用 int4（L）

**互斥约束**：`outlier_format is not None` 和 `group_format is not None` 不能同时成立。两种 sparse 模式互斥，在 `__post_init__` 中验证。

**PER_TENSOR 退化**：PER_TENSOR 只有 1 个 group。`group_ratio > 0` 时该唯一的 group 必然是 H，全用 `group_format`。`group_ratio = 0` 时全用 `format`（即 group sparse 关闭时走标准路径）。

### 2. Group 选择算法：per-group amax 评分 + top-k

**评分指标**：per-group `amax`（max absolute value）。

理由：amax 是动态范围的直接衡量。amax 大的 group 包含更大值 → 量化相对误差更大 → 更需要高精度格式。这与量化目标（最小化 clipping + rounding error）一致，且计算成本极低（本身就在量化路径中计算 amax）。

**动态路径**（推理时在线计算）：
```
给定 tensor x, granularity, group_format (非 None), group_ratio:
  1. 按 granularity 切分为 G 个 group
  2. 对每个 group 计算 amax → scores, shape (G,)
  3. k = max(1, int(G * group_ratio))
  4. top-k groups by amax → H groups, 其余 → L groups
  5. H group: group_format.quantize_elemwise(x_h / amax_h) * amax_h
  6. L group: self.format.quantize_elemwise(x_l / amax_l) * amax_l
```

**静态路径**（校准时预计算，推理时直接读取）：
```
校准期:
  1. 收集 S 个 sample 的 tensor（与 ADR-012 的 sample collection 复用）
  2. 对每个 sample 计算 per-group amax → (S, G) scores
  3. cross-sample aggregation: scores = max(amax over S dim) → (G,)
     （取 max 而非 mean：确保任一 sample 中出现过大值的 group 都被标记为 H）
  4. top-k groups by aggregated scores → 存为 _output_group_mask

推理期:
  - 读取 _output_group_mask（固定 bool tensor per-group）
  - H groups → group_format + 对应 amax
  - L groups → format + 对应 amax
```

### 3. GranularitySpec 不承载 group_ratio

`group_ratio` 放在 `QuantScheme` 上而非 `GranularitySpec` 上：

- `GranularitySpec` 描述 scale 共享的几何模式（pure geometry），与格式无关
- `group_ratio` 是格式分配策略，属于 scheme 级别
- 与 `outlier_ratio` 同级（`outlier_ratio` 目前在 `GranularitySpec` 上是因为历史原因，但 `group_ratio` 从一开始就放在正确的位置）

### 4. FormatBase 新增独立 dispatch 分支

在 `quantize()` 方法中，每个 granularity mode 新增 group_sparse 分支，与现有 sparse 分支并列且互斥：

```python
# 以 PER_CHANNEL 为例：
if mode == PER_CHANNEL:
    if granularity.outlier_ratio > 0.0:
        # 现有 ADR-012 sparse，完全不变
        if mask is not None:
            return self._quantize_static_sparse(...)
        return self._quantize_per_channel_sparse(...)
    if group_format is not None and group_ratio > 0.0:    # [NEW]
        if group_mask is not None:                          # [NEW] static
            return self._quantize_per_channel_group_sparse_static(...)
        return self._quantize_per_channel_group_sparse(...) # [NEW] dynamic
    if scale is not None:
        return self._quantize_per_channel(...)
    return self._quantize_per_channel(...)
```

新增方法矩阵：

| Mode | 动态方法 | 静态方法 |
|------|---------|---------|
| PER_TENSOR | `_quantize_per_tensor_group_sparse` | 退化：同动态 |
| PER_CHANNEL | `_quantize_per_channel_group_sparse` | `_quantize_per_channel_group_sparse_static` |
| PER_BLOCK | `_quantize_per_block_group_sparse` | `_quantize_per_block_group_sparse_static` |
| BANK | `_quantize_per_bank_group_sparse` | `_quantize_per_bank_group_sparse_static` |

**注意**：PER_BLOCK 的 group sparse 静态路径**可以实现**（ADR-012 PER_BLOCK element sparse 静态路径是 `NotImplementedError`，因为 per-element mask + shared exponent 的组合不自然）。但 group sparse 下，整个 block 统一 H 或 L → block 的 shared exponent 直接按所属组计算，结构清晰。

### 5. Group Mask 形状

| Mode | Group 数 | Mask 形状 | 示例 |
|------|---------|----------|------|
| PER_TENSOR | 1 | `()` scalar | `True` |
| PER_CHANNEL | C | `(C, 1, 1, ...)` broadcastable | `(64, 1, 1)` |
| PER_BLOCK | B | `(B,)` or broadcastable to block layout | `(128,)` |
| BANK | G | `(G, 1, ...)` broadcastable | `(16, 1, 1)` |

存储为 module buffer：`_output_group_mask` / `_input_group_mask`（与 ADR-012 的 `_output_mask` 命名区分）。

### 6. QuantConfig 新增用户面字段

```python
@dataclass
class QuantConfig:
    ...
    # ADR-012 element sparse (不变)
    outlier_ratio: float = 0.0
    outlier_format: Optional[str] = None
    a_outlier_format: Optional[str] = None

    # [NEW] Group sparse
    group_ratio: float = 0.0                       # H group 占比
    group_format: Optional[str] = None              # H 格式名（如 "int8"）
    a_group_ratio: Optional[float] = None           # Activation override
    a_group_format: Optional[str] = None            # Activation override
```

沿用 `a_format` / `a_outlier_format` 的 override 模式：`a_group_format` 为 None 时跟随 `group_format`。

**互斥验证**：`group_ratio > 0` 和 `outlier_ratio > 0` 不能同时为真。`group_format is not None` 和 `outlier_format is not None` 不能同时为真。

### 7. `compute_group_mask()` 独立函数

对标 `compute_sparse_mask()`，新增独立函数（新文件 `src/quantize/_group_mask.py`）：

```python
def compute_group_mask(
    x_calib: torch.Tensor,            # (S, D1, D2, ...)
    granularity: GranularitySpec,
    group_ratio: float,
) -> torch.Tensor:
    """
    Compute a fixed per-group boolean mask from calibration data.

    Step 1 — Per-sample per-group amax: compute amax within each
             granularity group for each calibration sample → (S, G).
    Step 2 — Cross-sample aggregation: max over sample dim → (G,).
    Step 3 — Top-k groups: select the k = group_ratio * G groups
             with highest aggregated amax.

    Returns:
        Boolean mask with per-group shape (see table in Decision 5).
        True = H group (uses group_format).
    """
```

### 8. CalibrationSession 集成

新增 `_compute_and_assign_group_sparse_state()` 方法，与现有 `_compute_and_assign_sparse_state()` 并列。触发条件：`scheme.group_format is not None and scheme.granularity.group_ratio > 0`（注意这里 `group_ratio` 在 `QuantScheme` 上而非 `GranularitySpec` 上）。

当两个 sparse 模式都未激活时，两个方法均为 no-op。

### 9. QuantScheme.to_op_config 转化

`QuantConfig.to_op_config()` 中，group sparse 参数通过 `QuantScheme` 传递到 `OpQuantConfig`：

```python
w_scheme = QuantScheme(
    format=w_fmt,
    granularity=w_gran,
    ...
    outlier_format=w_outlier_fmt,
    group_format=w_group_fmt,
    group_ratio=w_group_ratio,
)
```

## 备选方案（已拒绝）

**方案 A：在 GranularitySpec 上加 `group_ratio`**：`group_ratio` 是格式分配策略，与粒度几何无关。放在 `GranularitySpec` 上会违反单一职责——granularity 描述 scale 共享模式，不应知道 format 的存在。

**方案 B：复用 ADR-012 的 `outlier_ratio` + `outlier_format`，加一个 mode flag**：`sparse_mode: "element" | "group"` 改变现有字段的语义。风险：mode flag 让字段语义不唯一，代码路径中的条件判断爆炸（`if mode == "group" ... else ...`），且两个模式实际需要的参数不同（element sparse 需要 cross-sample voting 的 top-k 逻辑，group sparse 需要 per-group amax 聚合）。独立字段更清晰。

**方案 C：将 group mask 也存为 per-element 形状**：存储浪费（per-element bool tensor vs per-group bool tensor），且丧失了"组内一致"的语义清晰性。per-group mask + broadcasting 是正确且高效的设计。

## 实现阶段

| Phase | 内容 | 优先级 |
|-------|------|--------|
| P1 | `QuantScheme` 新增 `group_format` + `group_ratio` 字段 + 互斥验证 | P1 |
| P2 | `compute_group_mask()` 独立函数（`src/quantize/_group_mask.py`） | P1 |
| P3 | `FormatBase` 新增各 mode 的 group_sparse 量化方法（动态路径） | P1 |
| P4 | `CalibrationSession` 集成：`_compute_and_assign_group_sparse_state()` | P2 |
| P5 | `FormatBase` 静态路径（pre-computed group_mask） | P2 |
| P6 | `QuantConfig` 用户面字段 + `to_op_config()` 转化 | P2 |
| P7 | 测试：`src/tests/test_group_sparse.py` | P1 |

推荐顺序：**P1 → P2 → P3 → P7（P1-P3 的测试）→ P6 → P4 → P5 → P7（完整测试）**

## 不修改的文件（保证不破坏现有逻辑）

- `GranularitySpec` — 不新增字段
- `src/quantize/_sparse_mask.py` — 完全不动
- `src/formats/_outlier_utils.py` — 完全不动
- 所有现有 `_quantize_*_sparse()` 方法 — 完全不动
- 所有现有测试 — 继续通过
