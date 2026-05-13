# ADR-011: Sparse (Outlier Ratio) 泛化到所有 Granularity

**日期**: 2026-05-13
**状态**: 已决定
**涉及**: `GranularitySpec`、`FormatBase.quantize()`、校准 Pipeline

## 背景

当前 `GranularitySpec.outlier_ratio` 字段仅在 `PER_BLOCK` 模式下生效（`granularity.py:70-73` 硬编码限制）。但 outlier-bank 分离（top-k 离群点独立 scale）的概念与 granularity 正交——per_tensor 和 per_channel 同样受益于将离群点隔离到独立 scale 组。

## 决策

### 1. Sparse 不开第四轴

Sparse 不是与 format / granularity / transform 并列的概念。它是 granularity 的 **scale 分裂参数**——在每个 granularity group 内部将 scale 从 1 组分裂为 2 组（离群 + 正常）。保留 `outlier_ratio` 在 `GranularitySpec` 中。

**理由**:
- 只有一个 concrete case（top-k outlier split），没必要为单一实现建立新抽象层级
- Sparse 做的是"在给定空间分组内，scale 按 magnitude 再次分裂"，这从属于 granularity 的职责
- 删除 per_block 专属校验即可泛化，改动最小

### 2. 解除 PER_BLOCK 限制

`GranularitySpec.__post_init__` 删除 `outlier_ratio > 0 requires PER_BLOCK` 校验。三种 mode 均支持 `outlier_ratio > 0`。

### 3. k 的计算规则

| mode | 一个 group 的元素数 N | k |
|------|----------------------|---|
| PER_TENSOR | `x.numel()` | `max(1, int(N * outlier_ratio))` |
| PER_CHANNEL | `x.numel() / C`（per-channel） | `max(1, int(N * outlier_ratio))` |
| PER_BLOCK | `block_size` | `max(1, int(block_size * outlier_ratio))` |

k >= N 时退化到普通量化。

### 4. Scale 变化

| mode | 无 Sparse | 有 Sparse |
|------|----------|----------|
| PER_TENSOR | 1 个标量 amax | 2 个标量（amax_o, amax_n） |
| PER_CHANNEL | C 个 amax | 2C 个 amax（每组 C 个） |
| PER_BLOCK | 每个 block 1 个 shared exp | 每个 block 2 个 shared exp（已有） |

### 5. 静态量化路径（calibration 后续支持）

Calibration 预存 scale 时 shape 变化：
- PER_TENSOR: `()` → `(2,)`
- PER_CHANNEL: `(C,1,...)` → `(2, C, 1, ...)`
- 需额外存 `_output_sparse_threshold` buffer 用于推理时的 mask 构造

当前先行支持动态路径，静态路径留待后续。

## 实现变更

| 文件 | 变更 |
|------|------|
| `src/scheme/granularity.py` | `__post_init__` 删除 outlier_ratio 的 PER_BLOCK 专属校验 |
| `src/formats/base.py` | `quantize()` 调度加 sparse 分支；新增 `_quantize_per_tensor_sparse`、`_quantize_per_channel_sparse` |
| `src/session/_config.py` | `QuantConfig` 加 `outlier_ratio: float = 0.0`，`to_op_config()` 透传 |
| `src/tests/test_sparse_generalization.py` | per_tensor + per_channel sparse 的格式层测试 + Session 集成测试 |

## 备选方案（已拒绝）

**开第四轴 ScaleStratifyStrategy**: 将 sparse 提升为独立于 granularity 的概念，支持 Top-K / Two-tail / K-means 等多种分裂策略。拒绝原因：当前只有一个 concrete case，过度设计。如果未来确实出现 2-3 种分裂策略，可将 `outlier_ratio` 提升为 `StratifySpec` 子对象挂在 `GranularitySpec` 下。
