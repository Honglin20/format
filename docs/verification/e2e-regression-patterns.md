# E2E Regression Patterns — 端到端回归模式库

> **治理规范**: `docs/standards/e2e-testing.md` §四 — 定义本库的准入标准、合并规则和排除标准。
> 新增条目必须通过 C1-C4 四条件筛选，重复模式必须合并，临时小问题不进入。

每个条目记录一个曾经导致 E2E 精度下降或运行时崩溃的回归模式，包含根因、症状、检测方法和预防规则。
新增修改 `calibration` / `quantize` / `session` / `formats` 的代码时，**必须对照本文所有模式**。

---

## §1: Static Sparse Mask 包含 Batch 维度 (2026-05-15)

### 根因

`CalibrationSession._compute_sparse_state()` 使用 `torch.stack(samples, dim=0)` 把各 batch 的 tensor 堆叠，导致 `x_calib` 的 shape 是 `(num_batches, batch_size, D1, D2, ...)`。

`compute_sparse_mask()` 把 dim 0 当作"第几个 calibration sample"遍历——每次迭代拿到的是一个包含 batch_size 个元素的完整 batch 张量，而非单个 sample。所以：

- **mask shape bug**: mask 包含了 batch 维 → `(batch_size, D1, D2, ...)`，正确的应该是 `(1, *spatial)`
- **scale shape bug**: `_compute_sparse_scales()` 在 dim 0 迭代同样的错误数据，amax 包含了 batch-position 信息

### 症状

1. **运行时 crash**: eval batch_size ≠ calib batch_size 时，`_quantize_per_bank_static_sparse` 中 `mask.expand(x.shape)` 失败（维度不兼容）
2. **精度异常**: 即便 batch_size 碰巧相同，mask 语义也是错的——"第 i 个 batch element 的 position j 是 outlier" vs 正确的"position j 是 outlier（对所有 batch）"

### 修复

`_compute_sparse_state` 中：

```python
# Before (broken):
x_calib = torch.stack(samples, dim=0)  # (num_batches, batch_size, *spatial)
mask = compute_sparse_mask(x_calib, ...)  # mask shape: (batch_size, *spatial) ← 错误!

# After (fixed):
x_calib = torch.cat(samples, dim=0)  # (total_samples, *spatial)
mask = compute_sparse_mask(x_calib, ...)  # mask shape: (*spatial) ← 正确
mask = mask.unsqueeze(0)  # (1, *spatial) — 兼容推理时 broadcasting
```

### 检测脚本

`scripts/verify_batch_independence.py` — 6 项检查：

| 检查 | 内容 |
|------|------|
| R1 | mask shape 的 dim 0 == 1（不泄露 calib batch size） |
| R2 | 同一批 calib data，不同 batch 切分 → 第一层 input mask 相同 |
| R3 | mask 在 batch 维 expand 后所有位置相同（位置语义，非 batch-position） |
| R4 | 推理 batch ≠ 校准 batch 时不 crash，输出 finite |
| R5 | BANK granularity 同样通过 R1-R4 |
| R6 | Session 静态稀疏输出与手动 quantize(..., mask, scale, scale_o) 一致 |

### 预防规则

1. **任何操作 calibration samples 的代码**：必须用 `torch.cat(samples, dim=0)` 展开 batch 维，而非 `torch.stack`
2. **mask 的 shape 约定**：static sparse mask 必须 shape `(1, *spatial)`，dim 0 固定为 1
3. **新增 granularity mode 的 static sparse 支持**：必须在 `verify_batch_independence.py` 中加入对应测试
4. **修改 `CalibrationSession` 的 sample collection**：必须跑 `verify_batch_independence.py`

### 涉及文件

- `src/calibration/pipeline.py` — `_compute_sparse_state`, `_compute_sparse_scales`
- `src/quantize/_sparse_mask.py` — `compute_sparse_mask`
- `src/formats/base.py` — `_quantize_per_bank_static_sparse`, `_quantize_static_sparse`
- `src/ops/linear.py` — `QuantizedLinear.forward` (buffer reading)

---

## §2: Calibration Axis Off-by-One for Multi-Dim Tensors (2026-05-15)

### 根因

`GranularitySpec` 的轴索引（`bank_axis`、`channel_axis`）是为推理时的张量形状定义的，包含 batch 维度（如 `(B, C, L)`）。但在 `_compute_sparse_state` 中，`torch.cat(samples, dim=0)` 之后的逐样本处理丢失了 batch 维——每个 `x_s` 的形状是 `(*spatial)`，比推理时少一个维度。

- **负轴索引（如 `-1`）**：自动适应 rank 变化。`ndim=3` 时 `-1` → 第 2 维，`ndim=2` 时 `-1` → 第 1 维。✓
- **正轴索引（如 `bank_axis=1`）**：不自动适应。推理时 `axis=1` 指向 C 维（`(B, C, L)` 中的 C），但校准样本 `(C, L)` 中 `axis=1` 指向 L 维。mask 和 scale 在**错误的维度**上分组计算。

这导致：
- **BANK**: `_quantize_per_bank_static_sparse` 显式校验 amax numel == num_banks，num_banks 从推理张量推导，amax 从校准错误维度计算 → **numel 不匹配 crash**
- **PER_CHANNEL**: `_quantize_static_sparse` 不做显式校验，但 amax 和 mask 的 per-channel 分组在错误维度 → **静默精度退化**（不 crash，但语义错误）

### 症状

1. **BANK + Conv**: `_quantize_per_bank_static_sparse` 第 638 行 crash：
   ```
   ValueError: amax_n has K elements but M banks are expected.
   Shape (...) cannot be reshaped to target (...).
   ```
2. **PER_CHANNEL + Conv**: 不 crash，但 mask/scale 分组语义错误，表现为精度退化。

Linear 层不受影响（`bank_axis=-1` 或 1D 空间形状，正负轴索引等价）。Conv 层使用 `bank_axis=1`(channel dimension)时触发。

### 修复

`src/calibration/pipeline.py` 新增 `_adjust_gran_axes_for_calibration()` 函数，在 `_compute_sparse_state` 中调用：

```python
def _adjust_gran_axes_for_calibration(gran):
    """Calibration samples lack batch dim; shift positive axes down by 1."""
    if gran is None:
        return gran
    bank_axis = gran.bank_axis - 1 if gran.bank_axis > 0 else gran.bank_axis
    channel_axis = gran.channel_axis - 1 if gran.channel_axis > 0 else gran.channel_axis
    block_axis = gran.block_axis - 1 if gran.block_axis > 0 else gran.block_axis
    return GranularitySpec(
        mode=gran.mode, block_size=gran.block_size,
        channel_axis=channel_axis, block_axis=block_axis,
        bank_size=gran.bank_size, bank_axis=bank_axis,
        outlier_ratio=gran.outlier_ratio,
    )
```

- 只有正轴索引（`> 0`）需要递减 1。
- 负轴索引（`-1`、`-2`）自动适应 rank 变化，不变。
- `axis=0` 指向 batch 维本身（极少见），递减为 `-1` 也不合理，但此场景不应出现。

### 检测脚本

`scripts/verify_mask_shapes.py` §4 — BANK Conv1d/Conv2d 专项：
- 4.1: Conv1d, bank on channel axis (C=4, bank=4 → 1 bank)
- 4.2: Conv2d, bank on channel axis (C=8, bank=4 → 2 banks)

`scripts/verify_batch_independence.py` R1-R6 — 所有 mask shape dim0==1 检查。

### 预防规则

1. **任何 `_compute_sparse_state` 或类似校准函数**：处理逐样本张量前必须通过 `_adjust_gran_axes_for_calibration` 调整轴索引。
2. **新增 granularity mode 的 axis 参数**：必须在 `_adjust_gran_axes_for_calibration` 中添加对应字段的调整逻辑。
3. **修改 `compute_sparse_mask` 或 `_compute_sparse_scales`**：确认传入的 `GranularitySpec` 已调整轴索引。
4. **Conv 层 + BANK/PER_CHANNEL 测试**：必须覆盖 bank_axis=1 (channel dim) 的校准→推理全流程。

### 涉及文件

- `src/calibration/pipeline.py` — `_adjust_gran_axes_for_calibration`, `_compute_sparse_state`
- `src/quantize/_sparse_mask.py` — `compute_sparse_mask` (接收已调整的 gran)
- `src/formats/base.py` — `_quantize_per_bank_static_sparse` (校验 amax numel)
- `scripts/verify_mask_shapes.py` — §4 Conv BANK 测试
- `scripts/verify_batch_independence.py` — R1-R6 mask shape 检查

---
## §3: `_compute_group_amax` Broadcast ndim Mismatch (2026-05-15)

### 根因

`_compute_group_amax()` 计算 `x_sel = x * sel.float()` 后，mask 的 `(1, *spatial)` shape 与 per-sample tensor `x` 的 `(*spatial)` shape 广播，`x_sel` 的 ndim 比 `x` 多 1（mask 的 batch 维）。

但 PER_CHANNEL / BANK 模式的 axis 解析使用 `x.ndim`：
- `axis = x.ndim + axis`（负轴解析）
- `dims_to_reduce = [i for i in range(x.ndim) if i != axis]`
- `N_along = x.shape[axis]`（BANK 模式）

当 `x` 与 `x_sel` ndim 不同时，axis 指向了错误的语义维度，且 `dims_to_reduce` 为空时 `torch.amax(..., dim=())` 会**归约所有维度**（而非不归约）。

### 症状

1. **PER_CHANNEL + channel_axis=-1 + Linear**: scale numel = 1（应为 in_features）。静默精度退化——每个 channel 本应有独立的 amax，实际被「平均」为一个标量。
2. **BANK + bank_axis=-1 + 1D tensor**: 同样的广播 ndim 偏差导致错误 shape。
3. **PER_CHANNEL + channel_axis=1（正轴）**: 不受影响——`_adjust_gran_axes_for_calibration` 递减正轴，axis 指向 x_calib 的第二维，broadcast 添加的 batch 维恰好在 axis 之前，`dims_to_reduce` 非空。

### 修复

`src/calibration/pipeline.py` — `_compute_group_amax()`：axis 解析和 dims_to_reduce 全部使用 `x_sel` 而非 `x`，并处理 empty dims_to_reduce 边界情况：

```python
# PER_CHANNEL:
axis = gran.channel_axis
if axis < 0:
    axis = x_sel.ndim + axis       # ← was x.ndim
dims_to_reduce = [i for i in range(x_sel.ndim) if i != axis]  # ← was x.ndim
if not dims_to_reduce:
    return torch.abs(x_sel)        # ← handle empty: no reduction needed
return torch.amax(torch.abs(x_sel), dim=tuple(dims_to_reduce), keepdim=True)

# BANK: same changes — x_sel.ndim for axis resolution, x_sel.shape for N_along/new_shape
```

### 检测脚本

`scripts/verify_mask_shapes.py` §6.4 (PER_CHANNEL+int4+bfloat16 outlier)、§7.2 (PER_CHANNEL+int8)、§12 combinatorial — 这些用例在 Linear 层上使用 `channel_axis=-1`，会触发 broadcast ndim 偏差并验证 scale numel。

### 预防规则

1. **`_compute_group_amax` 或任何 `x * mask` 模式**：若 mask 带有 batch 维，必须使用乘法结果 `x_sel` 的 ndim/shape 进行 axis 解析。
2. **`torch.amax(tensor, dim=())` 行为陷阱**：PyTorch 中 `dim=()` 表示「归约所有维度」，而非「不归约」。`dims_to_reduce` 为空时需显式 return `torch.abs(x_sel)`。
3. **Linear 层 + channel_axis=-1 + PER_CHANNEL 测试**：必须覆盖此组合（1D tensor 最易触发 broadcast 维偏差）。

### 涉及文件

- `src/calibration/pipeline.py` — `_compute_group_amax`

---

### 根因

### 症状

### 修复

### 检测脚本

### 预防规则

### 涉及文件

-->
