# Fix Quantization Logic — 问题与修复方案

**来源分支**: `claude/fix-quantization-logic-WRBVt`
**审查日期**: 2026-05-06
**审查结论**: 方向正确，3 个 Critical/Major 问题需修复后合入

---

## 问题清单

### Issue 1 (Critical): PER_BLOCK 零填充污染测量

**位置**: `src/analysis/observer.py` SliceAwareObserver.on_event() PER_BLOCK 分支

**问题描述**:
当 `dim_size % block_size != 0` 时，`reshape(-1, bs)` 需要填充最后一个 block。当前用 `F.pad(..., (0, pad))` 填充零值。填充的零值在 fp32 和 quant 中都是 0，误差为 0，人為拉低 MSE、拉高 QSNR，并在直方图中产生虚假的零值尖峰。

**数值验证**:
```
[2, 10] tensor, bs=4, dim_size=10, pad=2:
  Ground truth mean MSE:     0.007829
  Padded (buggy) mean MSE:   0.007159  ← 9% 向下偏差
  Padded last block MSE:     0.005000  ← should be 0.010000 (50% 偏差)
```

**根因分析**:
`_measure_batch()` 使用 `.mean(dim=1)` 对每行做均值。填充行有 `bs` 个元素但只有 `dim_size % bs` 个真实值，除以 `bs` 而非真实元素数导致偏差。

**修复方案**: valid_counts 向量化掩码

在 PER_BLOCK 分支中构建 `valid_counts` 张量，指示每行有多少个真实元素：

```python
valid_counts = torch.full((fp32_2d.shape[0],), bs, dtype=torch.float32)
if dim_size % bs != 0:
    valid_counts[n_blocks - 1::n_blocks] = dim_size % bs
```

`_measure_batch` 签名增加 `valid_counts=None`，子类使用 `sum(dim=1) / valid_counts` 代替 `mean(dim=1)`。

**数学推导**:
设 block i 有 $k_i$ 个真实元素（$k_i = bs$ for 完整 block，$k_i = r$ for 最后 block，$r = dim\_size \bmod bs$）。
$$\text{MSE}_i = \frac{\sum_{j=1}^{k_i} (fp32_{i,j} - quant_{i,j})^2}{k_i}$$

若除以 `bs`（当前行为）：$\text{MSE}_i = \frac{\sum_{j=1}^{k_i} err^2}{bs} < \frac{\sum_{j=1}^{k_i} err^2}{k_i}$（当 $k_i < bs$）

除以 $k_i$（valid_counts）：数学严格正确。

**受影响文件**:
- `src/analysis/observer.py` — on_event() PER_BLOCK 分支 + _measure_batch 基类
- `src/analysis/observers.py` — QSNRObserver._measure_batch, MSEObserver._measure_batch

---

### Issue 2 (Critical): iter_slices 删除导致测试覆盖缺失

**位置**: `src/tests/test_slicing.py`（240 行，15 个测试）被删除

**问题描述**:
远程分支删除 `src/analysis/slicing.py` 和 `src/tests/test_slicing.py`，将逻辑内联到 `observer.py`。新内联逻辑没有任何直接测试。

**缺失的测试覆盖**:
- PER_TENSOR: key 格式、shape 正确性
- PER_CHANNEL: 正负 axis 等价性、axis 越界报错、key 格式
- PER_BLOCK: 正负 axis 等价性、非 last dim axis、整除/非整除 block、axis 越界
- DYNAMIC_GROUP: group_map 存在/缺失、key 格式
- Unknown mode 报错

**修复方案**:
在 `src/tests/` 中新建测试文件（如 `test_observer_slicing.py`），通过构造 QuantEvent 直接测试 SliceAwareObserver.on_event() 的 per-mode dispatch。

**受影响文件**:
- `src/tests/test_slicing.py` — 被删除
- 新建 `src/tests/test_observer_slicing.py` — 替代测试

---

### Issue 3 (Major): PER_BLOCK 语义变更 — block-index → 单个 MX block

**位置**: `src/analysis/observer.py` SliceAwareObserver.on_event()

**变更说明**:
```
[4, 64, 128] tensor, block_axis=-1, bs=32:
  旧: 4 个 block, 每个 [4, 64, 32] = 8192 elements → key ("block", 0)..("block", 3)
  新: 1024 个 block, 每个 [32] elements        → key ("block_agg",) 聚合统计
```

旧代码按 block **索引**分组（跨所有其他维度合并），新代码测量每个真正的 MX block。旧行为对多维张量在分析上是错误的——它混合了不同 MX scale 的 block。

**下游影响**: 无。`Report.iter_slices()` 和 `figures.py` 是 key-agnostic 的，不依赖具体 key 格式。

**处理**: 接受此变更。在 commit message 中明确说明语义变更。旧行为本身就是 bug。

---

### Issue 4 (Minor): scale=output_scale 从 storage quantize 中移除

**位置**: `src/ops/linear.py` LinearFunction.forward()

**变更说明**:
`quantize(y, cfg.storage, scale=output_scale)` → `quantize(y, cfg.storage)`

**验证**: `FormatBase._quantize_per_tensor()` (base.py:141-144) 忽略 `scale` 参数——直接调用 `quantize_elemwise()` 不使用 scale。`_quantize_per_block()` (base.py:181-187) 明确文档说明忽略 scale。storage 始终 per-tensor，所以 `scale=output_scale` 是无效操作。

**修复**: 应用此变更，添加注释说明为什么 storage 不需要 scale。

---

## 修复执行计划

### Step 1: Issue 1 — valid_counts 修复

1. 数学推导与测试用例设计
2. 写失败测试（`test_observer_slicing.py`）
3. 修改 `observer.py` + `observers.py`
4. 测试通过

### Step 2: Issue 2 — 补全测试覆盖

1. 为所有 4 种 GranularityMode 写 dispatch 测试
2. 为 axis 负值、越界、非整除等边界写测试
3. 验证 key 格式正确性

### Step 3: Issue 4 — scale 移除 + bias compute quant

1. 修改 `linear.py`：移除 storage 的 scale 参数，加注释
2. 修改 `linear.py`/`conv.py`：加 bias compute quant

### Step 4: 全量回归测试

`pytest src/tests/ -x -q`
