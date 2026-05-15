# P3: 各 Granularity Mode 接入静态 Mask 路径 — 实施计划

**依赖**: ADR-012 Section 2
**前置**: P1 (BANK), P2 (compute_sparse_mask)

---

## 目标

让所有 4 种 granularity mode 在 `outlier_ratio > 0` 时支持静态 scale（推理路径），不再 raise NotImplementedError。

## Sub-tasks

| Task | 内容 | 文件 |
|------|------|------|
| 1 | extend quantize() + FormatBase.quantize() signature: add `mask` param | `elemwise.py`, `base.py` |
| 2 | implement `_quantize_per_tensor_sparse_static()` | `base.py` |
| 3 | implement `_quantize_per_channel_sparse_static()` | `base.py` |
| 4 | implement PER_BLOCK sparse static path | `base.py` |
| 5 | implement BANK sparse static (via _quantize_per_bank or new) | `base.py` |
| 6 | fix observer BANK support | `observer/observer.py` |
| 7 | fix session integration xfail → passing | `test_bank_granularity.py` |
| 8 | E2E regression (MNIST + Transformer) | scripts |

## 设计决策

### mask 参数

```python
# quantize() 新增 mask 参数
def quantize(x, scheme=None, allow_denorm=True, scale=None, mask=None):
    ...

# FormatBase.quantize() 新增 mask 参数
def quantize(self, x, granularity, round_mode, allow_denorm=True,
             scale=None, scale_storage="pot", mask=None):
    ...
```

### 静态 sparse 统一逻辑

当 `scale is not None` 且 `outlier_ratio > 0` 时：
1. `scale` 是 normal 组的 amax（`amax_n`）
2. `scale_o` 是 outlier 组的 amax（从 `_input_scale_o` buffer 读取）
3. `mask` 是预存的固定 mask

或者更简单的：`scale` 是 tuple `(mask, amax_o, amax_n)` 或单独的 `mask` param + single `scale` as normal group amax.

采用方案：`mask` 作为独立参数，`scale` 作为 normal 组 amax，`scale_o` 作为 outlier 组 amax。
- `mask is None` → 动态 sparse（现有行为，用 topk）
- `mask is not None` → 静态 sparse（推理模式，用预存 mask）

### 模块 buffer 命名

Quantized 模块新增：
- `_input_sparse_mask` / `_output_sparse_mask`: bool tensor
- `_input_scale_o` / `_output_scale_o`: outlier 组 amax
- `_input_scale_n` / `_output_scale_n`: normal 组 amax（即现有的 `_input_scale`/`_output_scale`）
