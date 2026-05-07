# Redundancy Review Report — 2026-05-07

**审视范围**: `src/` 代码冗余、重复逻辑、死代码、过度抽象
**分支**: `feature/refactor-src`
**测试门**: 1,712 passed

---

## 总体评价

框架经过上一轮 P0–P2 修复后，主要的架构性冗余（3条反序列化路径、_utils/单文件包、双份 _VALID_ROUND_MODES）已消除。本轮审查聚焦于代码层面的重复：重复实现、纯委托样板、死代码、以及跨文件的复制粘贴逻辑。发现 1 个高危问题、4 个中等问题、4 个低优先级问题。

---

## 高危问题 (High)

### H1 — `src/quantize/vector.py` 是 `src/ops/vec_ops.py` 的完整副本

两个文件定义了相同的 12 个公开函数（`vec_quantize`, `vec_add`, `vec_sub`, `vec_mul`, `vec_div`, `vec_recip`, `vec_sqrt`, `vec_exp`, `vec_exp2`, `vec_tanh`, `vec_reduce_sum`, `vec_reduce_mean`），共 104 行。

**关键差异**：
- `src/ops/vec_ops.py` — 生产版本，所有内部算术操作包裹在 `_unpatched()` 上下文管理器中（防止重入量化）。被 5 个算子模块导入（`activations.py`, `elemwise.py`, `norm.py`, `pooling.py`, `softmax.py`）。
- `src/quantize/vector.py` — **缺少 `_unpatched()` 保护**，在 patched 上下文中可能触发意外的重入量化。**仅被 4 个测试引用**（`test_scheme_api.py:185,200,210,220`），无任何生产代码导入。

**影响**：两个版本共存造成维护风险——修复一个版本的 bug 不会传播到另一个。`quantize/vector.py` 的 `vec_reduce_mean` 中 `type(dim) is list` 与 `ops/vec_ops.py` 的 `isinstance(dim, list)` 写法也不一致。

**建议**：
1. 删除 `src/quantize/vector.py`
2. 从 `src/quantize/__init__.py` 移除 `from .vector import vec_quantize`
3. `test_scheme_api.py` 改为从 `src.ops.vec_ops` 导入

---

## 中等问题 (Medium)

### M1 — QSNR/MSE 提取循环重复（13 行）

`_session.py:373-386` 和 `_per_layer_opt.py:270-283` 有逐字相同的 13 行嵌套循环，从 observer report 原始数据中提取 `qsnr_db` 和 `mse`。

```python
for layer, roles in observers_data.items():
    for _role, stages in roles.items():
        for _stage, slices in stages.items():
            for _slice_key, metrics in slices.items():
                if "qsnr_db" in metrics:
                    ...
                if "mse" in metrics:
                    ...
```

**建议**：提取为共享工具函数 `_extract_qsnr_mse(observers_data) -> (Dict, Dict)`。

### M2 — `DeviceSpec` 和 `CostReport` 只有测试引用

| 文件 | 行数 | 生产导入 | 测试导入 |
|------|------|---------|---------|
| `cost/device.py` (DeviceSpec) | 46 | **0** | 2 |
| `cost/report.py` (CostReport) | 78 | **0** | 2 |

两个类都没有被任何生产代码导入。唯一的引用来自测试文件。如果这些类是面向未来的 API 预留，应标注 `# TODO:` 注释；否则属于死代码。

### M3 — 3 个 Format 子类的 `quantize()` 是纯委托样板

`FPFormat`, `IntFormat`, `LookupFormat` 各自定义了一个 4 行的 `quantize()` 方法，全部直接调用 `return super().quantize(...)`，无任何额外逻辑。

```python
# fp_formats.py:64-67, int_formats.py:45-48, lookup_formats.py:39-42
def quantize(self, x, granularity, round_mode="nearest", allow_denorm=True,
             scale=None, scale_storage="fp32"):
    return super().quantize(x, granularity, round_mode, allow_denorm,
                            scale=scale, scale_storage=scale_storage)
```

三个方法字符级相同。它们唯一的存在理由是 `FormatBase.quantize()` 被声明为 `@abstractmethod`。

**建议**：将 `FormatBase.quantize()` 改为具体方法（非抽象），移除 3 个空壳 override。`BFloat16Format` 和 `Float16Format` 保留 override（它们有实际的硬件 shortcut 逻辑）。

### M4 — BFloat16Format / Float16Format 的 `quantize()` 结构相同

两个方法的 12 行体完全一致，唯一差异是第 44/81 行的 dtype 字面量（`torch.bfloat16` vs `torch.float16`）。可通过参数化 `_torch_dtype` 属性合并。

此外，`_VALID_ROUND_MODES` 的校验在 fallback 路径上重复——两个子类先校验一次，然后 `super().quantize()` 再校验一次。

---

## 低优先级 (Low)

### L1 — ONNX `symbolic()` 模式重复（约 50 行）

`src/ops/elemwise.py` 中 6 个 SIMD 类的 `symbolic()` 方法遵循相同模板：

```
if inner_scheme: quantize in1
if inner_scheme and in2 is Value: quantize in2
out = g.op("OpName", in1, in2)
if inner_scheme: quantize out
return out
```

可提取为 `_emit_binary_onnx(g, in1, in2, inner_scheme, op_name)`。

### L2 — 测试文件名过时

`test_pipeline_runner.py` 和 `test_pipeline_integration.py` 不再测试 `src/pipeline/`（该包已删除），而是测试 `src.session`。文件名造成误导。

**建议**：重命名为 `test_session_runner.py` / `test_session_integration.py`。

### L3 — `cost/model_cost.py` 单函数文件

`cost/model_cost.py` (55 行) 只导出一个函数 `analyze_model_cost`。按照项目 "单文件包是错的" 原则，但这里 `cost/` 有 4 个文件，不是单文件包。该函数较小但职责独立（模型级 cost 分析），保留独立文件合理。

### L4 — `analysis/context.py` 使用率极低

`AnalysisContext` (57 行) 仅被 2 个生产文件导入（`_quant.py`, `compare.py`）。可考虑内联到调用方或合并到 `analysis/observers.py`。

---

## 确认无冗余的部分

以下模式经检查确认**不是**冗余：

1. **`TransformBase`/`IdentityTransform` 双重导出** — `transform/__init__.py` 有意从 `scheme.transform` 重导出，提供便利导入路径。注释已说明。
2. **`_make_calibrator` 在 `_session.py` 和 `_per_layer_opt.py` 之间** — 上一轮已修复为共享导入。
3. **`OpQuantConfig.from_descriptor()`** — 上一轮已删除。
4. **`_utils/` 包** — 上一轮已删除。
5. **格式注册表** — 惰性初始化 + 线程安全，无冗余。
6. **校准管线** — 4 种 Strategy 各有独立算法，无重复。

---

## 统计摘要

| 类别 | 数量 |
|------|------|
| 高危问题 | 1 |
| 中等问题 | 4 |
| 低优先级 | 4 |
| **待修复总计** | **9** |
| 确认无冗余的设计 | 6 |

---

## 修复优先级建议

**P0 (立即修复)**:
- H1: 删除 `src/quantize/vector.py`（重复实现，无生产引用）

**P1 (本次分支修复)**:
- M1: 提取 QSNR/MSE 提取为共享函数
- M2: 标注或删除 test-only 的 `DeviceSpec` / `CostReport`
- M3: 将 `FormatBase.quantize()` 改为具体方法，删除 3 个纯委托 override
- M4: 合并 BFloat16Format / Float16Format 的 quantize() 重复逻辑

**P2 (下个周期)**:
- L1: 提取 ONNX symbolic() 共享模板
- L2: 重命名过时测试文件
- L4: 评估 `analysis/context.py` 是否可内联
