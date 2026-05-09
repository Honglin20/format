# 新增 Observer 规范

## 架构位置

Observer 位于 `src/analysis/`，接口定义见 `docs/architecture/002-observer-analysis.md`。

## 核心模式

量化算子在关键点发事件，不做 analysis 计算。Observer 通过上下文管理器挂载，接收事件并分析。

### emit_fn 回调模式

- `_emit()` 只能在持有 `self` 的 `QuantizedXxx.forward()` 中调用
- `XxxFunction.forward()` 末尾参数接收 `emit_fn=None`（可选回调）
- `QuantizedXxx.forward()` 传入 `self._emit if self._observers else None`
- Function 内每个量化关键点用 `if emit_fn: emit_fn(role, stage_index, stage, fp32, quant, scheme)` 触发事件
- `stage_index` 固定：0 = storage，1 = compute，2 = output 第二阶段
- 关闭 analysis 时 `_emit` 直接 early return，零开销

## 事件阶段

```python
"input_pre_quant"
"weight_pre_quant"
"output_post_quant"
"grad_output_pre_quant"
"grad_weight_post_quant"
"grad_input_post_quant"
```

## 步骤

### 1. 阅读 ADR

先读 `docs/architecture/002-observer-analysis.md`，理解 Observer 模式。

### 2. 确定基类

- 简单分析：继承 `Observer`（`src/analysis/observers.py`）
- 粒度感知分析：继承 `SliceAwareObserver`
- SliceAwareObserver 子类只需实现 `_measure(key, fp32_slice, quant_slice) -> metric_dict`，自动按 granularity 循环聚合

### 3. 实现 _measure

```python
def _measure(self, key, fp32_slice, quant_slice):
    mse = (fp32_slice - quant_slice).pow(2).mean().item()
    return {"mse": mse}
```

### 4. 切片兼容

通过 `src/analysis/slicing.py::iter_slices(fp32, quant, granularity, group_map)` 统一获取切片。新增 `GranularityMode` 时需同步更新 `iter_slices`。

### 5. 测试

- 在 `src/tests/test_analysis.py` 中添加测试
- 验证 Event 触发次数和内容
- 验证 per-tensor / per-channel / per-block 切片正确

## 接口契约

- Observer 不能 import 量化核心函数（解耦）
- Observer 通过 `AnalysisContext` 挂载：`with AnalysisContext(model, observers=[...]) as ctx: ...`
