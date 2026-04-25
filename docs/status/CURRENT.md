# Current Task

**Task ID**: Phase 3 缺陷修复（fix-spec 落地）  
**修复规范**: `docs/plans/2026-04-25-defect-fix-specs.md`  
**Review报告**: `docs/plans/2026-04-25-phase3-review.md`  
**Plan**: `docs/plans/2026-04-24-phase3.md`  
**Branch**: `feature/refactor-src`

---

## Phase 3 实际完成状态

- [x] P3.0: P2F-7 缺陷收口
- [x] P3.1: OpQuantConfig + ObservableMixin + Linear/MatMul/BMM
- [x] P3.2: Conv + ConvTranspose（44 tests）
- [x] P3.3: BatchNorm/LayerNorm/GroupNorm/RMSNorm（32 tests）
- [x] P3.4: Activation/Softmax/AdaptiveAvgPool2d（bit-exact）
- [x] **缺陷修复（fix-spec，已完成）** — 8 个子任务全部修复
- [x] P3.5: Elemwise/SIMD/Vector ops（46 个等价性测试）
- [x] P3.6: mapping + 端到端验证（8 个 E2E 测试）

---

## 设计决策（已确认）

### M1：Activation/Softmax/Pool 接口统一方案 → **方案 A'**

模块类 `__init__` 统一为 `cfg: OpQuantConfig`，保留 `inner_scheme` 向下兼容参数（自动转换为 OpQuantConfig）。`XxxFunction` 内部参数不变（仍用 `inner_scheme`），由 module forward 从 `cfg.input[0]` 提取。

### C1：`_emit()` 接入模式 → **`emit_fn` 回调模式**

`QuantizedXxx.forward()` 传 `emit_fn = self._emit if self._observers else None` 给 Function，Function 内每个量化关键点用 `if emit_fn: emit_fn(...)` 触发。零开销，无需改动 mixin。

### M2：`iter_slices` PER_BLOCK 修复 → **沿 `block_axis` 切片**

用通用 index tuple 替换写死的 `fp32[..., sl]`，沿 `granularity.block_axis` 切片。

---

## 待实现子任务（按顺序）

- [x] **subtask-fix-1**: M2 — `src/analysis/slicing.py` block_axis 修复 + 4 条新测试
- [x] **subtask-fix-2**: M1 — Activation/Softmax/Pool `__init__` 接口统一 + `_compat.py` 适配器返回值变更 + 5 条新测试
- [x] **subtask-fix-3**: M3 — `_compat.py::_matmul_backward_pipelines` round_key 修正
- [x] **subtask-fix-4**: C1 — Linear/Conv/Matmul/BMM 的 `emit_fn` 接入 + 4 条 observer 测试
- [x] **subtask-fix-5**: C1 — Norm/Activation/Softmax/Pool 的 `emit_fn` 接入
- [x] **subtask-fix-6**: m1 — QuantizedLinear passthrough 缓存到 `_is_passthrough`
- [x] **subtask-fix-7**: m3 — QuantizedGELU `inner_scheme=None` 条件修正
- [x] **subtask-fix-8**: m5 — 删除 `src/ops/nn/` 空目录
---

## Phase 3 完成总结

| 阶段 | 状态 | 测试数 |
|---|---|---|
| P3.1 Matmul 家族 | 完成 | Linear/Matmul/BMM + emit_fn |
| P3.2 Conv | 完成 | Conv1d/2d/3d + ConvTranspose |
| P3.3 Norm | 完成 | BN/LN/GN/RMS + emit_fn |
| P3.4 Activation/Softmax/Pool | 完成 | 7 Act + Softmax + Pool |
| P3.5 Elemwise/SIMD | 完成 | 10 SIMD ops + 46 tests |
| P3.6 Mapping + E2E | 完成 | quantize_model + 8 E2E tests |
| 缺陷修复 | 完成 | M1/M2/M3 + C1 emit_fn + m1/m3/m5 |

**总计：730 tests passed，bit-exact**

---

## 下一步（具体动作）

Phase 3 全部完成。可进入 Phase 4（层级误差分析）或清理收尾。

---

## 断点续传必读文件

1. `docs/plans/2026-04-24-phase3.md`（全文）— Phase 3 完成记录
2. `docs/architecture/002-observer-analysis.md` — Phase 4 设计参考
3. `src/analysis/mixin.py`（全文）— Observer 基础设施

---

## 关键经验记录

1. **`emit_fn` 回调模式**：`autograd.Function` 无 self，不能直接调用 `_emit()`。解决方案是将绑定方法 `self._emit` 作为 `emit_fn` 参数传入 Function，无 observers 时传 `None`，靠 `if emit_fn:` 保证零开销。
2. **inner_scheme 到 OpQuantConfig 的映射约定**：Activation/Softmax/Pool 的 `inner_scheme` 对应 `cfg.input[0]`（forward）和 `cfg.grad_input[0]`（backward QAT）。`quantize_backprop=False` 等价于 `cfg.grad_input = ()`。
3. **iter_slices PER_BLOCK block_axis**：应使用通用 index tuple（`[slice(None)] * ndim`，然后在 `axis` 维度替换为范围切片），而非写死 `fp32[..., sl]`。
4. **P3 完成度约 70%**：fix-spec 全部完成后，P3.5 + P3.6 才是 Phase 3 的真正收尾。
