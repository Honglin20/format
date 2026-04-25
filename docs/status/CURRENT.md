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
- [ ] **缺陷修复（fix-spec，进行中）**
- [ ] P3.5: Elemwise/SIMD/Vector ops
- [ ] P3.6: mapping + 端到端验证

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
- [ ] **subtask-fix-3**: M3 — `_compat.py::_matmul_backward_pipelines` round_key 修正
- [ ] **subtask-fix-4**: C1 — Linear/Conv/Matmul/BMM 的 `emit_fn` 接入 + 4 条 observer 测试
- [ ] **subtask-fix-5**: C1 — Norm/Activation/Softmax/Pool 的 `emit_fn` 接入
- [ ] **subtask-fix-6**: m1 — QuantizedLinear passthrough 缓存到 `_is_passthrough`
- [ ] **subtask-fix-7**: m3 — QuantizedGELU `inner_scheme=None` 条件修正
- [ ] **subtask-fix-8**: m5 — 删除 `src/ops/nn/` 空目录

---

## 下一步（具体动作）

开始 subtask-fix-3：修正 `src/tests/_compat.py::_matmul_backward_pipelines` 中 grad_weight round_key 从 `"round_grad_input"` 改为 `"round_grad_weight"`（见 fix-spec 文档和 Phase 3 review 报告 M3 章节）。

---

## 断点续传必读文件

1. `docs/plans/2026-04-25-defect-fix-specs.md`（M3 章节，若有；否则参考 Phase 3 review 报告）— round_key 修正
2. `docs/plans/2026-04-25-phase3-review.md`（M3 条目）— matmul grad_weight round_key 详情
3. `src/tests/_compat.py`（`_matmul_backward_pipelines` 函数）— 修复点

---

## 关键经验记录

1. **`emit_fn` 回调模式**：`autograd.Function` 无 self，不能直接调用 `_emit()`。解决方案是将绑定方法 `self._emit` 作为 `emit_fn` 参数传入 Function，无 observers 时传 `None`，靠 `if emit_fn:` 保证零开销。
2. **inner_scheme 到 OpQuantConfig 的映射约定**：Activation/Softmax/Pool 的 `inner_scheme` 对应 `cfg.input[0]`（forward）和 `cfg.grad_input[0]`（backward QAT）。`quantize_backprop=False` 等价于 `cfg.grad_input = ()`。
3. **iter_slices PER_BLOCK block_axis**：应使用通用 index tuple（`[slice(None)] * ndim`，然后在 `axis` 维度替换为范围切片），而非写死 `fp32[..., sl]`。
4. **P3 完成度约 70%**：fix-spec 全部完成后，P3.5 + P3.6 才是 Phase 3 的真正收尾。
