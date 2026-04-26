# Current Task

**Task ID**: Phase 5 — ONNX Export（已完成）
**Plan**: `docs/plans/2026-04-26-phase5-onnx-export.md`
**Branch**: `feature/refactor-src`

---

## Progress

- [x] Task 1: `src/onnx/helpers.py` — `_is_standard_format()` + `_emit_quantize_node()`
- [x] Task 2: `LinearFunction.symbolic()`
- [x] Task 3: `ConvFunction.symbolic()` + `ConvTransposeFunction.symbolic()`
- [x] Task 4: `export_quantized_model()` + 端到端测试
- [x] Task 5: 更新 CURRENT.md

---

## 当前状态

**Phase 5 完成。** 973 tests passing（958 原有 + 15 新增 ONNX 测试）。

---

## 下一步

Phase 1–5 全部完成。可选方向：

1. **P2F-7** — Phase 2 review 遗留缺陷收口（granularity 类型 guard、channel_axis 越界断言、silent default docstring）
2. **后续扩展** — 新格式、新算子、ORT runtime 适配等

---

## 断点续传必读文件

1. `docs/plans/2026-04-26-phase5-onnx-export.md`（全文，了解设计决策）
2. `src/onnx/helpers.py`（全文）
3. `src/onnx/export.py`（全文）
4. `src/ops/linear.py`（185–220 行，`symbolic()` 实现）
5. `src/ops/conv.py`（230–310 行，`symbolic()` 实现）

---

## 关键经验记录

1. **JIT 追踪时 size() 返回 Tensor**：`torch.jit.trace` 内 `A.size()[i]` 返回 `torch.Tensor`（非 Python int）。`_reshape_to_blocks` 原代码用 TorchScript 专有 `.value` 导致崩溃，改为 `.item()` 修复。
2. **MX block 格式统一走自定义 op**：int8/fp4/fp6 + PER_BLOCK 全部走 `com.microxscaling::MxQuantize`，不走 QDQ。标准 QDQ 只给 int8/int4/int2/fp8 + per_tensor/per_channel。
3. **ONNX 追踪两阶段**：PyTorch ONNX 导出先用具体 dummy input 执行 `forward()`（JIT trace），再调用 `symbolic()` 生成 ONNX 节点。`forward()` 必须能跑通，`symbolic()` 才能被触发。
4. **QDQ scale 占位常量**：scale=1.0 / zp=0 是合法占位值，`onnx.checker` 可通过，不要求 runtime。
5. **Conv kernel_shape**：通过 `weight.type().sizes()[2:]` 在 `symbolic()` 内获取（static shape export 时可知）。
6. **custom_opsets 必须注册**：`torch.onnx.export(..., custom_opsets={"com.microxscaling": 1})` 是必须参数，否则 checker 拒绝自定义 domain 节点。
