# Current Task

**Task ID**: Phase 5 — ONNX Export
**Plan**: `docs/plans/2026-04-26-phase5-onnx-export.md`
**Branch**: `feature/refactor-src`

---

## Progress

- [ ] **Task 1（待做）**: `src/onnx/helpers.py` — `_is_standard_format()` + `_emit_quantize_node()`
- [ ] Task 2: `LinearFunction.symbolic()`
- [ ] Task 3: `ConvFunction.symbolic()` + `ConvTransposeFunction.symbolic()`
- [ ] Task 4: `export_quantized_model()` + 端到端测试
- [ ] Task 5: 更新 CURRENT.md

---

## 下一步（具体动作）

执行 `docs/plans/2026-04-26-phase5-onnx-export.md` Task 1：
创建 `src/onnx/__init__.py` + `src/onnx/helpers.py`，
先在 `src/tests/test_onnx_export.py` 写失败测试，再实现。

---

## 断点续传必读文件

1. `docs/plans/2026-04-26-phase5-onnx-export.md`（全文）
2. `src/ops/linear.py`（1–50 行，了解 LinearFunction 结构）
3. `src/ops/conv.py`（52–80 行，了解 ConvFunction 结构）
4. `src/onnx/helpers.py`（Task 1 完成后存在）

---

## 前置状态

- Phase 2–4 全部完成，958 tests passing（含 Phase 5 Task 1 之前新增的 helpers 测试后会更多）
- `onnx 1.21.0` 已安装
- `src/onnx/` 目录尚未创建
- 关键设计验证（新 session 无需重做）：
  - `symbolic()` on `autograd.Function` 在 PyTorch 2.2.2 可用 ✓
  - `OpQuantConfig` 冻结 dataclass 完整传入 `symbolic()` ✓
  - `None` bias → Python `None`（不是 JIT Value），用 `if b is None:` 判断 ✓
  - QDQ 节点生成 + `onnx.checker` 验证通过 ✓

## 关键经验记录

1. **MX block 格式统一走自定义 op**：int8/fp4/fp6 + PER_BLOCK 全部走 `com.microxscaling::MxQuantize`，不走 QDQ。标准 QDQ 只给 int8/int4/int2/fp8 + per_tensor/per_channel。
2. **fp8 QDQ 需 opset 21**：QuantizeLinear float8 variant 在 opset 21 才有（ONNX 1.16+），当前目标 opset 17 → fp8 走 MxQuantize。
3. **QDQ scale 占位常量**：Phase 5 goal 是图结构正确 + checker 通过，不要求 runtime。scale=1.0 / zp=0 是合法占位值。
4. **Conv kernel_shape**：通过 `weight.type().sizes()[2:]` 获取（concrete dummy input 导出时静态 shape 可知）。
5. **padding → ONNX pads**：PyTorch `(ph, pw)` → ONNX `[ph, pw, ph, pw]`（对称填充双写）。
