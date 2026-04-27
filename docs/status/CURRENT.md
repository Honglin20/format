# Current Task

**Status**: 全部实现 Phase 完成，待启动 Phase 8（研究能力扩展）
**Branch**: `feature/refactor-src`
**Tests**: 1068 passed, 0 xfail

---

## 完成状态

所有计划 Phase + 追加 Phase 全部完成。

- [x] **Phase 2**：三轴架构扶正（Format + Granularity + Transform）+ P2F-7 缺陷收口
- [x] **Phase 3**：全算子层（P3.0–P3.6）
- [x] **Phase 4**：层级误差分析（AnalysisContext + QSNR/MSE/Histogram/Distribution Observer）
- [x] **Phase 5**：ONNX export（QDQ + MxQuantize custom op，symbolic() 全算子覆盖）
- [x] **Phase 6**：QuantizeContext 统一 all-op 量化上下文管理器
- [x] **Phase 7**：Unified quantize_model（Module 替换 + forward patching + model.export_onnx）
- [x] **追加修复**：
  - Format.export_onnx() Strategy — 每个 Format 自声明 ONNX 导出行为
  - `_quantize_per_block` JIT tracing guard — 消除唯一 xfail（`torch.jit.is_tracing() → return x`）

---

## 下一步

启动 **Phase 8 — 研究能力扩展**。按优先级：

| 优先级 | 任务 | 核心范围 |
|---|---|---|
| **P1** | Transform 体系 | SmoothQuant、Bias correction、CLE、Hadamard rotation |
| **P2** | Calibration 管线 | 可插拔 scale 策略（max/percentile/MSE/KL）+ 校准数据集 + scale 持久化 |
| P3 | NF4 / 查找表格式 | FormatBase 支持 `levels: Optional[Tensor]` |
| P4 | 参数化格式注册 | `register_float_format()` 一行注册 |
| P5 | 可学习量化参数 | scale_mode="learnable", STE gradient |
| P6 | Coarse Model | 按 transform 类型估算延迟/吞吐 |

完整清单及细节见 `memory/format-research-roadmap.md`（auto memory，新 session 自动加载）。

---

## 断点续传必读文件

1. `CLAUDE.md`（全文 — 项目规范）
2. `docs/architecture/001-three-axis-quant-scheme.md`（QuantScheme + Transform 接口）
3. `src/formats/base.py`（FormatBase.quantize / _quantize_per_block / export_onnx）
4. `src/scheme/quant_scheme.py`（QuantScheme 数据类 + TransformBase）
5. `src/scheme/op_config.py`（OpQuantConfig pipeline 容器）

---

## 关键经验记录

1. **MX block 格式走自定义 op**：per_block 一律 `com.microxscaling::MxQuantize`，其余走 QDQ
2. **ONNX 导出两阶段分离**：JIT tracing for shapes (forward) / symbolic for ONNX nodes。`_quantize_per_block` 在 tracing 时 skip（`torch.jit.is_tracing()`），symbolic() 负责生成真实量化节点
3. **`_bp` vs forward format keys**：linear backward 用 `_bp` key，conv backward 用 forward key
4. **backward 保存 post-elemwise 张量**：所有 ops 在 backward 中保存 elemwise 后（pre-MX）的张量
5. **emit_fn 回调模式**：`_emit` 只在 `QuantizedXxx.forward()` 中持有，通过 `emit_fn` 参数传入 Function
6. **patches.py 的延迟导入陷阱**：所有 Function 类必须在 `_patches.py` 模块顶层 eager import，避免捕获已打补丁的版本
7. **Forward patching 比 Module 容器更优**：避免 state_dict key 前缀问题
8. **Forward patching 必须用闭包而非 `types.MethodType`**：`MethodType` 导致双重 `self`
9. **`torch.onnx.dynamo_export` 不可用**（PyTorch 2.2.2）：ContextVar 不兼容 Dynamo tracer
10. **Format.export_onnx() Strategy 模式**：每个 Format 自声明 ONNX 导出行为，与 `quantize()` Strategy 对称
