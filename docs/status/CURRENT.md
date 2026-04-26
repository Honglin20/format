# Current Task

**Task ID**: 全部完成
**Branch**: `feature/refactor-src`

---

## 完成状态

所有计划 Phase 全部完成，973 tests passing。

- [x] **Phase 2**：三轴架构扶正（Format + Granularity + Transform）+ P2F-7 缺陷收口
- [x] **Phase 3**：全算子层（P3.0–P3.6）
  - [x] P3.0: P2F-7 granularity 类型 guard + channel_axis 越界断言
  - [x] P3.1: OpQuantConfig、ObservableMixin、iter_slices、Linear/Matmul/BMM
  - [x] P3.2: Conv1d/2d/3d + ConvTranspose1d/2d/3d
  - [x] P3.3: BatchNorm1d/2d/3d + LayerNorm + GroupNorm + RMSNorm
  - [x] P3.4: 7 种激活函数 + Softmax + AdaptiveAvgPool
  - [x] P3.5: Elementwise 算子 + SIMD / vec_ops
  - [x] P3.6: quantize_model 入口 + 端到端 small model bit-exact 验证
- [x] **Phase 4**：层级误差分析（AnalysisContext + QSNR/MSE/Histogram/Distribution Observer）
- [x] **Phase 5**：ONNX export（QDQ + com.microxscaling::MxQuantize）

---

## 下一步

无计划内未完成项。可选扩展方向：
- 新格式（如 fp4_e3m0、int1）
- ORT / TensorRT runtime 适配
- RNN 家族算子（原计划排除）
- 动态分组量化（iter_slices 已预留扩展位）

---

## 关键经验记录

1. **MX block 格式走自定义 op**：per_block 一律 `com.microxscaling::MxQuantize`，其余走 QDQ
2. **JIT 追踪时 size() 返回 Tensor**：`_reshape_to_blocks` 的 `.value` 改为 `.item()`
3. **ONNX 导出两阶段**：先执行 `forward()` 建图（需跑通），再调 `symbolic()` 生成节点
4. **`_bp` vs forward format keys**：linear backward 用 `_bp` key，conv backward 用 forward key
5. **backward 保存 post-elemwise 张量**：所有 ops 在 backward 中保存 elemwise 后（pre-MX）的张量
6. **emit_fn 回调模式**：`_emit` 只在 `QuantizedXxx.forward()` 中持有，通过 `emit_fn` 参数传入 Function
