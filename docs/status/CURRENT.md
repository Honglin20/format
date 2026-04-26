# Current Task

**Task ID**: 全部完成
**Branch**: `feature/refactor-src`

---

## 完成状态

所有计划 Phase 全部完成，1003 tests passing。

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
- [x] **QuantizeContext（Phase 6）**：统一 all-op 量化上下文管理器
  - patches torch.matmul/mm/bmm、torch.add/sub/mul/div/exp/log、F.linear
  - Module-stack hooks 支持 Observer 层名追踪
  - ONNX export via ctx.export_onnx()（在激活 context 内运行，symbolic() 正常触发）
  - MatMulFunction、BMMFunction、6 个 SIMD Function 新增 symbolic() 方法
  - 实现计划：docs/plans/2026-04-26-quantize-context.md

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
7. **namespace patch 的延迟导入陷阱**：`_patches.py` 中不能用延迟 import（函数体内 `from src.ops.matmul import ...`）。若 matmul.py 在 patches 激活后才首次被导入，`_torch_matmul = torch.matmul` 会捕获已打补丁的版本，导致无限递归。所有 Function 类必须在 `_patches.py` 模块顶层 eager import。
8. **torch namespace patch 不影响已保存引用**：`_torch_matmul = torch.matmul`（模块加载时）之后 `torch.matmul = _patched_matmul` 不会修改 `_torch_matmul`，但前提是 eager import 先于任何 patch（见经验 7）。
