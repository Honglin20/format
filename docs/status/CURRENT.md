# Current Task

**当前任务**: 空闲 — GPTQ 幂等修复已完成，GPTQ+sparse 待讨论
**Branch**: `feature/refactor-src`

---

## 已完成：GPTQ 幂等修复

GPTQ 量化后写入 `module._weight_scale` buffer，forward pass 读取该 buffer 传入 `quantize()`，确保 re-quantization 幂等。

E2E 结果：int4-pc baseline 上 GPTQ 从 -0.0004 变为 +0.0004 增益，QSNR 全面提升（13.3→17.1 dB）。

### 提交记录

- `7bde2c3` feat(buffers): add weight_scale field to CalibrationBuffers
- `2ee78aa` test(gptq): add failing tests for _weight_scale buffer and idempotency
- `3b289ae` feat(gptq): register _weight_scale buffer for idempotent re-quantization
- `ecf7af3` test(gptq): add forward-pass weight_scale integration test
- `052f430` feat(linear): read _weight_scale buffer in forward pass
- `e43917d` feat(conv): read _weight_scale buffer in forward pass

---

## 待讨论：GPTQ + Sparse/Group-Sparse

**问题**：GPTQ + sparse/group_sparse 精度仍为负增益（sparse -0.0008, gsparse -0.0018）。

**根因**：GPTQ 内部 `quantize(W_block, scheme, scale=...)` 走了 sparse 路径（int8 outlier + int4 normal），
Hessian 补偿假设每列量化精度相同，但 sparse 打破了这一假设：
- int8 outlier 误差极小，Hessian 补偿在这些列上浪费
- int4 normal 误差大，补偿不足
- sparse mask 在每个 block 子集上独立计算，与 forward 时的 mask 不一致

**待讨论方案**：GPTQ 用纯 int4 scheme（不含 sparse）做 Hessian 补偿，sparse 在 forward 独立处理。
GPTQ 输出的 FP32 权重作为 sparse 的输入，forward sparse mask 从 GPTQ 优化后的权重动态/静态计算。

---

## 断点续传必读文件

1. `src/calibration/gptq_optimizer.py` — GPTQ + _weight_scale buffer
2. `src/ops/linear.py` — QuantizedLinear forward 读取 _weight_scale
3. `src/ops/_calib_buffers.py` — CalibrationBuffers.weight_scale 字段
4. `docs/plans/2026-05-17-gptq-idempotent-design.md` — 幂等修复设计
5. `docs/plans/2026-05-17-gptq-idempotent-plan.md` — 幂等修复实现计划

---

## 已知测试状态

`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"` → 2,627 passed, 40 failed

40 个预存在失败: NF4 equiv tests (~38) + test_viz_save (1) + test_4bit_sparse_analysis (1)
— 均与 GPTQ 幂等修复变更无关。
