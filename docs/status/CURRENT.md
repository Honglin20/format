# Current Task

**Task ID**: P8.R1 — OpQuantConfig 两阶段重构（tuple pipeline → QuantScheme|None）
**Plan**: docs/plans/2026-04-27-opconfig-two-level-impl.md
**Design**: docs/plans/2026-04-27-opconfig-two-level-design.md
**Branch**: refactor/opconfig-two-level（独立 worktree：.claude/worktrees/opconfig-refactor）
**Tests baseline**: 1247 passed, 0 xfail

## 目标

将 OpQuantConfig 从 `Tuple[QuantScheme, ...]` pipeline 重构为 `QuantScheme | None` 两阶段模型（storage + compute），消除所有算子中的 for 循环和 GranularityMode 拆分。

## Progress

- [ ] Task 1: 重构 OpQuantConfig 核心（storage 字段 + QuantScheme|None）
- [ ] Task 2: 更新 Linear 算子
- [ ] Task 3: 更新 Conv / ConvTranspose 算子
- [ ] Task 4: 更新 Norm 算子
- [ ] Task 5: 更新 Activation 算子
- [ ] Task 6: 更新 Softmax
- [ ] Task 7: 更新 Pool
- [ ] Task 8: 更新 Elemwise / Vec ops
- [ ] Task 9: 更新 mapping / quantize_model
- [ ] Task 10: 更新 context / QuantizeContext
- [ ] Task 11: 更新 ONNX helpers
- [ ] Task 12: 更新 session.py
- [ ] Task 13: 更新兼容层（_compat.py）
- [ ] Task 14: 更新所有测试文件
- [ ] Task 15: 全量测试 + 修复 regression（目标：1247+ passed, 0 xfail）
- [ ] Task 16: 更新文档（ADR-005 + CLAUDE.md）
- [ ] Task 17: 最终 Review

## 关键设计决策（已确认）

1. **storage + compute 两步都用 quantize()**：不引入新函数名。quantize() 是 polymorphic 的（FormatBase 根据 granularity 内部分派）
2. **cfg.input 语义复用**：matmul 族作为主输入量化，非 matmul 族自动作为 inner_scheme
3. **直接 breaking change**：不保留 tuple 向后兼容
4. **storage 是 per_tensor elemwise**：包含 bfloat16 硬件 fast path（mx 行为对齐）

## 下一步

在新会话中，以 `.claude/worktrees/opconfig-refactor` 为工作目录，加载 executing-plans skill，执行 Task 1。

## 断点续传必读文件

1. `CLAUDE.md`（全文）
2. `docs/plans/2026-04-27-opconfig-two-level-design.md`（全文）
3. `docs/plans/2026-04-27-opconfig-two-level-impl.md`（全文）
4. `src/scheme/op_config.py`（当前实现）
5. `src/ops/linear.py`（代表性消费方）

## 关键经验记录

（从当前会话继承）
1. **量化两步是 mx 固有设计**：mx 也用 quantize_elemwise_op + quantize_mx_op 两步，这是 MX 规范的固有要求，不是 src/ 的设计缺陷
2. **bfloat16 fast path**：mx 在 round='even' + CUDA BF16 时用 x.to(torch.bfloat16) 硬件截断，src 当前缺失此优化
3. **backward 需要 post-storage 中间值**：两步量化不可合并为一个调用的根本原因
