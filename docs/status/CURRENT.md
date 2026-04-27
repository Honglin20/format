# Current Task

**Task ID**: P8.1 — Transform + Calibration
**Plan**: docs/plans/2026-04-27-phase8-transform-calibration.md
**Branch**: feature/refactor-src
**Tests baseline**: 1096 passed, 0 xfail

## Progress

- [x] **8A.1 Hadamard Transform（已完成）**
- [ ] **8B.1 Scale Strategy（进行中）**
- [ ] 8B.2 Calibration Pipeline
- [ ] 8A.2 SmoothQuant Transform

## 下一步（具体动作）

8A.1 已完成（HadamardTransform + 18 tests, 1096 total passing）。下一步 8A.2 SmoothQuant Transform。

## 断点续传必读文件

1. `CLAUDE.md`（全文 — 项目规范）
2. `docs/plans/2026-04-27-phase8-transform-calibration.md`（全文 — 实现计划 + 设计决策）
3. `docs/architecture/001-three-axis-quant-scheme.md`（TransformBase 接口）
4. `src/scheme/transform.py`（TransformBase + IdentityTransform 现有实现）
5. `src/transform/hadamard.py`（HadamardTransform 实现模式 — 8A.2 SmoothQuant 参考）
6. `src/formats/base.py`（_quantize_per_channel 现有 amax scale 逻辑，8B.1 回归目标）

## 关键经验记录

1. **MX block 格式走自定义 op**：per_block 一律 `com.microxscaling::MxQuantize`，其余走 QDQ
2. **ONNX 导出两阶段分离**：JIT tracing for shapes (forward) / symbolic for ONNX nodes
3. **SmoothQuant 不可变设计**：scale 在 `__init__` 传入，`from_calibration()` 工厂方法，消除"未校准"非法状态
4. **CLE / Bias Correction 不在本次范围**
5. **FWHT view corruption**：在 `x_2d[:, i:i+h] = a + b` 中 `a` 和 `b` 是视图，写入 `x_2d` 会同时修改 `a`/`b` 的底层内存。必须在写入前同时计算 `sum_ab` 和 `diff_ab` 两个新张量，再做两次写入
