# Current Task

**Task ID**: P8.1 — Transform + Calibration
**Plan**: docs/plans/2026-04-27-phase8-transform-calibration.md
**Branch**: feature/refactor-src
**Tests baseline**: 1121 passed, 0 xfail

## Progress

- [x] 8A.1 Hadamard Transform（已完成 — 18 tests）
- [x] 8B.1 Scale Strategy（已完成 — 20 tests, review fixes applied）
- [x] **8B.2 Calibration Pipeline（已完成 — 15 tests）**
- [ ] **8A.2 SmoothQuant Transform（进行中）**

## 下一步（具体动作）

8A.2 继续：修复 `test_compute_scale_all_positive` 中的 shape 不匹配（X act_amax != W w_amax）。问题在 `src/tests/test_transform_smooth_quant.py:136`：X(4,8) 与 W(8,16) 的 input channel 不匹配。

## 断点续传必读文件

1. `CLAUDE.md`（全文）
2. `docs/plans/2026-04-27-phase8-transform-calibration.md`（全文 — 设计决策 + 接口规范）
3. `src/transform/smooth_quant.py`（SmoothQuant 实现 — 8A.2 正在修改）
4. `src/tests/test_transform_smooth_quant.py`（测试 — 存在 1 个已知失败）
5. `src/calibration/pipeline.py`（CalibrationPipeline 实现 — 8B.2 已完成）

## 关键经验记录

1. **FWHT view corruption**：蝶形算法写入前必须同时计算 sum/diff 两个新张量
2. **SmoothQuant 不可变设计**：scale 在 `__init__` 传入，`from_calibration()` 工厂方法
3. **KL 逐 slice（非逐 position）**：`_compute_kl_divergence` 将 axis 排到首维后 reshape(n_slices, -1)，每 slice 建一个直方图（TensorRT 风格）
4. **ScaleStrategy ABC 必须声明 __eq__/__hash__**：否则用作 frozen dataclass 字段时 id-based hash 静默 bug
5. **CalibrationPipeline hook 管理**：`finally` 块保证钩子始终移除（即使 forward raise），`try/finally` 避免资源泄漏
6. **KL scale shape vs other strategies**：Max/Percentile/MSE 返回 per-position scale（axis dim = 1），KL 返回 per-slice scale（n_slices along axis, 1 elsewhere）。测试 shape 断言必须策略感知，不能统一假设
