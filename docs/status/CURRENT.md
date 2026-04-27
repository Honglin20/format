# Current Task

**Task ID**: P8.1 — Transform + Calibration
**Plan**: docs/plans/2026-04-27-phase8-transform-calibration.md
**Branch**: feature/refactor-src
**Tests baseline**: 1150 passed, 0 xfail

## Progress

- [x] 8A.1 Hadamard Transform（已完成 — 18 tests）
- [x] 8B.1 Scale Strategy（已完成 — 20 tests, review fixes applied）
- [x] 8B.2 Calibration Pipeline（已完成 — 15 tests）
- [x] **8A.2 SmoothQuant Transform（已完成 — 29 tests）**

## 下一步（具体动作）

Phase 8 全完成。建议开启新 task 推进 P1（Transform 体系扩展）或 P2（Calibration 管线完善）：
- P1: CLE / Bias Correction / Hadamard 大维度优化
- P2: Scale persistence（8B.3）+ 校准数据集适配
- 见 `format-research-roadmap.md` 完整优先级清单

## 断点续传必读文件

1. `CLAUDE.md`（全文）
2. `docs/plans/2026-04-27-phase8-transform-calibration.md`（全文）
3. `src/transform/smooth_quant.py`（SmoothQuant 实现 — 已完成）
4. `src/tests/test_transform_smooth_quant.py`（29 tests 全绿）
5. `src/calibration/pipeline.py`（CalibrationPipeline 实现 — 已完成）

## 关键经验记录

1. **FWHT view corruption**：蝶形算法写入前必须同时计算 sum/diff 两个新张量
2. **SmoothQuant 不可变设计**：scale 在 `__init__` 传入，`from_calibration()` 工厂方法
3. **KL 逐 slice（非逐 position）**：`_compute_kl_divergence` 将 axis 排到首维后 reshape(n_slices, -1)，每 slice 建一个直方图（TensorRT 风格）
4. **ScaleStrategy ABC 必须声明 __eq__/__hash__**：否则用作 frozen dataclass 字段时 id-based hash 静默 bug
5. **CalibrationPipeline hook 管理**：`finally` 块保证钩子始终移除（即使 forward raise），`try/finally` 避免资源泄漏
6. **KL scale shape vs other strategies**：Max/Percentile/MSE 返回 per-position scale（axis dim = 1），KL 返回 per-slice scale（n_slices along axis, 1 elsewhere）。测试 shape 断言必须策略感知，不能统一假设
7. **SmoothQuant roundtrip 非 bit-exact**：`forward(x)=x/scale` + `inverse(x_q)=x_q*scale` 的 roundtrip 仅在 scale 为 2 的幂时 bit-exact（`torch.equal`）。非 2 的幂 scale 产生浮点舍入误差（IEEE 754），需用 `allclose(atol=1e-6)`。测试中：手动构造的 power-of-2 scale 用 `torch.equal`，`from_calibration` 计算出的 scale 用 `allclose`
