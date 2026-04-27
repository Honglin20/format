# Current Task

**Task ID**: P8.1 — Transform + Calibration
**Plan**: docs/plans/2026-04-27-phase8-transform-calibration.md
**Branch**: feature/refactor-src
**Tests baseline**: 1096 passed, 0 xfail

## Progress

- [x] 8A.1 Hadamard Transform（已完成 — 18 tests）
- [x] 8B.1 Scale Strategy（已完成 — 10 tests）
- [ ] **8B.2 Calibration Pipeline（下一步）**
- [ ] 8A.2 SmoothQuant Transform

## 下一步（具体动作）

8B.2 Calibration Pipeline — 创建 `src/calibration/pipeline.py`（DataLoader 遍历 + 统计聚合），依赖 8B.1 的 ScaleStrategy。

## 断点续传必读文件

1. `CLAUDE.md`（全文）
2. `docs/plans/2026-04-27-phase8-transform-calibration.md`（全文）
3. `src/calibration/strategies.py`（ScaleStrategy 接口 — 8B.2 依赖）
4. `src/transform/hadamard.py`（HadamardTransform 实现模式 — 8A.2 参考）

## 关键经验记录

1. **FWHT view corruption**：蝶形算法中 `a`/`b` 是视图，写入前必须同时计算 `sum_ab` 和 `diff_ab` 两个新张量，再做两次写入
2. **SmoothQuant 不可变设计**：scale 在 `__init__` 传入，`from_calibration()` 工厂方法
3. **CLE / Bias Correction 不在本次范围**
