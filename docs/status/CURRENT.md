# Current Task

**当前任务**: Sparse 泛化完成 → ADR-009 收尾 + P7 自动格式搜索
**下一任务**: Calibration 增强（含 sparse 静态路径） + QAT 验证
**Branch**: `feature/refactor-src`

---

## Sparse (outlier_ratio) 泛化 — 已完成 (2026-05-13)

`outlier_ratio` 从仅 `PER_BLOCK` 扩展到 `PER_TENSOR` 和 `PER_CHANNEL`。
设计决策: 不开第四轴，保留在 `GranularitySpec` 中（见 ADR-011）。

| Phase | 内容 | 状态 |
|-------|------|------|
| 设计文档 | `docs/architecture/011-sparse-generalization.md` | [x] |
| 数学推导 | `docs/verification/018-sparse-per-tensor.md` + `019-sparse-per-channel.md` | [x] |
| TDD RED | 23 fail (功能缺失) | [x] |
| TDD GREEN | 28 pass (per_tensor + per_channel sparse) | [x] |
| E2E 回归 | MNIST + Transformer/AG News 通过 | [x] |

---

## 断点续传必读文件

1. `docs/architecture/011-sparse-generalization.md`（sparse 泛化设计决策）
2. `docs/architecture/010-systematic-error-analysis.md`（ADR-010：API 设计 + 架构决策）
3. `docs/status/CHANGELOG.md`（已完成任务记录）

---

## 已知测试状态

`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"` → 2,449 passed
4 个预存在失败: test_4bit_sparse_analysis (3), test_adaptive_transform (1)
