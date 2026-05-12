# Current Task

**当前任务**: ADR-009 收尾 + P7 自动格式搜索
**下一任务**: Calibration 增强 + QAT 验证
**Branch**: `feature/refactor-src`

---

## ADR-010 实施 — 已完成 (2026-05-13)

| Phase | 内容 | 状态 |
|-------|------|------|
| P1 | 多 role QSNR 提取 | [x] |
| P2 | ErrorProvenance + SessionPlotAccessor | [x] |
| P3 | DistributionDiagnosis + 规则引擎 | [x] |
| P4 | InterventionPlanner + overrides + 对比 | [x] |

E2E 回归通过 (MNIST + Transformer/AG News)。

---

## 断点续传必读文件

1. `docs/architecture/010-systematic-error-analysis.md`（ADR-010：API 设计 + 架构决策）
2. `docs/architecture/010-plan.md`（实施计划）
3. `docs/status/CHANGELOG.md`（已完成任务记录）

---

## 已知测试状态

`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"` → 2,420 passed
5 个预存在失败: test_4bit_sparse_analysis (3), test_adaptive_transform (1), test_print_summary_empty (1)
