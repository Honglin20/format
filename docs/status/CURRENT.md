# Current Task

**当前任务**: ADR-012 全部完成 — code review 通过
**下一任务**: 处理 42 个预存在测试失败（NF4 equiv ~38 + 4bit_sparse 3 + adaptive_transform 1）
**Branch**: `feature/refactor-src`

---

## 背景

ADR-012 四个能力缺口已全部实现。分四阶段开发，每阶段独立 review。

### 完成状态

| Phase | 内容 | 状态 |
|-------|------|------|
| P1 | BANK 粒度 — GranularityMode.BANK + _quantize_per_bank + calibration | ✅ review 通过 |
| P2 | compute_sparse_mask() — per-sample top-k + cross-sample voting | ✅ review 通过 |
| P3 | Sparse 静态量化 — 全部 granularity mode 接入 static mask 路径 | ✅ review 通过 |
| P4 | 可配置 outlier_format — QuantScheme/QuantConfig | ✅ review 通过 |

### Review 修复 (commit 14d8a54)

- **_mask_per_block group size**: block_size → 完整 block tile 元素数
- **PER_BLOCK static sparse**: raise NotImplementedError（非静默错误结果）
- **BANK amax reshape**: 元素数验证 + axis bounds check
- **Calibration**: 非整除维度 warning + axis bounds check

---

## 断点续传必读文件

1. `docs/architecture/012-bank-sparse-static-outlier-format.md`（ADR-012：全量设计决策）
2. `src/formats/base.py`（FormatBase — dispatch + static sparse + outlier_format）
3. `src/quantize/_sparse_mask.py`（compute_sparse_mask + per-mode mask helpers）
4. `src/session/_config.py`（QuantConfig — outlier_format/a_outlier_format 字段）
5. `src/tests/test_static_sparse.py`（27 tests: static sparse + outlier_format）

---

## 已知测试状态

`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"` → 2,499 passed, 42 failed

42 个预存在失败: NF4 SIMD/Pool/Norm/Softmax/Activation equiv tests (~38) + test_4bit_sparse_analysis (3) + test_adaptive_transform (1)
— 均与 ADR-012 变更无关，在 f6eb5cd baseline 已存在。
