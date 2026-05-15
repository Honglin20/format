# Current Task

**当前任务**: 空闲 — ADR-013 Group Sparse 已全部完成，等待下一个任务
**Branch**: `feature/refactor-src`

---

## 背景

ADR-012 和 ADR-013 全部阶段已完成。

### 完成状态

| Phase | 内容 | 状态 |
|-------|------|------|
| P1 | BANK 粒度 — GranularityMode.BANK + _quantize_per_bank + calibration | ✅ review 通过 |
| P2 | compute_sparse_mask() — per-sample top-k + cross-sample voting | ✅ review 通过 |
| P3 | Sparse 静态量化 — 全部 granularity mode 接入 static mask 路径 | ✅ review 通过 |
| P4 | 可配置 outlier_format — QuantScheme/QuantConfig | ✅ review 通过 |

### ADR-013 完成状态

| Phase | 内容 | 状态 |
|-------|------|------|
| P1-P6 | Group Sparse 全阶段（详见 CHANGELOG） | ✅ 完成 |

---

## 断点续传必读文件

1. `docs/architecture/013-group-sparse-format-assignment.md` — ADR-013 设计
2. `src/formats/base.py` — FormatBase dispatch + static sparse + group_sparse
3. `src/quantize/_group_mask.py` — compute_group_mask
4. `src/scheme/quant_scheme.py` — group_format/group_ratio 字段
5. `src/calibration/pipeline.py` — CalibrationSession 集成

---

## 已知测试状态

`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"` → 2,590 passed, 42 failed

42 个预存在失败: NF4 SIMD/Pool/Norm/Softmax/Activation equiv tests (~38) + test_4bit_sparse_analysis (3) + test_adaptive_transform (1)
— 均与 ADR-012/ADR-013 变更无关。
