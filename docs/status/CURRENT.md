# Current Task

**Task ID**: Phase 4 + 4.5 — 层级误差分析 + 多格式对比（已完成）
**Design**: `docs/plans/2026-04-26-phase4-design.md`
**Plan**: `docs/plans/2026-04-26-phase4-implementation.md`
**Branch**: `feature/refactor-src`

---

## Phase 4 完成状态

### P4.1 — Observer + Context + Report
- [x] P4.1-1: `DistributionObserver`（含 skew/kurtosis/bimodality/entropy）
- [x] P4.1-2: `QSNRObserver`
- [x] P4.1-3: `MSEObserver`
- [x] P4.1-4: `HistogramObserver`
- [x] P4.1-5: `AnalysisContext`
- [x] P4.1-6: `Report`（to_dataframe / print_summary / summary / to_json / to_csv）
- [x] P4.1-7: 更新 `src/analysis/__init__.py` 公开导出
- [x] P4.1-8: 端到端集成测试

### P4.2 — 分布画像 + 分布归类 + 误差关联
- [x] P4.2-1: `DistributionProfile.from_report()`
- [x] P4.2-2: `DistributionTaxonomy`（8 簇规则归类 + get_exemplars + print_taxonomy）
- [x] P4.2-3: `ErrorByDistribution`（group_by_range + rank_layers + print_correlation）
- [x] P4.2-4: `LayerSensitivity`（topk + by_layer_type + above_threshold）

### P4.3 — 导出
- [x] P4.3-1: JSON / CSV 导出（Report.to_json / Report.to_csv）
- [x] P4.3-2: `print_summary()` Markdown 格式化

### P4.5 — 多格式对比
- [x] P4.5-1: `compare_formats()` + `ComparisonReport`
- [x] P4.5-2: `summary()`, `rank_formats()`, `recommend()`, `print_comparison()`
- [x] P4.5-3: 7 个 compare tests

**总计：40 个新 analysis tests + 951 tests passed（零 regression）**

---

## 新增文件清单

```
src/analysis/
├── observers.py       # DistributionObserver, QSNRObserver, MSEObserver, HistogramObserver
├── context.py         # AnalysisContext
├── report.py          # Report (to_dataframe / summary / print_summary / to_json / to_csv)
├── correlation.py     # DistributionProfile, DistributionTaxonomy, ErrorByDistribution, LayerSensitivity
└── export.py          # Export placeholder

src/tests/
└── test_analysis.py   # 33 tests (unit + integration + E2E)
```

---

## 下一步（具体动作）

所有分析模块已完整交付（Phase 4 + 4.5）。可进入 Phase 5（ONNX Export）或按需扩展新格式。

---

## 断点续传必读文件

1. `src/analysis/observers.py`（全文）
2. `src/analysis/context.py`（全文）
3. `src/analysis/report.py`（全文）
4. `src/analysis/correlation.py`（全文）
5. `docs/plans/2026-04-26-phase4-design.md`（#11 Phase 4.5 展望）

---

## 关键经验记录

1. **Multi-observer deep merge**：`AnalysisContext.report()` 合并多个 observer 数据时，必须用深层合并（`setdefault` 到 slice key 级别再 `update`），否则同层不同 observer 的指标互相覆盖。
2. **nn.Sequential 命名问题**：`nn.Sequential` 用数字作为模块名，会覆盖 `QuantizedXxx(name=...)` 的 `_analysis_name`。E2E 测试中使用自定义 `nn.Module` 子类保留有意义的属性名。
3. **DistributionTaxonomy 规则顺序**：先匹配双峰（bimodality_coefficient > 0.555），再匹配偏态。因为双峰分布可能同时满足偏态条件（如 bimodal+positive-skewed），先匹配双峰避免错误归类。
