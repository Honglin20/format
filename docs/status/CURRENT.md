# Current Task

**Task ID**: Phase 4 — 层级误差分析（单配置）
**Design**: `docs/plans/2026-04-26-phase4-design.md`
**Plan**: `docs/plans/2026-04-26-phase4-implementation.md`
**Branch**: `feature/refactor-src`

---

## Phase 3 完成状态

- [x] Phase 3 全部完成（P3.1 ~ P3.6 + 缺陷修复）
- [x] 730 tests passed，bit-exact

---

## Phase 4 子任务进度

### P4.1 — Observer + Context + Report
- [ ] P4.1-1: `DistributionObserver`（含 skew/kurtosis/bimodality/entropy）
- [ ] P4.1-2: `QSNRObserver`
- [ ] P4.1-3: `MSEObserver`
- [ ] P4.1-4: `HistogramObserver`
- [ ] P4.1-5: `AnalysisContext`
- [ ] P4.1-6: `Report`（to_dataframe / print_summary / summary）
- [ ] P4.1-7: 更新 `src/analysis/__init__.py` 公开导出
- [ ] P4.1-8: 端到端集成测试

### P4.2 — 分布画像 + 分布归类 + 误差关联
- [ ] P4.2-1: `DistributionProfile.from_report()`
- [ ] P4.2-2: `DistributionTaxonomy`（8 簇规则归类 + ASCII 直方图 + get_exemplars）
- [ ] P4.2-3: `ErrorByDistribution`（group_by_range + rank_layers）
- [ ] P4.2-4: `LayerSensitivity`（topk + by_layer_type）

### P4.3 — 导出
- [ ] P4.3-1: JSON / CSV 导出
- [ ] P4.3-2: `print_summary()` Markdown 格式化

---

## 设计决策（已确认）

| 决策 | 结论 |
|---|---|
| 分析范围 | Inference only（forward pass） |
| 触发方式 | `with AnalysisContext(model, observers):` 上下文管理器 |
| 多格式对比 | 拆分到 Phase 4.5 |
| 报告交互 | Python API + `print_summary()` 两者都要 |
| 分布归类 | 规则匹配（8 个预定义簇） |
| 可视化 | ASCII 终端直方图 + `get_exemplars()` histogram data export |

---

## 下一步（具体动作）

从 P4.1-1 开始：实现 `DistributionObserver`（TDD 先写测试）。

---

## 断点续传必读文件

1. `docs/plans/2026-04-26-phase4-design.md`（全文）
2. `src/analysis/observer.py`（全文 — SliceAwareObserver 基类）
3. `src/analysis/slicing.py`（全文 — iter_slices）
4. `src/analysis/events.py`（全文 — QuantEvent）
5. `src/analysis/mixin.py`（全文 — ObservableMixin）

---

## 关键经验记录

1. **`emit_fn` 回调模式**：`autograd.Function` 无 self，不能直接调用 `_emit()`。解决方案是将绑定方法 `self._emit` 作为 `emit_fn` 参数传入 Function，无 observers 时传 `None`，靠 `if emit_fn:` 保证零开销。
2. **inner_scheme 到 OpQuantConfig 的映射约定**：Activation/Softmax/Pool 的 `inner_scheme` 对应 `cfg.input[0]`（forward）和 `cfg.grad_input[0]`（backward QAT）。
3. **iter_slices PER_BLOCK block_axis**：应使用通用 index tuple（`[slice(None)] * ndim`，然后在 `axis` 维度替换为范围切片），而非写死 `fp32[..., sl]`。
4. **`quantize_elemwise` 是格式的本质**（2026-04-26 重构）：逐元素量化是所有量化管线的统一必经步骤。自定义格式只需重写此方法，MX 路径自动生效。
5. **`allow_denorm` 在 PER_BLOCK 路径被静默丢弃**：`FormatBase.quantize(x, PER_BLOCK, allow_denorm=False)` 中 `allow_denorm` 不会被传入 `_quantize_per_block` → `_quantize_mx`。与旧 mx/ 代码行为一致（MX 始终 `allow_denorm=True`）。
6. **per_channel 逐元素调用次数**：per-channel 路径先按 per-channel amax 归一化整个张量，然后只调用一次 `quantize_elemwise` —— 不是逐 channel 多次调用。
