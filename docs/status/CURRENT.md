# Current Task

**Task ID**: Phase 8 — P7 Auto Format Search（未开始）
**Plan**: 待创建
**Branch**: feature/refactor-src

## Progress

### P1 — Transform 体系 ✅

- [x] Hadamard rotation (`src/transform/hadamard.py`, 244 行测试)
- [x] SmoothQuant (`src/transform/smooth_quant.py`, 578 行测试)
- [x] PreScale (`src/transform/pre_scale.py`, 198 行测试)

### P2 — Calibration 管线 ✅

- [x] 4 种 ScaleStrategy（max/percentile/MSE/KL）(`src/calibration/strategies.py`)
- [x] CalibrationPipeline (`src/calibration/pipeline.py`, 316 行测试)
- [x] LSQ Optimizer (`src/calibration/lsq_optimizer.py`, 9 测试)

### P3 — NF4 / 查找表格式 ✅

- [x] LookupFormat + NF4Format (`src/formats/lookup_formats.py`, 462 行测试)

### P4 — 参数化格式注册 ✅

- [x] register_float_format / register_int_format / from_str / 自动解析 (`src/formats/registry.py`, 294 行测试)

### P5 — 可学习量化参数 ✅ (ADR-006)

- [x] LayerwiseScaleOptimizer + PreScaleTransform（Transform 槽位方案，非 scale_mode 字段方案）

### P8.R1 — Pipeline Refactor ✅

- [x] `src/pipeline/` + `src/viz/`（52 新测试，分支 `claude/pipeline-refactor` 已合入）

### P6 — Coarse Model 性能估算 ✅

- [x] `src/cost/` 包（defaults, device, op_cost, model_cost, report）
- [x] 39 新测试（test_cost_op_cost, test_cost_report, test_cost_model_cost, test_cost_integration）
- [x] `QuantSession.estimate_cost()` — 无 forward pass，同步返回
- [x] `run_experiment()` 返回 dict 附加 `cost` / `cost_fp32` 键
- [x] 修复计划中 `_elem_bits` 公式错误（`ebits==0` 时取 `mbits`，否则 `ebits+mbits-1`）
- [x] 全量测试：1387 passed（无 regression）

### P1 收尾项（全局最低优先级，P7-P9 完成后再关注）

- [ ] Bias Correction
- [ ] Cross-Layer Equalization (CLE)
- [ ] Transform 组合与注册

### 剩余（P7-P9，未开始）

- [ ] P7 — 自动格式搜索
- [ ] P8 — 融合 Kernel
- [ ] P9 — ONNX custom op（ORT 可推理）

## 待讨论设计决策

> 无活跃决策。P7-P9 推进顺序待用户选定。

## 下一步

从 P7/P8/P9 中选定一个方向，创建实现计划并开始。

## 断点续传必读文件

1. `docs/architecture/007-p6-cost-model.md`（P6 Cost Model 已完成，可参考作为下个 phase 的模板）
2. `~/.claude/projects/.../memory/format-research-roadmap.md`（优先级全貌）
3. `src/cost/` 目录（P6 实现，共 5 个文件）

## 已知预存在测试失败

`pytest src/tests/` 有 26 个预存在失败（非本分支引入）：
- `test_golden_equiv.py` — 26 tests FileNotFoundError（golden data `.pt` 文件未 staging）
- 排除 golden 测试后全部通过：`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` → 1,387 passed

## 关键经验记录

1. **P5 LSQ 走 Transform 槽位而非 OpQuantConfig 字段**：ADR-006 明确拒绝 `scale_mode` / `learnable` 字段，改为 PreScaleTransform 持有 nn.Parameter 引用 + 外部 LSQ 优化器。这是架构决策，不是缺失。
2. **Pipeline Refactor IoC 模式验证通过**：单回调驱动 calibrate/analyze/evaluate 三阶段，模型交互完全由用户控制。
3. **Module boundary 强制执行**：viz 模块不含 pipeline/session import（AST 静态检查通过）。
4. **Type guards 是 CLAUDE.md §5.1 硬性要求**：每个公共 API 参数的类型守卫必须配一条 pytest.raises + match= 测试。
5. **quantize_model 不替换根模块**：向 QuantSession 传入裸 `nn.Linear` 时，该 Linear 本身是 root（name=""）不会被 quantize_model 替换。测试需用 wrapper 模型。
6. **_elem_bits 公式**：IntFormat/LookupFormat（ebits=0）：取 `mbits`；FPFormat/BFloat16Format（ebits>0）：取 `ebits+mbits-1`（mbits 包含 sign + implicit bit）。
