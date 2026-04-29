# Current Task

**Task ID**: Phase 8 — P6 Coarse Model
**Plan**: `docs/plans/YYYY-MM-DD-p6-cost-model.md`（待创建）
**Design Ref**: `docs/architecture/007-p6-cost-model.md`（公式权威来源）
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

### P1 收尾项（全局最低优先级，P6-P9 完成后再关注）

- [ ] Bias Correction
- [ ] Cross-Layer Equalization (CLE)
- [ ] Transform 组合与注册

### 剩余（P6-P9，未开始）

- [ ] P6 — Coarse Model 性能估算
- [ ] P7 — 自动格式搜索
- [ ] P8 — 融合 Kernel
- [ ] P9 — ONNX custom op（ORT 可推理）

## 待讨论设计决策

> 无活跃决策。P6-P9 推进顺序待用户选定。

## 下一步

从 P6/P7/P8/P9 中选定一个方向，创建实现计划并开始。

## 断点续传必读文件

1. `docs/architecture/007-p6-cost-model.md`（P6 Cost Model 公式 + 架构，权威来源）
2. `~/.claude/projects/.../memory/format-research-roadmap.md`（优先级全貌）
3. `src/session.py`（1-372 行，了解 session 架构以集成 cost 方法）
4. `src/ops/` 目录下算子文件（forward 量化步骤数需从此提取）

## 已知预存在测试失败

`pytest src/tests/` 有 26 个预存在失败（非本分支引入）：
- `test_golden_equiv.py` — 26 tests FileNotFoundError（golden data `.pt` 文件未 staging）
- 排除 golden 测试后全部通过：`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` → 1,348 passed

## 关键经验记录

1. **P5 LSQ 走 Transform 槽位而非 OpQuantConfig 字段**：ADR-006 明确拒绝 `scale_mode` / `learnable` 字段，改为 PreScaleTransform 持有 nn.Parameter 引用 + 外部 LSQ 优化器。这是架构决策，不是缺失。
2. **Pipeline Refactor IoC 模式验证通过**：单回调驱动 calibrate/analyze/evaluate 三阶段，模型交互完全由用户控制。
3. **Module boundary 强制执行**：viz 模块不含 pipeline/session import（AST 静态检查通过）。
4. **Type guards 是 CLAUDE.md §5.1 硬性要求**：每个公共 API 参数的类型守卫必须配一条 pytest.raises + match= 测试。
