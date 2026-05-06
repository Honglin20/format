# Current Task

**Task ID**: Phase 8 — P7 Auto Format Search（未开始）
**Plan**: 待创建
**Branch**: feature/refactor-src

## Progress

### R1 — Architecture Refactor ✅

- [x] 设计文档：四层依赖架构（Math → Ops → Integration → Tools）
- [x] Task 1: 打破 formats↔quantize 循环依赖（`formats/_core.py`）
- [x] Task 2: 提取 observer/ 横切包（`observer/`）
- [x] Task 3: 合并过小文件（cost/device, pipeline/runner, viz/figures）
- [x] Task 4: 创建 session/ 驱动层（吸收 session.py + mapping/ + context/）
- [x] Task 5: 删除死代码（config/, analysis/export.py）
- [x] Task 6: 创建 _utils/ 私有工具
- [x] Task 7: 最终验证（1,468 passed, 无 regression）

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
- [x] PreScaleTransform channel_axis 支持（SmoothQuant 风格广播）
- [x] `initialize_pre_scales` 增强：trainable/non-trainable, per_tensor/per_channel, ones/amax/pot_amax
- [x] `_ACTIVATION_ROLES` / `_INPUT_ACTIVATION_ROLES` + `_replace_transform_activation_only(roles=...)`
- [x] `QuantSession._collect_input_amax` — forward hook 收集 per-module amax
- [x] `_INPUT_ACTIVATION_ROLES` 用于 per-channel（matmul 输入/输出 channel 维度不同，输出角色不可替换）
- [x] Hierarchical study: `part_hierarchical` in STUDY_CONFIG + `_run_hierarchical_part` runner
- [x] 14 PreScale 新测试 + 全量 1416 passed
- [x] Review fixes: multi-batch amax accumulation, _infer_out_channels guard scoped to per_channel, orphaned table7 removed

### P8.R1 — Pipeline Refactor ✅

- [x] `src/pipeline/` + `src/viz/`（52 新测试，分支 `claude/pipeline-refactor` 已合入）

### P8.R2 — Format Study 三层分离 ✅

- [x] `ExperimentResult` dataclass + 简化 `ExperimentRunner`（`runner.py`, 159 行）
- [x] `StudyReport` 声明式输出层（`report.py`, 新建）
- [x] `format_study.py` 纯编排层（1086 → 711 行）
- [x] Config schema 去 `type` 字段，统一为 configs list
- [x] 4 张表格生成器迁移至 `src/viz/tables.py`
- [x] `scale_format` 字段（fp32/pot），per-config 独立设置
- [x] SQ 预处理在编排层（非 Runner 自动检测）
- [x] Per-Layer Optimal 保留在编排层作为组合能力
- [x] 增量保存：`on_config_done` 回调
- [x] `examples/format_study_random.py` — 随机 tensor 验证示例
- [x] 1,467 passed（-1 测试：旧 `skip_calib_when_none` 行为不再支持）

### P6 — Coarse Model 性能估算 ✅

- [x] `src/cost/` 包（defaults, device, op_cost, model_cost, report）
- [x] 39 新测试（test_cost_op_cost, test_cost_report, test_cost_model_cost, test_cost_integration）
- [x] `QuantSession.estimate_cost()` — 无 forward pass，同步返回
- [x] `run_experiment()` 返回 dict 附加 `cost` / `cost_fp32` 键
- [x] 修复计划中 `_elem_bits` 公式错误（`ebits==0` 时取 `mbits`，否则 `ebits+mbits-1`）
- [x] 全量测试：1415 passed（无 regression）

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

`examples/format_study_random.py` 可直接运行验证核心结论。后续用户可替换 `build_model`/`make_calib_data`/`eval_fn` 在自己的模型上验证。

P7/P8/P9 推进顺序待用户选定。

## 最近变更

- 2026-05-06: **Format Study 三层分离完成**。runner（执行）、report（输出）、format_study（编排）三层职责分离。统一 config schema，声明式输出，scale_format per-config。
- 2026-05-06: **架构重构完成**。四层依赖模型：Math (formats/transform/scheme/quantize) → Ops → Integration (session/) → Tools (calibration/analysis/pipeline/cost/viz/onnx)。observer/ 为横切基础设施。删除 config/、mapping/、context/ 三个包。

## 断点续传必读文件

1. `docs/architecture/007-p6-cost-model.md`（P6 Cost Model，可参考作为下个 phase 的模板）
2. `~/.claude/projects/.../memory/format-research-roadmap.md`（优先级全貌）
3. `CLAUDE.md` 架构第一性原理章节（四层依赖模型的权威定义）

## 已知预存在测试失败

`pytest src/tests/` 有 26 个预存在失败（非本分支引入）：
- `test_golden_equiv.py` — 26 tests FileNotFoundError（golden data `.pt` 文件未 staging）
- 排除 golden 测试后全部通过：`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` → 1,467 passed

## 关键经验记录

1. **P5 LSQ 走 Transform 槽位而非 OpQuantConfig 字段**：ADR-006 明确拒绝 `scale_mode` / `learnable` 字段，改为 PreScaleTransform 持有 nn.Parameter 引用 + 外部 LSQ 优化器。这是架构决策，不是缺失。
2. **Pipeline Refactor IoC 模式验证通过**：单回调驱动 calibrate/analyze/evaluate 三阶段，模型交互完全由用户控制。
3. **Module boundary 强制执行**：viz 模块不含 pipeline/session import（AST 静态检查通过）。
4. **Type guards 是 CLAUDE.md §5.1 硬性要求**：每个公共 API 参数的类型守卫必须配一条 pytest.raises + match= 测试。
5. **quantize_model 不替换根模块**：向 QuantSession 传入裸 `nn.Linear` 时，该 Linear 本身是 root（name=""）不会被 quantize_model 替换。测试需用 wrapper 模型。
6. **_elem_bits 公式**：IntFormat/LookupFormat（ebits=0）：取 `mbits`；FPFormat/BFloat16Format（ebits>0）：取 `ebits+mbits-1`（mbits 包含 sign + implicit bit）。
7. **Per-channel PreScaleTransform 不能用于 matmul 输出角色**：`forward(x) = x * s[in_features]` → `y = x' @ W^T` → `inverse(y_q) = y_q / s[in_features]` 但 y 的 shape 是 `(B, out_features)`。s 无法从 matmul 因子化出来。解决方案：per-channel 只替换 `input` / `grad_input`（`_INPUT_ACTIVATION_ROLES`），per-tensor 安全用于所有算子。
8. **Hierarchical = PreScaleTransform + MX PER_BLOCK**：两级 scale（全局 PoT pre-scale + block 共享指数），使用现有框架原语组合而成，无需新概念。
