# Current Task

**Task ID**: Phase 8.R3 — Session 统一入口重构 ✅ → 下一阶段：P7 自动格式搜索
**Plan**: `docs/plans/2026-05-07-session-refactor.md`
**Design**: `docs/architecture/008-session-refactor.md`
**Branch**: `feature/refactor-src`

## Progress

### R1 — Architecture Refactor ✅

- [x] 设计文档：四层依赖架构（Math → Ops → Integration → Tools）
- [x] Task 1: 打破 formats↔quantize 循环依赖（`formats/_core.py`）
- [x] Task 2: 提取 observer/ 横切包
- [x] Task 3: 合并过小文件（cost/device, pipeline/runner, viz/figures）
- [x] Task 4: 创建 session/ 驱动层（吸收 session.py + mapping/ + context/）
- [x] Task 5: 删除死代码（config/, analysis/export.py）
- [x] Task 6: 创建 _utils/ 私有工具
- [x] Task 7: 最终验证（1,468 passed, 无 regression）

### P1 — Transform 体系 ✅

- [x] Hadamard rotation
- [x] SmoothQuant
- [x] PreScale

### P2 — Calibration 管线 ✅

- [x] 4 种 ScaleStrategy（max/percentile/MSE/KL）
- [x] CalibrationPipeline + CalibrationSession
- [x] LSQ Optimizer

### P3 — NF4 / 查找表格式 ✅

### P4 — 参数化格式注册 ✅

### P5 — 可学习量化参数 ✅ (ADR-006)

- [x] LayerwiseScaleOptimizer + PreScaleTransform（Transform 槽位方案）
- [x] PreScaleTransform channel_axis 支持
- [x] `initialize_pre_scales` 增强
- [x] Hierarchical study: `part_hierarchical`
- [x] 全量 1494 passed

### P8.R1 — Pipeline Refactor ✅

### P8.R2 — Format Study 三层分离 ✅

- [x] ExperimentResult dataclass + 简化 ExperimentRunner
- [x] StudyReport 声明式输出层
- [x] format_study.py 纯编排层
- [x] scale_format 字段 + act_format wXaY mixed-precision
- [x] 1,494 passed

### P6 — Coarse Model 性能估算 ✅

### P8.R3 — Session 统一入口重构 ✅

- [x] 设计文档 ADR-008（`docs/architecture/008-session-refactor.md`，review 修正版）
- [x] 实施计划（`docs/plans/2026-05-07-session-refactor.md`，review 修正版）
- [x] Review 审视（`docs/reviews/2026-05-07-adr008-review.md`，12 个问题全部修正入设计）
- [x] Task 1: QuantConfig dataclass（`src/session/_config.py`，32 → 48 tests）
- [x] Task 2: SmoothQuant helpers 迁移（`src/transform/smooth_quant.py`，12 → 19 tests）
- [x] Task 3: report/ 包 — Output-Driven 输出系统（6 文件，54 tests）
- [x] Task 4: Session 执行单元（`src/session/_session.py`，38 tests）
- [x] Task 5: Study 聚合层 + per_layer_optimal 工具（`_study.py` + `_per_layer_opt.py`，20 tests）
- [x] Task 6: 清理 pipeline/ + C1-C5 修复 + 兼容（删除 `src/pipeline/`，更新所有 import）
- [x] Task 7: 文档更新（CLAUDE.md + CURRENT.md）

**状态**: Phase 8.R3 全部完成。1671 passed，0 regression。`src/pipeline/` 已删除。

### 剩余（P7-P9，未开始）

- [ ] P7 — 自动格式搜索
- [ ] P8 — 融合 Kernel
- [ ] P9 — ONNX custom op（ORT 可推理）

### P1 收尾项（全局最低优先级）

- [ ] Bias Correction
- [ ] Cross-Layer Equalization (CLE)
- [ ] Transform 组合与注册

## 断点续传必读文件

1. `docs/architecture/008-session-refactor.md`（ADR-008：Session 统一入口设计，已实施）
2. `docs/architecture/005-op-quant-config.md`（OpQuantConfig 两阶段模型）
3. `docs/workflow/phase-plan.md`（整体进度，下一阶段：P7 自动格式搜索）
4. `CLAUDE.md`（架构层级 + 新用户 API）

## 已知预存在测试失败

`pytest src/tests/` 有 26 个预存在失败（非本分支引入）：
- `test_golden_equiv.py` — 26 tests FileNotFoundError（golden data `.pt` 文件未 staging）
- 排除 golden 测试后全部通过：`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` → 1,671 passed

## 关键经验记录

1. **P5 LSQ 走 Transform 槽位**：ADR-006 明确拒绝 `scale_mode` / `learnable` 字段，改为 PreScaleTransform 持有 nn.Parameter + 外部 LSQ 优化器。`QuantConfig` 中体现为 `transform="prescale"` + `lsq_steps > 0`，不可引入独立 `lsq` 开关。
2. **Pipeline Refactor IoC 模式验证通过**：单回调驱动 calibrate/analyze/evaluate 三阶段，模型交互完全由用户控制。
3. **Module boundary 强制执行**：viz 模块不含 pipeline/session import（AST 静态检查通过）。
4. **Type guards 是 CLAUDE.md §5.1 硬性要求**：每个公共 API 参数的类型守卫必须配一条 pytest.raises + match= 测试。
5. **quantize_model 不替换根模块**：向 QuantSession 传入裸 `nn.Linear` 时，该 Linear 本身是 root（name=""）不会被 quantize_model 替换。测试需用 wrapper 模型。
6. **_elem_bits 公式**：IntFormat/LookupFormat（ebits=0）：取 `mbits`；FPFormat/BFloat16Format（ebits>0）：取 `ebits+mbits-1`。
7. **Per-channel PreScaleTransform 不能用于 matmul 输出角色**：s 无法从 matmul 因子化出来。解决方案：per-channel 只替换 `_INPUT_ACTIVATION_ROLES`，per-tensor 安全用于所有算子。
8. **Hierarchical = PreScaleTransform + MX PER_BLOCK**：两级 scale（全局 PoT pre-scale + block 共享指数），使用现有框架原语组合而成。

## 最近变更

- 2026-05-07: **Session 统一入口重构完成**（Phase 8.R3）。`QuantConfig` 唯一配置入口、`Session`/`Study` 三概念层级、`report/` Output-Driven 输出系统。`src/pipeline/` 已删除，C1-C5 全部修复。全量 1,671 passed。
- 2026-05-06: **Format Study 三层分离完成**。runner（执行）、report（输出）、format_study（编排）三层职责分离。
- 2026-05-06: **架构重构完成**。四层依赖模型，observer/ 为横切基础设施。
