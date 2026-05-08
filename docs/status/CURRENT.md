# Current Task

**Task ID**: Phase 8.R3 — Session 统一入口重构 ✅ → Session 2.0 分层 API ✅ → 下一阶段：P7 自动格式搜索
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

**状态**: Phase 8.R3 全部完成。1,708 passed，0 regression。Session 2.0 分层 API 已完成。

### Session 2.0 分层 API（2026-05-07）

- [x] Session 链式 API：`.quantize()` → `.calibrate()` → `.analyze()` → `.evaluate()` → `.cost()` → `.result`
- [x] 所有步骤方法返回 `self`（可链式调用）
- [x] MX per_block 格式 `calibrate()` 自动跳过（scale 动态计算）
- [x] `.qmodel` / `.fp32_model` property（`.quantize()` 后可访问）
- [x] `session(x)` 推理委托（`__call__`）
- [x] `.use_fp32()` / `.use_quant()` / `.mode` 模式切换
- [x] `SessionResult` 访问方法：`.summary()` / `.accuracy_table()` / `.top_k_qsnr(k)` / `.layer_report()`
- [x] `run()` 保持向后兼容（快捷方式）
- [x] 全部 guard：所有方法在 `.quantize()` 之前调用抛出 `RuntimeError("Call .quantize() first")`
- [x] README 更新：三种使用方式（全自动 / 分步链式 / MX 直接推理）
- [x] 76 tests（38 原有 + 38 新增），全量 1,709 passed

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
- 排除 golden 测试后全部通过：`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` → 2,069 passed

## 关键经验记录

1. **P5 LSQ 走 Transform 槽位**：ADR-006 明确拒绝 `scale_mode` / `learnable` 字段，改为 PreScaleTransform 持有 nn.Parameter + 外部 LSQ 优化器。`QuantConfig` 中体现为 `transform="prescale"` + `lsq_steps > 0`，不可引入独立 `lsq` 开关。
2. **Pipeline Refactor IoC 模式验证通过**：单回调驱动 calibrate/analyze/evaluate 三阶段，模型交互完全由用户控制。
3. **Module boundary 强制执行**：viz 模块不含 pipeline/session import（AST 静态检查通过）。
4. **Type guards 是 CLAUDE.md §5.1 硬性要求**：每个公共 API 参数的类型守卫必须配一条 pytest.raises + match= 测试。
5. **quantize_model 不替换根模块**：向 QuantSession 传入裸 `nn.Linear` 时，该 Linear 本身是 root（name=""）不会被 quantize_model 替换。测试需用 wrapper 模型。
6. **_elem_bits 公式**：IntFormat/LookupFormat（ebits=0）：取 `mbits`；FPFormat/BFloat16Format（ebits>0）：取 `ebits+mbits-1`。
7. **Per-channel PreScaleTransform 不能用于 matmul 输出角色**：s 无法从 matmul 因子化出来。解决方案：per-channel 只替换 `_INPUT_ACTIVATION_ROLES`，per-tensor 安全用于所有算子。
8. **Hierarchical = PreScaleTransform + MX PER_BLOCK**：两级 scale（全局 PoT pre-scale + block 共享指数），使用现有框架原语组合而成。

- 2026-05-08: **quantize_backprop 修正 — Transformer 全量 backward bit-exact 验证通过**。根因：`_make_ln/gn/bn/rms_norm` 传入 `quantize_backprop=cfg.is_training` 但 `cfg` 为原始 config（backward fields 均为 None → `is_training=False`），导致 norm Function 的 backward 中 `inner_scheme_bw=None`，所有 vec_ops 以 fp32 执行而非 bf16 量化，与 MX 的逐步 elemwise 量化行为不一致。修正：传入 `_non_matmul_cfg(cfg).is_training`（storage 存在时为 True）。同时修复了 activations/softmax/pooling 模块多余的 pre/post quantization（在 autograd Function 外部调用 `quantize()`, `torch.floor` backward 为 0 导致梯度断裂）。Transformer 验证：`tools/verify_transformer_equiv.py` smoke（6 配置）全部 PASS，包含 bf16/none storage + fp8/int4 compute 的 forward + backward bit-exact 等价。fp8 storage 配置已知不通过（`_build_op_config` 中 fp8_e5m2 格式与 MX 的 fp8 实现不一致，需单独排查）。全量 2,067 passed（+0）。
- 2026-05-08: **E2E 测试矩阵扩展完成**。`test_e2e_all_ops.py` 新增 6 个测试类，覆盖 8 种格式 × 3 种 storage × 2 种 QBP 的完整参数矩阵。测试类：TestE2EAllOpsSmoke（5 configs）、Full（9 configs, slow）、BF10Smoke/Full（4/9 configs）、PureMX（9 configs, slow）、STE（5 configs）、UnifiedCfg（2 tests）、PatternMatch（5 configs）、Backward（5 configs，4 bit-exact + 1 allclose）。全量 2,069 passed（+35）。
- 2026-05-08: **quantize_nonlinear 开关**。`QuantConfig.quantize_nonlinear=False` 使 norm / activation / pool 保持 fp32，仅 Linear / Conv 做量化。`QuantSession` 公开别名加入 `__all__`。README 和 quickstart-details 文档全面更新，与代码一致。全量 2,034 passed。
- 2026-05-07: **Session 内部三层委托关系文档化**。ADR-008 §5.1.1 记录了 `Session` → `_QuantSession` → `quantize_model()` 的分层委托关系和各自的使用场景。
- 2026-05-07: **全算子端到端等价性验证通过**。`tools/verify_layer_equiv.py` 验证了全部 21 种模块类型 + 全部 inline ops 的 bit-exact 等价性。修复了 5 个 bug：`bfloat=16` → `storage_bits/storage_kind` 参数重命名、`model.eval()` 必须在 `quantize_model()` 之后调用、`to_op_config()`/`resolve_config()` 不应设置 `output` field（MX 不施加 per-block 输出量化）、per-module dict 模式需要显式 `op_cfgs` 以配置 inline matmul 系列 op、MX 参考链中 combine-add 的结合律需与模型 forward 一致（左结合）。全量 1,712 passed。
- 2026-05-07: **Session 统一入口重构完成**（Phase 8.R3）。`QuantConfig` 唯一配置入口、`Session`/`Study` 三概念层级、`report/` Output-Driven 输出系统。`src/pipeline/` 已删除，C1-C5 全部修复。全量 1,671 passed。
- 2026-05-07: **Framework review 13 issues resolved** (P0–P2). C1 (unified deserialization), C2 (STUDY_CONFIG fixes), C3 (output compute in to_op_config), I1 (storage_bits/storage_kind rename), I2 (deleted _utils/), I3 (scale_storage naming unified), I4 (prescale_granularity conditional), M1 (_QuantSession private), M3 (w_axis/a_axis fields), L1 (per_layer_optimal reuses Session helpers), L2 (deduplicated _VALID_ROUND_MODES), L3 (study_config.py translated to English). 1,712 passed.
- 2026-05-06: **Format Study 三层分离完成**。runner（执行）、report（输出）、format_study（编排）三层职责分离。
- 2026-05-06: **架构重构完成**。四层依赖模型，observer/ 为横切基础设施。
