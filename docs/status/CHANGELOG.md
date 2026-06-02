# Changelog — 已完成任务归档

> 本文件记录已完成的 Phase、子任务和关键 bug 修复。
> 当前进行中的任务见 `CURRENT.md`。

---

## GPTQ 幂等修复 ✅ (2026-05-17)

**问题：** GPTQ 量化不幂等 — GPTQ 用 `_precompute_scale()` 从原始 FP32 权重算出 per-channel amax 并写入 FP32 值到 `module.weight.data`，但 forward pass 中 `quantize(w, cfg.weight)` 不传 `scale`，重新从修改后权重算 amax，导致 re-quant 产出不同结果。GPTQ 不仅不提升精度，反而降低（int4-pc 从 -0.0050 降至 -0.0054）。

**方案：** GPTQ 量化后注册 `_weight_scale` buffer，forward 读 buffer 而非重算 amax。

**改动文件：**

| 文件 | 变更 |
|------|------|
| `src/ops/_calib_buffers.py` | 新增 `weight_scale: Optional[torch.Tensor] = None` 字段 |
| `src/calibration/gptq_optimizer.py` | `_gptq_quantize` 返回 `(W_q, full_scale)`；`optimize()` 注册 `_weight_scale` buffer |
| `src/ops/linear.py` | `QuantizedLinear.forward()` 读取 `_weight_scale`；`LinearFunction.forward()` 传 `scale=buffers.weight_scale` |
| `src/ops/conv.py` | 6 个 QuantizedConv 类 + 2 个 Function 类同上 |
| `src/tests/test_gptq_optimizer.py` | 4 个新测试：buffer 存在、幂等 re-quant、forward buffer check、forward 幂等 |

**E2E 结果（MNIST MLP, FP32=0.9789）：**

| Config | 修复前 Δ | 修复后 Δ | GPTQ gain (修复前) | GPTQ gain (修复后) |
|--------|---------|---------|-------------------|-------------------|
| int4-pc | -0.0054 | -0.0046 | -0.0004 | **+0.0004** |
| int4-pc + sparse | -0.0011 | -0.0011 | -0.0008 | -0.0008 |
| int4-pc + gsparse | -0.0039 | -0.0034 | -0.0005 | -0.0018 |

int4-pc baseline 上 GPTQ 从负增益变为正增益。QSNR 全面提升（13.3→17.1 dB）。

**遗留问题：** GPTQ + sparse/group_sparse 仍有轻微负增益。根因：GPTQ 内部 `quantize()` 走 sparse 路径时，Hessian 补偿假设每列量化精度相同，但 sparse 打破了该假设。待讨论方案：GPTQ 用纯 int4 scheme 做 Hessian 补偿，sparse 在 forward 独立处理。详见 CURRENT.md。

**Commit 历史：**

| Commit | 内容 |
|--------|------|
| `7bde2c3` | feat(buffers): add weight_scale field to CalibrationBuffers |
| `2ee78aa` | test(gptq): add failing tests for _weight_scale buffer and idempotency |
| `3b289ae` | feat(gptq): register _weight_scale buffer for idempotent re-quantization |
| `ecf7af3` | test(gptq): add forward-pass weight_scale integration test |
| `052f430` | feat(linear): read _weight_scale buffer in forward pass |
| `e43917d` | feat(conv): read _weight_scale buffer in forward pass |

**设计文档**: `docs/plans/2026-05-17-gptq-idempotent-design.md`
**实现计划**: `docs/plans/2026-05-17-gptq-idempotent-plan.md`
**测试**: 28 GPTQ tests passed, 全量 2,627 passed, 40 预存在失败

---

## ADR-013: Group Sparse 按粒度组格式分配 ✅ (2026-05-15)

**六阶段实现，P1-P6 全部完成:**

| Phase | 内容 | 关键文件 | 测试 |
|-------|------|---------|------|
| P1 | QuantScheme 新增 group_format + group_ratio + 互斥验证 | `quant_scheme.py` | 12 tests |
| P2 | compute_group_mask() 独立函数 | `quantize/_group_mask.py` — per-group amax → cross-sample max → top-k | 19 tests |
| P3 | FormatBase 各 mode group_sparse 量化方法（动态路径） | `base.py` — 4 个 _quantize_*_group_sparse 方法 | 11 tests |
| P4 | CalibrationSession 集成 | `pipeline.py` — _compute_and_assign_group_sparse_state() | 7 tests |
| P5 | FormatBase 静态路径（pre-computed group_mask） | `base.py` — group_mask 参数 threading + static mask in all modes | 6 tests |
| P6 | QuantConfig 用户面字段 + to_op_config() + from_descriptor() | `_config.py` — 4 个新字段 + validation + conversion | 25 tests |

**设计文档**: `docs/architecture/013-group-sparse-format-assignment.md`

**核心架构决策:**
- Group sparse mask 为 per-granularity-group（非 per-element），与 ADR-012 正交并存
- `group_format` + `group_ratio` 放在 QuantScheme（非 GranularitySpec），与 outlier_format 互斥
- 动态路径：per-group amax → top-k groups → torch.where(H/L)
- 静态路径：CalibrationSession 收集样本 → compute_group_mask → _output_group_mask buffer → 推理期读取
- PER_BLOCK group_sparse: 两个 shared-exponent 副本（H 用 group_format.emax offset, L 用 self.emax）
- Float 格式（ebits > 0）委托标准路径，不作 group split
- QuantConfig 沿用 a_format 的 override 模式：a_group_format / a_group_ratio 覆盖激活

**Code Review 修复:**
- C1: QuantScheme 新增 ratio-level 互斥验证（group_ratio > 0 且 outlier_ratio > 0 抛异常）
- I1: _group_sparse_config 清理未使用的 group_fmt 返回值
- I2: 新增 PER_BLOCK static path 测试覆盖
- I3: 文档化 PER_BLOCK float-format 处理差异（MX 规范）

**全量测试**: 2,590 passed, 42 预存在失败 | **测试文件**: `src/tests/test_group_sparse.py` (85 tests)

**四阶段实现，每阶段独立 review:**

| Phase | 内容 | 关键文件 | 测试 |
|-------|------|---------|------|
| P1 | BANK 粒度 | `granularity.py`, `base.py` (_quantize_per_bank), `_config.py` (QuantConfig bank), `pipeline.py` (_compute_bank_amax) | 27 tests |
| P2 | compute_sparse_mask() | `quantize/_sparse_mask.py` — per-sample group top-k + cross-sample voting | 13 tests |
| P3 | 静态 sparse 全 mode | `base.py` dispatch: mask!=None → static path; scale-without-mask → dynamic fallthrough | 27 tests |
| P4 | 可配置 outlier_format | `quant_scheme.py` (outlier_format field), `_config.py` (outlier_format/a_outlier_format), `base.py` all sparse methods accept outlier_format | 17 tests |

**设计文档**: `docs/architecture/012-bank-sparse-static-outlier-format.md`
**验证文档**: `docs/verification/020-bank-granularity.md`, `docs/verification/021-sparse-mask-voting.md`

**核心架构决策:**
- BANK 为独立 `GranularityMode`，非 PER_BLOCK 特例（scale 类型和 reduction 语义不同）
- `compute_sparse_mask()` 与 FormatBase 量化 dispatch 解耦，单一职责
- Static sparse: 校准期 pre-compute mask + scales，推理期无 topk 排序开销
- `outlier_format` 放在 QuantScheme（三轴之一），沿用 `a_format` 的 override 模式
- Scale-without-mask 优雅 fallthrough 到 dynamic sparse（不抛异常）

**Review 修复 (14d8a54):**
- `_mask_per_block` group size: block_size → 完整 block tile 元素数
- PER_BLOCK static sparse: raise NotImplementedError（非静默错误）
- BANK amax reshape: 元素数验证 + axis bounds check
- Calibration: 非整除维度 warning

**Commit 历史:**
| Commit | 内容 |
|--------|------|
| `40494bc` | GranularityMode.BANK + bank_size/bank_axis |
| `1c6e88a` | _quantize_per_bank + BANK dispatch |
| `a0bc748` | QuantConfig bank support |
| `33bf717` | Session integration test |
| `3ad0b02` | E2E regression verified |
| `f6eb5cd` | compute_sparse_mask + per-mode helpers |
| `927d8c0` | Static sparse paths + outlier_format (P3+P4) |
| `7b7ea11` | Revert output=a_scheme (separate concern) |
| `14d8a54` | Code review fixes |

**全量测试**: 2,499 passed | **E2E 回归**: MNIST + Transformer 通过

---

## Sparse (outlier_ratio) 泛化 ✅ (2026-05-13)

**实现内容**: 将 `outlier_ratio` 从仅 PER_BLOCK 扩展到所有 granularity 模式。

| 文件 | 变更 |
|------|------|
| `docs/architecture/011-sparse-generalization.md` | 设计决策：不开第四轴，保留在 GranularitySpec |
| `docs/verification/018-sparse-per-tensor.md` | per_tensor + sparse 数学推导 + 期望值 |
| `docs/verification/019-sparse-per-channel.md` | per_channel + sparse 数学推导 + 期望值 |
| `src/scheme/granularity.py` | 解除 `outlier_ratio > 0` 的 PER_BLOCK 专属校验 |
| `src/formats/base.py` | 新增 `_quantize_per_tensor_sparse`、`_quantize_per_channel_sparse` |
| `src/session/_config.py` | `QuantConfig` 加 `outlier_ratio` 字段，透传至 GranularitySpec |
| `src/tests/test_sparse_generalization.py` | 28 个测试用例（构造/bitexact/形状/Session集成） |
| `src/tests/test_granularity_outlier_bank.py` | 更新旧测试反映新行为 |

**TDD**: RED(23 fail) → GREEN(28 pass) | **全量**: 2449 passed | **E2E**: MNIST + Transformer 回归通过

---

## ADR-010: 系统化误差分析闭环 ✅ (2026-05-13)

**实现内容 (4 阶段, 9 新文件, 3 修改文件):**

| Phase | 内容 | 文件 |
|-------|------|------|
| P1 | 多 role QSNR/MSE 提取 (`_extract_all_roles_qsnr_mse`) | `_session.py`, `_result.py` |
| P2 | ErrorProvenance + per-role 绘图 + error source tables | `_error_provenance.py`, `_per_role.py`, `_propagation.py`, `_plot.py`, `_session_tables.py` |
| P3 | DistributionDiagnosis + 6-规则退化分类引擎 | `_distribution_diagnosis.py` |
| P4 | InterventionPlan/Planner + Session overrides + InterventionAccessor/Comparison | `_intervention.py`, `_intervention_accessor.py`, `_session.py`, `_result.py` |

**核心 API (全部通过 `SessionResult` 属性访问):**
- `result.diagnose` → `ErrorProvenance` — per-role per-layer QSNR, top-K, error source analysis
- `result.characterize` → `DistributionDiagnosis` — 分布退化分类 + 因果分析
- `result.plan` → `InterventionPlanner` — top_k_boost / recommend / transform_ranking
- `result.intervention` → `InterventionAccessor` — compare(基线 vs 干预)
- `result.plot` → `SessionPlotAccessor` — 12+ 图表方法 (qsnr_comparison, error_propagation, per_role bars, histograms, channel_heterogeneity 等)

**E2E 回归**: MNIST + Transformer/AG News 通过所有合理性判据。
**测试**: 2,420 passed, 5 预存在失败 (无回归)。

---

## Report 接口统一与 QSNR 类型开关 ✅ (2026-05-11)

**问题：**
- `StudyReport.print_summary()` 逐 part 打印分隔表格，不是 DataFrame，不显示 FP32 基线
- 接口与 `SessionResult.summary()` 不统一
- 所有 summary/table 方法硬编码使用 local QSNR，无法查看端到端 accumulated QSNR

**方案：**
- 新增 `StudyReport.summary_dataframe()` — 所有 part 的所有 config 统一为一个 DataFrame，含 `fp32_*` / `quant_*` / `delta_*` / `avg_qsnr_db` / `avg_mse` 列
- 重写 `print_summary()` — 使用 DataFrame 输出，含 FP32 基线，无 pandas 时优雅降级
- 6 个核心入口方法新增 `qsnr_type="local"` 参数（`"local"` | `"accum"`）：
  - `SessionResult.summary()` / `top_k_qsnr()`
  - `StudyReport.print_summary()` / `summary_dataframe()`
  - `SessionTablesAccessor.per_layer_qsnr()` / `StudyTablesAccessor.per_layer_qsnr()`
- `_avg_qsnr_mse` 私有 helper 支持开关，所有 StudyReport 方法统一透传

**改动文件：**
- `src/report/_study_report.py` — 新增 `summary_dataframe()`，重写 `print_summary()`，更新 `_avg_qsnr_mse`
- `src/session/_result.py` — `summary()` / `top_k_qsnr()` 加 `qsnr_type` 参数
- `src/report/_session_tables.py` — `per_layer_qsnr()` 加 `qsnr_type`
- `src/report/_tables.py` — `per_layer_qsnr()` 加 `qsnr_type`
- `src/tests/test_study_report.py` — 7 个新测试 + 更新已有测试
- `src/tests/test_session_unit.py` — 4 个新 accum QSNR 测试

测试：全量 2,399 passed（fast）

---

## Bug Fix — True Error 累积误差分析修复 ✅ (2026-05-10)

修复 `Session.analyze(true_error=True)` 两个 bug 并简化实现：

**问题：**
- `_is_passthrough` 只在 `QuantizedLinear` 设置，其他 21 个 Quantized* 类型空 cfg 模块被错误纳入对比，产出 QSNR = ∞
- 多 batch 只保留最后一个，之前 batch 的 hook 捕获值被覆盖

**方案：**
- fp32 参考用现有原始 nn.Module deep copy（`fp32_model`），天然 golden reference，零验证成本
- forward hook 捕获模块 output（累积误差天然定义在模块边界）
- 新增 `cfg_causes_quantization(cfg)` 在 `src/scheme/op_config.py`，遍历所有 dataclass 字段判断是否真的会触发量化
- 多 batch 逐 batch 累加 `Σsignal / Σerror / count`，最终 `QSNR = 10 * log10(mean_signal / mean_error)`
- `eval_fn` 作为第一优先级，所有 forward pass 优先走 eval_fn
- 可同时与 observer 共存（一个 AnalysisContext 包裹整个循环）

**与原方案对比：**
- 放弃双 Quantized* passthrough 树（需改 22 个模块 + bit-exact 验证）
- 放弃统一 stash 机制（input QSNR 与上一层 output 冗余，weight QSNR 是局部的可直接公式算）
- 放弃 Phase 3 内联 op stash（observer 已覆盖局部误差）
- ops/ 层零改动，只改 session 层

**改动文件：**
- `src/scheme/op_config.py` — 新增 `cfg_causes_quantization()`
- `src/session/_session.py` — 重写 `analyze()` true-error 路径 + `quantize()`/`run()` 透传 eval_fn
- `src/session/_model.py` — `_get_quantized_modules` 用 `cfg_causes_quantization`
- `src/tests/test_session.py` — 7 个新测试
- `docs/plans/2026-05-10-unified-stash-true-error.md` — 更新为简化方案文档

测试：全量 2,496 passed（含 7 个新增）

---

## Phase 9 — 学术量化研究可视化扩展

### P9.V1 — 8 个新增研究图表/表格 ✅ (2026-05-09)

基于 observer 数据采集能力缺口分析，补齐学术量化研究中 8 个关键可视化：

**P0（关键）：**
- **P0.1 Outlier 分析** — 逐层 outlier ratio 柱状图 + outlier vs QSNR 散点图
- **P0.2 逐 Block QSNR 分布** — 逐层 QSNR 标准差箱线图 + min-vs-mean 散点
- **P0.3 分布拟合分类表** — best_fit 分布类型计数表（DistributionFitObserver）
- **P0.4 Pareto 前沿** — QSNR/Accuracy vs bit-width/latency/memory 多面板散点

**P1（重要）：**
- **P1.5 分布特征 × QSNR 相关性热力图** — 8 种分布特征与 QSNR/MSE 的 Pearson 相关系数矩阵
- **P1.6 成本分解图** — 每个 config 的 math/quantize/transform FLOPs 堆叠柱状图
- **P1.7 跨角色分布对比图** — input vs weight vs output 的 skewness/kurtosis/entropy 箱线图
- **P1.8 Transform 逐层收益表** — 每层 baseline QSNR + 各 transform QSNR + delta

**实现细节：**
- `src/report/_plot.py` — 6 个新 StudyPlotAccessor 方法（195→370 行）
- `src/viz/figures.py` — 4 个新 standalone 图表函数（`outlier_analysis`、`per_block_qsnr`、`correlation_heatmap`、`role_distribution_comparison`）
- `src/viz/tables.py` — 2 个新表格函数（`distribution_fit_table`、`transform_benefit_table`）
- `src/report/_registry.py` — 8 个新注册 entry + `src/report/_spec.py` — 6 个新 output spec
- `src/report/_study_report.py:save()` — 自动生成所有新图表（try/except 优雅降级）
- 所有函数遵循 `ValueError` + 可操作提示的错误处理规范
- 41 个新测试，全量 2,276 passed

---

## Phase 8 — 研究能力扩展

### P8.R3 — Session 统一入口重构 ✅ (2026-05-07)

- QuantConfig dataclass（`src/session/_config.py`，48 tests）
- SmoothQuant helpers 迁移（`src/transform/smooth_quant.py`，19 tests）
- report/ 包 — Output-Driven 输出系统（6 文件，54 tests）
- Session 执行单元（`src/session/_session.py`，38 tests）
- Study 聚合层 + per_layer_optimal 工具（20 tests）
- 清理 pipeline/ + C1-C5 修复 + 兼容
- Session 2.0 分层 API：`.quantize()` → `.calibrate()` → `.analyze()` → `.evaluate()` → `.cost()` → `.result`
- 全量 2,069 passed

详见 ADR-008（`docs/architecture/008-session-refactor.md`）

### P8.R2 — Format Study 三层分离 ✅ (2026-05-06)

- ExperimentResult dataclass + 简化 ExperimentRunner
- StudyReport 声明式输出层
- format_study.py 纯编排层
- scale_format 字段 + act_format wXaY mixed-precision

### P8.R1 — Pipeline Refactor ✅

IoC 模式：单回调驱动 calibrate/analyze/evaluate 三阶段。

### P6 — Coarse Model 性能估算 ✅

### P5 — 可学习量化参数 ✅ (ADR-006)

- LayerwiseScaleOptimizer + PreScaleTransform（Transform 槽位方案）
- PreScaleTransform channel_axis 支持
- Hierarchical study: `part_hierarchical`
- 关键决策：LSQ 走 Transform 槽位，不可引入独立 `lsq` 开关

### P4 — 参数化格式注册 ✅

### P3 — NF4 / 查找表格式 ✅

### P2 — Calibration 管线 ✅

- 4 种 ScaleStrategy（max/percentile/MSE/KL）
- CalibrationPipeline + CalibrationSession
- LSQ Optimizer

### P1 — Transform 体系 ✅

- Hadamard rotation
- SmoothQuant
- PreScale

---

## Phase 2–7（早期 Phase）

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 2 | 三轴扶正：GranularitySpec、TransformBase、FormatBase、消除 MxSpecs | ✅ |
| Phase 3 | 全算子族：Linear/Conv/Norm/Activation/Softmax/Pool/Elemwise/SIMD | ✅ |
| Phase 4 | 层级误差分析：AnalysisContext + QSNR/MSE/Histogram/Distribution Observer | ✅ |
| Phase 5 | ONNX Export：全算子 symbolic() + QDQ + MxQuantize | ✅ |
| Phase 6 | QuantizeContext：torch/F 命名空间 patch + module-stack hooks | ✅ |
| Phase 7 | Unified quantize_model：Module 替换 + forward patching | ✅ |

---

## Bug 修复日志

### 2026-05-12 — SmoothQuant & Hadamard 三大 Bug 修复

**SmoothQuant 根因分析 → 两个独立 bug：**

**Bug 1 — SmoothQuant FP32 基线使用融合后模型**

- 现象：MNIST 上 SmoothQuant 配置 FP32 准确率显示 0.9643（应为 0.9789）；Transformer 上 FP32 困惑度从 7.95 爆炸到 2.78 亿
- 根因：`Session.quantize()` 调用 `fuse_smoothquant_weights()` 创建融合模型（`W ← W * s`），然后传给 `_QuantSession`。`_QuantSession.__init__` 用 `copy.deepcopy(model)` 存储 `fp32_model`，导致 FP32 基线也是融合后模型
- 修复：`_QuantSession.__init__` 新增 `fp32_ref` 参数。当 SmoothQuant 活跃时，`Session.quantize()` 传递原始 `self._model` 作为 `fp32_ref`；`_per_layer_opt.py` 同理
- 改动：`src/session/_quant.py`、`src/session/_session.py`、`src/session/_per_layer_opt.py`

**Bug 2 — SmoothQuant double-scaling（inverse = x * s）**

- 现象：修复 Bug 1 后 FP32 基线正确，但量化后 MNIST 下降 -1.87%、Transformer 困惑度 1.98 亿
- 根因：`SmoothQuantTransform.inverse(x_q) = x_q * s`，导致 `quantize()` 返回 `Q(x/s) * s`。但权重融合已做 `W * s`。Matmul 结果：`(Q(x/s)*s) @ Q(W*s)^T ≈ (x*s) @ W^T ≠ x @ W^T`。s 因子沿 input-channel 维度无法从求和号中因子化出来，输出被 channel-wise scale 污染
- 修复：`inverse` 改为 identity（return x_q）。激活值保持在平滑域 `Q(x/s)`，matmul `Q(x/s) @ Q(W*s)^T ≈ x @ W^T` 正确。`invertible = False`
- 效果：MNIST int4-pb32-sq -0.26%（原 -1.87%），Transformer int4-pb32-sq +4.4%（原灾难性）
- 改动：`src/transform/smooth_quant.py` + `src/tests/test_transform_smooth_quant.py`

**Bug 3 — Hadamard 非 2 的幂维度截断**

- 现象：MNIST Hadamard -21.2%、Transformer Hadamard +100% 困惑度，QSNR 6.9 dB
- 根因：旧实现对非 2 的幂维度 padding 到下一个 2 的幂 → Hadamard → 截断。截断是 lossy 操作：forward 丢弃的 padding 区域包含信息，inverse 时补零是错误的 Hilbert 空间基。d=192（pad to 256）时 roundtrip 矩阵对角仅 0.75、跨元素串扰 25%
- 修复：将任意维度分解为 2 的幂 chunks（如 192=128+64），每个 chunk 独立 Hadamard（block-diagonal orthogonal）。`hadamard(hadamard(x)) == x` 对所有维度精确成立（max error ~7e-7）
- 效果：MNIST Hadamard -0.26%（原 -21.2%），Transformer Hadamard +5.9%（原 +100%），QSNR 6.9→18.8 dB
- 改动：`src/transform/hadamard.py`

**最终 Transformer 排名（FP32=7.95）：**

| Config | Quant PPL | Δ | QSNR |
|--------|-----------|---|------|
| nf4-pc | 8.16 | +0.21 | 26.8 dB |
| int4-pb32-sq | 8.30 | +0.35 | 1.8 dB |
| int4-pb32-had | 8.42 | +0.47 | 18.8 dB |
| int4-pb32 | 8.43 | +0.48 | 19.0 dB |

测试：全量 174 passed（session + SQ + Hadamard + per_layer_opt）

---

### 2026-05-12 — Per-Channel / Int Per-Tensor GELU NaN 修复

**问题：**
- Transformer 上 int4-pc / int4-pt / int8-pc 三种 config 的 quant_perplexity 均为 NaN
- MNIST 上同样 config 也产生 NaN accuracy

**根因：**
- `_activation_cfg()` 对 per_channel 和 integer per_tensor compat-style configs 返回 cfg 本身，导致激活函数（GELU/SiLU/Softmax）的每个中间步骤都用相同的 scheme 量化
- GELU detailed 路径中间产生 exp(48) ≈ 7e20，同一 channel 内 small values（~1.0）归一化后被 crush 为 0
- `vec_recip(0) → inf`，后续 `inf * 0 → NaN`
- Float per_tensor（bf16/bf10, ebits>0）不受影响 — `_quantize_per_tensor` 对 float 格式直接 elemwise，不做 amax 归一化

**方案：**
- 新增 `_scheme_normalizes_by_amax()` — 判断 scheme 是否会做 amax 归一化（per_channel 总是归一化；integer per_tensor ebits=0 也归一化）
- `_activation_cfg()` 对这类 scheme 返回空 OpQuantConfig，激活函数中间步骤在 fp32 运行
- Float per_tensor 保持 pass-through（直接 elemwise，安全）
- 激活输入已由前一层 Linear/Conv 量化，中间无需重复量化

**改动文件：**
- `src/session/_model.py` — 新增 `_scheme_normalizes_by_amax()` + 更新 `_activation_cfg()`
- `src/tests/test_session_unit.py` — 更新原有测试 + 新增 2 个测试（int per_tensor、per_channel）

**Transformer 修复后（FP32=0.8224）：**
| Config | Quant Acc | Δ | QSNR |
|--------|----------|---|------|
| int4-pc | 0.8004 | -2.2% | 25.7 dB |
| int8-pc | 0.8234 | +0.1% | 48.9 dB |

测试：全量 2,421 passed（fast）

---

### 2026-05-08

- **quantize_backprop 修正 — Transformer 全量 backward bit-exact 验证通过**。根因：`_make_ln/gn/bn/rms_norm` 传入 `quantize_backprop=cfg.is_training` 但 `cfg` 为原始 config（backward fields 均为 None），导致 norm backward 中 vec_ops 以 fp32 执行而非 bf16 量化。修正：传入 `_non_matmul_cfg(cfg).is_training`。同时修复了 activations/softmax/pooling 模块多余的 pre/post quantization（在 autograd Function 外部调用导致梯度断裂）。Transformer 验证 6 配置全部 PASS。
- **fp8 storage 修正 + storage_format 显式格式名**。`verify_transformer_equiv.py` 中 fp8 storage `mbits=2` 修正为 `FormatBase.from_str("fp8_e5m2")`（匹配 MX `exp_bits=5, mantissa_bits=2`）。`QuantConfig` 新增 `storage_format` 字段，支持显式格式名。Transformer 全矩阵 12 配置全部 PASS。
- **回归测试覆盖补充**。针对以上 2 个 bug 补充 15 tests（_non_matmul_cfg、norm backward、STE、E2E backward）。
- **E2E 测试矩阵扩展完成**。新增 6 个测试类，覆盖 8 种格式 × 3 种 storage × 2 种 QBP。全量 2,069 passed。
- **quantize_nonlinear 开关**。`QuantConfig.quantize_nonlinear=False` 使 norm/activation/pool 保持 fp32。README 全面更新。
- **ADR-009: quantize_nonlinear 非线性算子统一量化策略**。决策：`quantize_nonlinear=True` 对非线性算子入口 operand 施加两级量化，中间 vec_ops 和 backward 保持 storage-only。

### 2026-05-07

- **全算子端到端等价性验证通过**。验证全部 21 种模块类型 + 全部 inline ops 的 bit-exact 等价性。修复 5 个 bug：参数重命名、model.eval() 顺序、output field 错误施加、per-module dict 模式 op_cfgs、combine-add 结合律。
- **Framework review 13 issues resolved**（P0–P2）：C1 统一反序列化、C2 STUDY_CONFIG 修复、C3 to_op_config output compute、I1 storage_bits/storage_kind 重命名、I2 删除 _utils/、I3 scale_storage 统一命名、I4 prescale_granularity 条件化、M1 _QuantSession 私有化、M3 w_axis/a_axis 字段、L1 per_layer_optimal 复用 Session helpers、L2 去重 _VALID_ROUND_MODES、L3 study_config.py 英文化。

---

## 关键经验记录

1. **P5 LSQ 走 Transform 槽位**（ADR-006）：`QuantConfig` 中体现为 `transform="prescale"` + `lsq_steps > 0`，不可引入独立 `lsq` 开关。
2. **Pipeline Refactor IoC 模式**：单回调驱动 calibrate/analyze/evaluate 三阶段。
3. **Module boundary 强制执行**：viz 模块不含 pipeline/session import（AST 静态检查通过）。
4. **Type guards 是硬性要求**：每个公共 API 参数的类型守卫必须配 `pytest.raises + match=` 测试。
5. **quantize_model 不替换根模块**：裸 `nn.Linear` 需用 wrapper 模型。
6. **_elem_bits 公式**：IntFormat/LookupFormat 取 `mbits`；FPFormat/BFloat16Format 取 `ebits+mbits-1`。
7. **Per-channel PreScaleTransform 不能用于 matmul 输出角色**：s 无法从 matmul 因子化出来。
8. **Hierarchical = PreScaleTransform + MX PER_BLOCK**：两级 scale（全局 PoT pre-scale + block 共享指数）。
9. **Session 内部三层委托**（ADR-008 §5.1.1）：`Session` → `_QuantSession` → `quantize_model()`。
