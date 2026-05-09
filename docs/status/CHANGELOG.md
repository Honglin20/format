# Changelog — 已完成任务归档

> 本文件记录已完成的 Phase、子任务和关键 bug 修复。
> 当前进行中的任务见 `CURRENT.md`。

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
