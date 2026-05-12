# ADR-010: 系统化误差分析闭环

**状态**: 待实施
**日期**: 2026-05-12
**依赖**: ADR-002 (Observer), ADR-005 (OpQuantConfig), ADR-008 (Session/Study)

---

## 背景与问题

当前分析能力由低层 building blocks 组成（QSNRObserver、DistributionObserver、error_source_analysis），但缺少系统化的分析工作流。用户的诊断流程是手动的、碎片化的：

```python
# 当前：散落的工具调用，用户自己串流程
result = Session(model, cfg).run(calib_data, eval_fn=eval_fn)
print(result.top_k_qsnr(5))                    # "哪层最差"
print(result.tables.error_source_analysis())    # "误差从哪来"
# ... 然后用户手工去找问题层的分布、手工改 config、手工对比
```

### 核心缺陷

1. **单 role 视角**：`_extract_qsnr_mse()` 和 `qsnr_per_role()` 默认只看 output role，input/weight QSNR 虽在 raw data 中但无系统化提取和分析
2. **分布数据"采而不用"**：DistributionObserver 采集 16 个统计量（crest_factor、skewness、bimodality、outlier_ratio 等），但没有从"分布特征"到"量化退化机制"的因果映射
3. **干预靠手工**：用户发现某层 weight QSNR 低 → 手工改 QuantConfig → 重新 run → 手工对比。没有"生成 per-layer override → 自动对比"的工具链
4. **Transform 效果不透明**：`adaptive` transform selection 只报告最终选了哪个，不报告各候选的 margin 和 ranking
5. **误差传播不可视**：accumulated QSNR 已采集，但没有 DAG/瀑布图展示误差如何在层间传播和放大
6. **图散落各处**：单 result 的诊断图没有统一的 plot accessor（StudyReport 有 `report.plot`，SessionResult 没有对应的 `result.plot`）

---

## 决策

### 1. 四阶段闭环

```
┌─ DIAGNOSE ───────── result.diagnose (ErrorProvenance)
│  全局概览 → 误差溯源 → 定位问题层/role
│
├─ CHARACTERIZE ───── result.characterize (DistributionDiagnosis)
│  问题层分布特征 → 退化机制分类 → 因果映射
│
├─ INTERVENE ──────── result.plan (InterventionPlanner)
│  精度分配 → transform ranking → 推荐方案 → 生成 per-layer override
│  ↓ 产出 InterventionPlan
│  result.intervention (InterventionAccessor)
│  .compare(plan) → 应用方案 → 跑新 Session → 对比 → 验证
│
└─ VERIFY ─────────── 闭环确认
   InterventionComparison.print_summary() + .plot.before_after()
```

四个 accessor 均挂在 `SessionResult` 上，零学习成本。

### 2. Session per-layer override

底层的 `_QuantSession.__init__` 已接受 `Union[OpQuantConfig, Dict[str, OpQuantConfig]]`，`quantize_model` 已处理 dict。只需在 `Session` 层暴露 `overrides` 参数：

```python
# Session.quantize() 中:
if overrides:
    base_cfg = self._config.to_op_config()
    per_layer = {name: base_cfg for name in matched_module_names}
    per_layer.update(overrides)
    self._quant_session = _QuantSession(model, per_layer, ...)
```

`overrides` 的 key 是 module name（与 `named_modules()` 一致），value 是 `OpQuantConfig`——用户可以用更 bit/换 transform 的 scheme 覆盖任意 field。

### 3. 图归入 result.plot

新增 `SessionPlotAccessor`，挂在 `SessionResult.plot`，包含所有单 result 诊断图。与 `StudyReport.plot`（多 config 对比图）互补：

| Accessor | 挂载点 | 用途 |
|----------|--------|------|
| `SessionPlotAccessor` | `result.plot` | 单 config 诊断（分布、传播、per-role） |
| `StudyPlotAccessor` | `report.plot` | 多 config 对比（已有，不变） |

### 4. top-k 精度分配：按 role 独立决策

`InterventionPlanner.top_k_boost(k, role, target_bits)` 的核心逻辑：

1. 按指定 role 的 QSNR 升序排列所有层
2. 取最差的 k 层
3. 为每层生成一个新的 `OpQuantConfig`：该 role 的 scheme 替换为 `target_bits` 宽度的格式，其余 field 不变
4. role="auto" 时：每层选 QSNR 最低的 role 作为该层的提升目标

不涉及复杂优化——纯 QSNR 排序 + config 替换。

---

## 目标 API

### Diagnose: ErrorProvenance

```python
prov = result.diagnose

# 总览：按 role × layer_type 分层统计
print(prov.summary())
# Role     Type      Count   Avg QSNR   Min QSNR   Std
# input    Linear    12      28.3       18.2       5.1
# weight   Linear    12      22.1        8.7       7.8
# output   Linear    12      25.4       12.1       6.3

# 每层 input/weight/output 三列并排
print(prov.per_role_table(max_layers=20))
# Layer              Input    Weight   Output   Dominant
# layer1.linear      35.2     12.1     24.3     weight
# layer3.linear      30.1      8.7     18.2     weight

# 误差溯源（已有能力，整合）
print(prov.error_source_analysis())

# Top-K 问题定位
prov.top_k(k=5, role="weight")        # weight role 最差的 5 层
prov.top_k(k=10, role="auto")         # 每层取最差 role 再全局排序
```

**数据来源**: `qsnr_per_role()` 对 input/weight/output 各调一次 + `accum_qsnr_per_layer`。

### Characterize: DistributionDiagnosis

```python
diag = result.characterize

# 单层深度诊断
print(diag.profile("layer3.linear", role="weight"))
# layer3.linear (weight)  QSNR=8.7 dB
#   Distribution: Laplace (KS=0.03)
#   Crest factor: 18.3 (high)
#   Outlier ratio: 4.2% (>3σ)
#   Diagnosis: outlier-dominated

# 全局因果矩阵
print(diag.causal_analysis())
# Layer            Role    QSNR   Crest   OL%    DR_bits  Classification
# layer3.linear    weight   8.7   18.3   4.2%    11.2     outlier_dominated
# layer5.conv      input   15.2    6.1   0.3%     8.1     high_dynamic_range

# 退化机制分类（规则引擎）
diag.classify("layer3.linear", role="weight")
# → "outlier_dominated"
```

**规则引擎的退化分类**（基于 DistributionObserver 统计量）：

| 分类 | 判定条件 | 建议干预 |
|------|---------|---------|
| `outlier_dominated` | outlier_ratio > 2% 且 crest_factor > 10 | per_channel / hadamard / boost bit |
| `high_dynamic_range` | dynamic_range_bits > 8 且 outlier_ratio ≤ 2% | per_channel / hadamard |
| `heavy_tailed` | excess_kurtosis > 3 且 outlier_ratio > 1% | smoothquant / pre-scale |
| `bimodal` | bimodality_coefficient > 0.7 | 不易量化，考虑保留 fp32 |
| `low_entropy` | norm_entropy < 0.3 | 量化友好，可激进压缩 |
| `benign` | 以上均不满足 | 维持当前配置 |

### Plan: InterventionPlanner

```python
planner = result.plan

# Top-K 精度提升
plan = planner.top_k_boost(k=3, role="weight", target_bits=8)
plan = planner.top_k_boost(k=5, role="auto", target_bits=8)

# Transform 效果排序
ranking = planner.transform_ranking(k=5)
# Layer            none    hadamard  smooth   Best      Margin
# layer3.linear     8.7    16.2      19.8     smooth    +3.6 dB
# layer1.linear    22.1    24.5      25.1     hadamard  +0.6 dB (marginal)

# 自动推荐
plan = planner.recommend(strategy="conservative")  # 少改
plan = planner.recommend(strategy="aggressive")    # 能改都改

# InterventionPlan
plan.overrides           # Dict[str, OpQuantConfig]
plan.explain()           # 文本表: 每层改了什么、为什么
plan.to_dict()           # 序列化
```

### Intervention: InterventionAccessor

```python
# 应用并对比
cmp = result.intervention.compare(
    plan,
    model=model,
    calib_data=calib_data,
    eval_fn=eval_fn,
)

cmp.print_summary()
# Config          Acc(FP32)  Acc(Quant)  ΔAcc    Avg QSNR  #Modified
# baseline        0.9500     0.9100      -0.040  18.3      0
# top3_w8         0.9500     0.9270      -0.023  22.1      3

cmp.plot.before_after()
cmp.plot.qsnr_improvement()
```

`.compare()` 内部：用 `plan.overrides` 创建新 `Session` → `run()` → 拿到新 `SessionResult`，然后与 `self`（baseline）对比。返回的 `InterventionComparison` 有自己的 `.plot` 和 `.tables`，模式与 `StudyReport` 一致。

### Plot: SessionPlotAccessor

```python
# 误差传播
result.plot.propagation_dag()          # DAG: 节点=层, 颜色=local QSNR, 边=accum衰减
result.plot.error_waterfall()          # 逐层 accum QSNR 瀑布下降图
result.plot.local_vs_accum_scatter()   # 散点: local vs accum QSNR

# Per-role
result.plot.per_role_qsnr_bars()       # 每层堆叠 bar: input/weight/output QSNR
result.plot.depth_decay(role="output") # QSNR 随深度衰减曲线

# 分布
result.plot.layer_histogram("layer3.linear", role="weight")
result.plot.channel_heterogeneity("layer3.linear", role="weight")
```

---

## 新增数据（复用现有 observer，零新数据采集）

以上所有能力只消费 `SessionResult` 中已有字段：

| 字段 | 来源 | 被哪些 accessor 消费 |
|------|------|---------------------|
| `qsnr_per_layer` | QSNRObserver (output role, 已有) | diagnose, plan |
| `mse_per_layer` | MSEObserver (已有) | diagnose, plan |
| `accum_qsnr_per_layer` | analyze hook path (已有) | diagnose, visualize |
| `accum_mse_per_layer` | analyze hook path (已有) | diagnose |
| `observers_data` (input/weight roles) | QSNRObserver + DistributionObserver (已有，需开启 distribution) | diagnose, characterize |

`result.diagnose` 需要按 role 提取 QSNR，只需在现有 `qsnr_per_role()` 基础上扩展——该方法已支持 role 参数，现在默认只调了 output，diagnose 会对 input/weight/output 各调一次。

**唯一的新数据采集需求**：开启 `DistributionObserver`（`outputs=["distribution"]`）才能使用 `characterize`。如果不开启，`characterize` 的方法优雅降级，返回带 HOW-TO 提示的文本。

---

## 文件变更

### 新增文件

```
src/analysis/_error_provenance.py       ErrorProvenance accessor
src/analysis/_distribution_diagnosis.py DistributionDiagnosis accessor (规则引擎)
src/analysis/_intervention.py           InterventionPlanner + InterventionPlan
src/analysis/_intervention_accessor.py  InterventionAccessor + InterventionComparison
src/report/_session_plot.py             SessionPlotAccessor (单 result 诊断图)
src/viz/_propagation.py                 DAG / waterfall / scatter 图实现
src/viz/_per_role.py                    per-role bar / depth decay 图实现
```

### 修改文件

```
src/session/_result.py                  新增 .diagnose / .characterize / .plan
                                        / .intervention / .plot 五个 property
src/session/_session.py                 Session.__init__ 新增 overrides 参数;
                                        Session.quantize() 合并 overrides 到
                                        _QuantSession per-layer dict
src/report/_session_tables.py           新增 per_role_qsnr() 表方法
docs/architecture/INDEX.md              新增 ADR-010 条目
```

### 不修改

- `src/analysis/observers.py` — 所有 observer 保持不变
- `src/scheme/op_config.py` — OpQuantConfig 保持不变
- `src/session/_quant.py` — _QuantSession 已支持 dict cfg
- `src/report/_study_report.py` — StudyReport.plot 保持不变
- `src/report/_plot.py` — StudyPlotAccessor 保持不变

---

## Session API 变更（最小侵入）

```python
# 唯一的新参数：overrides
class Session:
    def __init__(
        self,
        model: nn.Module,
        config: QuantConfig,
        *,
        keep_fp32: bool = True,
        overrides: Optional[Dict[str, OpQuantConfig]] = None,  # NEW
    ):
```

`overrides` 在 `quantize()` 中与 base config 合并：

```python
def quantize(self, ...):
    base_cfg = self._config.to_op_config()
    if self._overrides:
        # 收集所有量化模块名，base_cfg 作为默认
        from src.session._model import _get_quantized_modules
        per_layer = {}
        for name, _ in _get_quantized_modules(model):
            per_layer[name] = base_cfg
        per_layer.update(self._overrides)
        self._quant_session = _QuantSession(model, per_layer, ...)
    else:
        self._quant_session = _QuantSession(model, base_cfg, ...)
```

---

## 实施阶段

### Phase 1: 数据基础（`qsnr_per_role` 扩展）

- `_extract_qsnr_mse()` 现只提取单个 role。改为一次遍历提取所有 role
- `SessionResult.qsnr_per_role()` 已支持 role 参数，没问题
- 在 `analyze()` 中自动对所有 role 收集 QSNR（目前 observer 已经全部采集，只是提取环节只取了 output）

### Phase 2: Diagnose + Plot

- `ErrorProvenance` accessor: summary / per_role_table / error_source / top_k
- `SessionPlotAccessor`: per_role_qsnr_bars / depth_decay / propagation DAG / waterfall / scatter
- `result.diagnose` 和 `result.plot` property

### Phase 3: Characterize

- 规则引擎：6 个退化分类（outlier_dominated / high_dynamic_range / heavy_tailed / bimodal / low_entropy / benign）
- `DistributionDiagnosis` accessor: profile / classify / causal_analysis
- `result.characterize` property
- plot: layer_histogram / channel_heterogeneity

### Phase 4: Intervene

- `InterventionPlanner` + `InterventionPlan`: top_k_boost / transform_ranking / recommend
- `Session` `overrides` 参数打通
- `InterventionAccessor` + `InterventionComparison`: compare / print_summary
- `result.plan` 和 `result.intervention` property

---

## 风险与取舍

| 风险 | 缓解 |
|------|------|
| `qsnr_per_role` 对 input/weight/output 全提取会增加 O(role) 的时间/内存 | 复杂度从 O(n) 到 O(3n)，n=层数，实际影响可忽略。observer data 已在内存中 |
| 规则引擎误分类 — 分布特征与 QSNR 退化之间的因果不是严格的 | 规则引擎输出"建议"，不是自动执行。用户始终可以无视 recommendation 手工指定 plan |
| `top_k_boost` 只按 QSNR 排序，未考虑层重要性差异 | Phase 1 先做纯 QSNR。后续可加 layer sensitivity weighting（基于 gradient norm 或 accuracy impact） |
| overrides 的 module name 匹配容易写错 | `InterventionPlan.explain()` 打印所有匹配结果，warning 未匹配的 key |

---

## E2E 回归门

所有新增的分析模块（不修改量化路径）不会影响现有 E2E。但 `Session` 新增 `overrides` 参数需要测试。回归门不变：

```
PYTHONPATH=. python scripts/mnist_hadamard_study.py
PYTHONPATH=. python scripts/transformer_agnews_study.py
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
```
