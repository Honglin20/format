# ADR-008: Session 统一入口 + Output-Driven 架构

**状态**: 待实施（2026-05-07 review 修正版）
**日期**: 2026-05-07
**审视**: `docs/reviews/2026-05-07-adr008-review.md`

---

## 背景与问题

当前架构中，用户完成"比较多个量化格式"需要理解两个入口、散落的配置 dict、手动指定 Observer：

```python
# 当前：两个概念、散落字段、手动 Observer
session = QuantSession(model, cfg, observers=[QSNRObserver(), MSEObserver()])
results = run_format_study(build_model=..., make_calib_data=..., eval_fn=...)
```

### 核心架构缺陷

1. **入口割裂**：`QuantSession`（单配置）和 `run_format_study`（多配置）是两个独立概念，用户必须理解二者关系
2. **Pipeline 依赖倒挂**：`pipeline/ExperimentRunner` 依赖 `session/QuantSession`，但 CLAUDE.md 声称 tools 被 session 依赖——方向相反
3. **Observer 暴露**：用户需要知道 `QSNRObserver`、`MSEObserver`、`HistogramObserver` 的存在，而这本是实现细节
4. **配置散落**：量化参数分散在 dict 的多个 key（`format`、`granularity`、`lsq_steps`、`transform`...），无 IDE 补全，字段不可发现
5. **LSQ 定位模糊**：`lsq_steps` 作为独立开关存在，但 LSQ 本质是 PreScaleTransform + 外部优化器（ADR-006 已确认），不是独立概念
6. **工具层分类扁平**：`calibration`、`analysis`、`cost`、`onnx`、`pipeline`、`viz` 扁平排列，缺少"能力 vs 输出"的区分

---

## 决策

### 1. 三概念层级

```
Study          ← 聚合层：N 个 Session → 对比报告（纯 for 循环，零量化逻辑）
  │
  └─ Session   ← 执行单元：一个 QuantConfig → 一个 SessionResult
       │
       └─ QuantSession  ← 低层：一个模型 + 一个 OpQuantConfig 的生命周期
                           （calibrate / analyze / evaluate / export）
```

**关键原则：** Study 不是一种 Session，Study 是 Session 结果的聚合。两者职责完全不同。

- `Session` = 原子执行单元，不可再分。内部包含量化、标定、分析、评估的完整流程
- `Study` = 纯聚合，内部只做 `for cfg in configs: Session(model, cfg).run()`，不包含任何量化逻辑
- `QuantSession` = 低层 API，给需要精细控制的用户

**PerLayerOpt 保留为独立工具函数**（非 Study 方法）。PerLayerOpt 是"按层选最优 transform 并重跑"的后处理算法（对应 `format_study.py:540-643` ~80 行逻辑），放在 `session/_per_layer_opt.py` 作为公开工具：

```python
from src.session import per_layer_optimal

results = study.run(...)                       # 纯聚合
opt_result = per_layer_optimal(results, ...)    # 用户显式调用 PerLayerOpt
```

此函数不打破 Study 的纯聚合承诺，用户主动使用。

### 2. QuantConfig —— 用户唯一配置入口

所有可配置维度集中在一个 dataclass，IDE 自动补全即可发现全部可配置项：

```python
@dataclass
class QuantConfig:
    name: str = ""

    # ── Weight 格式 ──
    w_format: str = "int8"
    w_granularity: str = "per_tensor"   # per_tensor | per_channel | per_block
    w_block_size: int | None = None     # 仅 per_block 需要

    # ── Activation 格式（None = 同 weight）──
    a_format: str | None = None
    a_granularity: str = "per_tensor"
    a_block_size: int | None = None

    # ── Transform（见 §2.1 transform 规则表）──
    transform: str = "none"             # none | hadamard | smoothquant | prescale
    sq_alpha: float = 0.5               # SmoothQuant 迁移强度
    prescale_init: str = "ones"         # prescale 初始化: ones | amax | pot_amax
    prescale_pot: bool = False          # prescale 投影到 PoT
    prescale_granularity: str = "per_tensor"  # per_tensor | per_channel
        # 默认跟随 a_granularity。仅当 prescale 粒度需不同于量化粒度时显式覆写

    # ── LSQ ──
    # 仅当 transform="prescale" 时生效。若 transform≠"prescale" 且 lsq_steps>0，应报错
    lsq_steps: int = 0                  # 0 = 静态 prescale, >0 = LSQ 优化步数
    lsq_lr: float = 1e-3

    # ── Scale 存储 ──
    scale_storage: str = "fp32"         # fp32 | pot

    # ── 标定策略 ──
    calibrator: str = "mse"             # mse | max | percentile | kl

    # ── 模式 ──
    weight_only: bool = False           # 只量化 weight，不量化 activation
```

**设计原则：**
- `calibrator` 在 `QuantConfig` 中——"用户唯一配置入口"的承诺必须完整。`Session.__init__` 不再暴露独立 calibrator 参数
- `transform="prescale"` + `lsq_steps > 0` = 可学习 prescale（即 LSQ）。`lsq_steps` 只是 prescale 的优化配置，不是独立模式。若 `transform≠"prescale"` 且 `lsq_steps > 0`，`to_op_config()` 应抛出 `ValueError`
- `a_format=None` → activation 与 weight 同格式；设值 → wXaY mixed-precision
- `scale_storage` 独立于 format/granularity/transform，正交维度
- `prescale_granularity` 默认跟随 `a_granularity`，仅极边缘场景需显式覆写
- `QuantConfig.to_op_config()` 负责转换为内部 `OpQuantConfig`，用户无需关心

#### 2.1 transform → weight/activation 分配规则

`to_op_config()` 对不同 transform 值的 weight/act 分配不同（对应 `resolve_config()` 和 `_make_sq_op_cfg()` 的行为差异）：

| transform | weight 角色 | activation 角色（input/output） |
|-----------|------------|-------------------------------|
| `"none"` | Identity | Identity |
| `"hadamard"` | Hadamard | Hadamard |
| `"prescale"` | Identity（占位，运行时替换） | Identity（占位，运行时替换） |
| `"smoothquant"` | Identity | SmoothQuant（per-layer，运行时标定） |

**注意 `"prescale"` 的两步翻译**：`to_op_config()` 无法一步完成 prescale 翻译，因为 `PreScaleTransform(scale=tensor)` 的 tensor 在 `initialize_pre_scales()` 之后才存在（§2.2）。

#### 2.2 prescale 两步翻译机制

`transform="prescale"` 的翻译必须在运行时分两步完成（对应当前 `ExperimentRunner.run()` L108-124 + `QuantSession.initialize_pre_scales()` L302-408）：

```
Step 1 (QuantConfig.to_op_config):
    输出 OpQuantConfig，其中 transform 均为 IdentityTransform() 占位

Step 2 (Session.run 内部):
    a. qs = QuantSession(model, op_cfg)
    b. qs.initialize_pre_scales(calib_data, init=config.prescale_init,
                                pot=config.prescale_pot,
                                granularity=config.prescale_granularity)
       → 创建 PreScaleTransform(scale=tensor)
       → _replace_transform(qs.qmodel.module.cfg, pre_scale_transform)
    c. [if lsq_steps > 0] qs.optimize_scales(opt, calib_data)
    d. 继续 calibrate → analyze → evaluate
```

**设计约束**：`to_op_config()` 永远是纯数据转换（无副作用、无运行时张量依赖）。prescale 的张量创建必须在 `Session.run()` 的运行时阶段完成。

### 3. Output-Driven Observer 选择

用户声明想要的**输出**，系统推导需要的 Observer。Observer 是内部实现细节，永不暴露给用户。

```python
# 用户声明
outputs=["histogram", "qsnr", "accuracy", "cost"]

# 系统推导（查 _OUTPUT_SPEC 表）
→ observers: {HistogramObserver, QSNRObserver}
→ needs_eval: True (accuracy 需要)
→ needs_cost: True (cost 需要)
→ 执行: calibrate → analyze(2 observers) → evaluate → cost
→ 生成: histogram_overlay, qsnr_line_chart, accuracy_table, cost_report
```

#### 3.1 _OUTPUT_SPEC 完整映射表（17 个 key）

`_spec.py` **只使用字符串 key**（`"qsnr"`、`"mse"` 等），不持有 observer class 引用。observer class 解析逻辑放在 `session/_session.py` 中。`report/` 包完全不 import `analysis.observers`。

```python
# src/report/_spec.py — 只含字符串 key，不 import analysis
_OUTPUT_SPEC = {
    # ── 表格 ──
    "accuracy":         {"observers": [],              "needs_eval": True},
    "sensitivity":      {"observers": ["qsnr"],        "needs_eval": True},
    "pot_delta":        {"observers": [],              "needs_eval": True},
    "transform_matrix": {"observers": ["qsnr"],        "needs_eval": True},
    "transform_dist":   {"observers": ["qsnr"],        "needs_eval": True},

    # ── 图表 ──
    "qsnr":             {"observers": ["qsnr"],        "needs_eval": False},
    "mse":              {"observers": ["mse"],         "needs_eval": False},
    "histogram":        {"observers": ["histogram"],   "needs_eval": False},
    "error_dist":       {"observers": ["distribution", "mse"], "needs_eval": False},
    "transform_heatmap":{"observers": ["qsnr"],        "needs_eval": True},
    "transform_pie":    {"observers": ["qsnr"],        "needs_eval": True},
    "transform_delta":  {"observers": ["qsnr", "mse"], "needs_eval": True},
    "layer_qsnr":       {"observers": ["qsnr"],        "needs_eval": False},
    "block_sweep":      {"observers": ["qsnr"],        "needs_eval": True},
    "hierarchical":     {"observers": ["qsnr", "mse"], "needs_eval": True},
    "pot_delta_bar":    {"observers": [],              "needs_eval": True},

    # ── 其他 ──
    "cost":             {"observers": [],              "needs_eval": False, "needs_cost": True},
}
```

**presets:**
- `outputs="default"` → `["accuracy", "qsnr"]`
- `outputs="all"` → 全部 17 个 key

**observer key → class 解析**（在 `session/_session.py` 中，不在 `report/` 中）：

```python
_OBSERVER_CLASSES = {
    "qsnr": QSNRObserver,
    "mse": MSEObserver,
    "histogram": HistogramObserver,
    "distribution": DistributionObserver,
}
```

### 4. 包结构变更

```
session/                          ← 用户唯一入口
├── __init__.py                   QuantConfig, Session, SessionResult, Study,
│                                 QuantSession, quantize_model, per_layer_optimal
├── _config.py                    QuantConfig dataclass + to_op_config()
├── _session.py                   Session + SessionResult（NEW，吸收 pipeline/runner.py）
├── _quant.py                     QuantSession（原 _session.py，重命名）
├── _study.py                     Study（NEW，纯聚合）
├── _per_layer_opt.py             per_layer_optimal 工具函数（NEW，吸收 format_study.py:540-643）
├── _model.py                     quantize_model（不变）
├── _context.py                   QuantizeContext（不变）
├── _patches.py                   torch/F patching（不变）

report/                           ← NEW：声明式输出层
├── __init__.py                   SessionReport, StudyReport
├── _spec.py                      _OUTPUT_SPEC 映射表（只含字符串 key，不 import analysis）
├── _registry.py                  viz 函数 lazy 注册表（从 pipeline/report.py 迁移）
├── _converters.py                SessionResult → viz dict 转换器
├── _session_report.py            SessionReport（单 config 结果）
├── _study_report.py              StudyReport（多 config 聚合，吸收 pipeline/report.py）

REMOVED:
pipeline/                         ✕ 整个包删除

MODIFIED:
scheme/op_config.py               + OpQuantConfig.from_descriptor() + from_parts()
transform/smooth_quant.py         + SmoothQuantTransform.from_model_calibration() + fuse_weights()
```

### 5. 依赖方向

#### 5.1 整体层级

```
        ┌─────────────────────────────────────────┐
        │              session (驱动层)             │
        │  Study             聚合（依赖 Session）    │
        │  per_layer_optimal 后处理工具             │
        │  Session           执行（依赖 QuantSession）│
        │  QuantSession      低层（依赖各能力包）     │
        │  quantize_model                          │
        └──────┬──────────┬───────────┬────────────┘
               │          │           │
    ┌──────────┼──────────┼───────────┼──────────────┐
    │          ▼          ▼           ▼              │
    │    calibration  analysis      cost     onnx   │ 能力层
    │          │          │                         │
    │          └────┬─────┘                         │
    │              ▼                                │
    │          observer                             │ 横切基础设施
    └───────────────────────────────────────────────┘

    ┌──────────┐
    │  report   │  ← 输出层（消费 SessionResult，调用 viz）
    └────┬─────┘
         ▼
        viz

    数学层: formats / scheme / transform / quantize
    算子层: ops/
```

#### 5.1.1 session/ 内部三层委托

`Session`、`_QuantSession`、`quantize_model()` 三者不是冗余——它们是严格分层委托关系：

```
Session                    ← 用户 API：QuantConfig（字符串）、transform 预处理、链式 .quantize().calibrate()...
  │  .to_op_config() + 创建
  ▼
_QuantSession              ← 工作流包装：calibrate / analyze / compare / export / LSQ / cost / mode 切换
  │  在 __init__ 中调用 quantize_model()
  ▼
quantize_model()           ← 唯一引擎：递归替换 nn.Module → QuantizedXxx + 劫持 forward 套 QuantizeContext
```

**各层职责：**

| 层 | 输入 | 职责 | 何时直接用 |
|---|---|---|---|
| `Session` | `QuantConfig`（字符串字段） | 翻译字符串→对象、SmoothQuant/Prescale 预处理、收集 SessionResult | 正常用户入口 |
| `_QuantSession` | `OpQuantConfig`（对象） | 工作流编排：calibrate/analyze/compare/export/LSQ/cost | 需要精细控制工作流的用户 |
| `quantize_model()` | `OpQuantConfig` | 模块替换 + forward 劫持。纯引擎，无工作流概念 | 测试、验证脚本、只要模块替换不要工作流的场景 |

**关键设计约束：**

- `quantize_model()` 是 session/ 内**唯一**进行模块替换和 forward 劫持的位置。`_QuantSession` 和 `Session` 都不直接操作模块替换逻辑。
- `_QuantSession.__init__` 只做三件事：deepcopy fp32 模型、调用 `quantize_model()`、存储引用。所有工作流方法（calibrate/analyze/...）都通过已有的 `self.qmodel` 操作。
- `Session.quantize()` 创建 `_QuantSession` 之前，先处理 SmoothQuant（计算 per-channel scale + 融合权重）和 Prescale（初始化 pre_scale + LSQ 优化），因为这些需要访问**原始 fp32 模型**。
- `quantize_model()` 保留为 public API（`from src.session import quantize_model`），因为验证脚本和单元测试需要直接调用它来验证模块替换的 bit-exact 等价性，而不希望引入 session 工作流的开销。

#### 5.2 session/ ↔ report/ 边界（解决循环依赖）

`Study.run()` 返回 `StudyReport`（来自 `report/`）。`StudyReport.__init__` 接收 `Dict[str, List[SessionResult]]`（`SessionResult` 来自 `session/`）。

这构成了潜在循环依赖：
```
session/_study.py  →  import StudyReport  from report/     (方向 A)
report/_study_report.py  →  import SessionResult from session/  (方向 B)
```

**解决方案：方向 A 使用函数体内 lazy import，模块级保持单向。**

```python
# session/_study.py — 模块级不 import report/
class Study:
    def run(self, ...) -> "StudyReport":
        ...
        from src.report import StudyReport    # ← lazy import，仅运行时执行
        return StudyReport(results)
```

```
模块级依赖:  report → session  (report 依赖 session 的 SessionResult 类型)
运行时依赖:  session._study.run() → lazy import report.StudyReport
```

这是 Python 标准模式（Django、SQLAlchemy、Pydantic 均使用），不是 hack。`SessionResult` 作为简单 dataclass 定义在 `session/_session.py` 中，与 `Session` 同居——生产者拥有自己的输出类型。

### 6. 用户 API

```python
from src.session import Session, Study, QuantConfig, per_layer_optimal


# ═══ 单配置 ═══
cfg = QuantConfig(name="int8", w_format="int8")
result = Session(model, cfg).run(calib_data, eval_data=eval_loader)


# ═══ 多配置对比 ═══
study = Study([cfg_a, cfg_b, cfg_lsq], model=model)
report = study.run(calib_data, eval_data=eval_loader, outputs="all")
report.save("results/")


# ═══ PerLayerOpt（用户显式调用）═══
results = study.run(calib_data, eval_data=eval_loader)
opt_result = per_layer_optimal(results, calib_data, eval_fn=my_eval)
full_report = StudyReport({**results, "per_layer_opt": [opt_result]})


# ═══ 重新可视化历史结果 ═══
from src.report import StudyReport
StudyReport.from_file("results/results.json").save("results/regen/")


# ═══ 低层精细控制 ═══
from src.session import QuantSession
qs = QuantSession(model, cfg.to_op_config())
```

---

## 与 ADR-006 的关系

ADR-006 确认 LSQ 走 Transform 槽位方案：PreScaleTransform 持有 `nn.Parameter`，LayerwiseScaleOptimizer 是外部优化器。本 ADR 在配置层面落实这一架构：

- `transform="prescale"` + `lsq_steps > 0` 表达"使用 prescale + 启用 LSQ 优化"
- `lsq_steps` 字段从用户视角看起来像独立开关，但其实质是 prescale 的优化参数——若 `transform≠"prescale"` 且 `lsq_steps > 0`，`to_op_config()` 抛出 `ValueError`
- prescale 翻译分两步执行（§2.2），第一步 `to_op_config()` 输出 Identity 占位，第二步 `Session.run()` 内创建真正的 PreScaleTransform

---

## 与 ADR-005 的关系

ADR-005（OpQuantConfig 两阶段模型）的 `storage` 字段在 `QuantConfig` 中通过 `scale_storage` 表达。`QuantConfig` 是面向用户的高层配置，`OpQuantConfig` 是内部算子级配置。`QuantConfig.to_op_config()` 完成翻译。

---

## Review 修正记录（2026-05-07）

本 ADR 经 `docs/reviews/2026-05-07-adr008-review.md` 审视后，修正以下内容：

| 问题 | 修正 |
|------|------|
| D1 session↔report 循环依赖 | 明确 lazy import 方案（§5.2） |
| D2 prescale 两步翻译 | 新增 §2.2 两步翻译机制 |
| H1 PerLayerOpt 丢失 | 保留为 `session/_per_layer_opt.py`，在 §1 和 §6 中说明 |
| H2 calibrator 缺失 | `QuantConfig` 增加 `calibrator: str = "mse"`（§2） |
| H3 transform 语义歧义 | 新增 §2.1 transform 规则表 |
| S1 _spec.py 不应持有 observer class | 明确 _spec.py 只用字符串 key，class 解析在 session/ 中（§3.1） |
| S2 outputs="all" 16 key 未定义 | 补充完整 17 key 枚举（§3.1），更正数字 |
| P1 lsq_steps 与 ADR-006 矛盾 | 明确约束：`transform≠"prescale"` 且 `lsq_steps>0` 时 `to_op_config()` 报错（§2、§与 ADR-006 的关系） |
| P2 prescale_granularity 双粒度 | 文档记录默认规则：跟随 `a_granularity`（§2） |
