# Session 统一入口重构 — 实施计划

> **设计文档**: `docs/architecture/008-session-refactor.md`（2026-05-07 review 修正版）
> **审视**: `docs/reviews/2026-05-07-adr008-review.md`
> **日期**: 2026-05-07
> **分支**: `feature/refactor-src`

---

## 目标

1. 消除 `pipeline/` 包，功能拆分到 `session/`、`report/`、`scheme/`、`transform/`
2. 新增 `QuantConfig` dataclass 作为用户唯一配置入口
3. 新增 `Session`（执行单元）+ `Study`（聚合层），职责清晰分离
4. 新增 `report/` 包，实现 output-driven 输出系统
5. 修正依赖方向：`report → session → capabilities`（模块级严格单向）
6. 随重构修复 5 个代码缺陷（C1-C5）

---

## Task 分解

### Task 1: `QuantConfig` dataclass

**文件**: `src/session/_config.py`（新建，~160 行）

```
QuantConfig dataclass:
  字段（完整列表）:
    name: str
    w_format: str, w_granularity: str, w_block_size: int|None
    a_format: str|None, a_granularity: str, a_block_size: int|None
    transform: str, sq_alpha: float
    prescale_init: str, prescale_pot: bool, prescale_granularity: str
    lsq_steps: int, lsq_lr: float
    scale_storage: str
    calibrator: str                          ← H2: 从 review 新增
    weight_only: bool

  方法:
    to_op_config() → OpQuantConfig
      - 实现 §2.1 transform 规则表（hadamard→both, smoothquant→activation only, prescale→Identity 占位）
      - transform="prescale" 且 lsq_steps>0 时：检查 transform=="prescale"，否则抛 ValueError
      - calibrator 字段翻译为对应的 ScaleStrategy 实例
      - (prescale 的 PreScaleTransform 不在此创建，见 §2.2 两步机制)

    _resolve_granularity(granularity_str, block_size, axis=-1) → GranularitySpec
      (从 pipeline/config.py _resolve_granularity 迁移)

  TypeGuard:
    - w_format/a_format: str, 非法值 FormatBase.from_str 会抛出
    - w_granularity/a_granularity: str, 只接受 per_tensor|per_channel|per_block
    - transform: str, 只接受 none|hadamard|smoothquant|prescale
    - calibrator: str, 只接受 mse|max|percentile|kl
    - scale_storage: str, 只接受 fp32|pot
    - lsq_steps: int >= 0
    - weight_only=True 时 a_format 必须为 None（互斥）
    - transform≠"prescale" 且 lsq_steps>0 → ValueError

测试: test_quant_config.py (~28 tests)
  - 默认值正确
  - w_format + a_format → OpQuantConfig（wXaY mixed-precision）
  - transform 规则表每种组合 (none/hadamard/smoothquant/prescale × weight/act)
  - prescale: to_op_config() 输出 IdentityTransform 占位
  - transform="smoothquant" 且 lsq_steps>0 → ValueError
  - calibrator="percentile" → PercentileScaleStrategy 实例
  - scale_storage="pot" → scale_format 传递
  - weight_only=True + a_format="int8" → ValueError
  - 非法 granularity/transform/calibrator → ValueError
```

**涉及文件:**
- NEW: `src/session/_config.py`
- MODIFY: `src/session/__init__.py`（导出 QuantConfig）
- MODIFY: `src/scheme/op_config.py`（+ `from_descriptor()` 类方法，吸收 pipeline/config.py resolve_config）

---

### Task 2: SmoothQuant helpers 迁移

**文件**: `src/transform/smooth_quant.py`（追加 ~80 行）

```
新增:
  SmoothQuantTransform.from_model_calibration(fp32_model, calib_data, *, eval_fn=None)
    → Dict[str, SmoothQuantTransform]
    (从 pipeline/format_study.py _make_smoothquant_transforms 迁移)

  fuse_smoothquant_weights(fp32_model, sq_transforms, *, layer_names=None)
    → nn.Module (deep copy with fused weights)
    (从 pipeline/format_study.py _fuse_smoothquant_weights 迁移)

测试: test_smooth_quant_helpers.py (~12 tests)
  - from_model_calibration 在 Linear/Conv2d 模型上
  - fuse_weights 不改变 fp32 输出
  - empty model / missing weight / no activations
```

**涉及文件:**
- MODIFY: `src/transform/smooth_quant.py`
- MODIFY: `src/transform/__init__.py`（导出新函数）

---

### Task 3: `report/` 包 — Output-Driven 输出系统

**文件:** 全部新建

```
src/report/
├── __init__.py              # SessionReport, StudyReport
├── _spec.py                 # _OUTPUT_SPEC 映射表 (~100 行，S1: 只含字符串 key)
├── _registry.py             # viz 函数 lazy 注册表 (~60 行，从 pipeline/report.py 迁移)
├── _converters.py           # SessionResult → viz dict 转换器 (~70 行)
├── _session_report.py       # SessionReport 类 (~80 行)
├── _study_report.py         # StudyReport 类 (~160 行，吸收 pipeline/report.py)
```

#### _spec.py 设计（注意 S1 约束）

**只暴露字符串 key 和 boolean 标志，不持有 observer class 引用，不 import `analysis.observers`**：

```python
# src/report/_spec.py — 零 analysis/observer 依赖

_OUTPUT_SPEC = {
    "accuracy":         {"observers": [],              "needs_eval": True},
    "sensitivity":      {"observers": ["qsnr"],        "needs_eval": True},
    "pot_delta":        {"observers": [],              "needs_eval": True},
    "transform_matrix": {"observers": ["qsnr"],        "needs_eval": True},
    "transform_dist":   {"observers": ["qsnr"],        "needs_eval": True},
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
    "cost":             {"observers": [],              "needs_eval": False, "needs_cost": True},
}

PRESETS = {
    "default": ["accuracy", "qsnr"],
    "all": list(_OUTPUT_SPEC.keys()),
}


def resolve_outputs(output_keys: list[str]) -> tuple[set[str], bool, bool]:
    """返回 (observer_keys, needs_eval, needs_cost)"""
    obs = set()
    needs_eval = False
    needs_cost = False
    for key in output_keys:
        spec = _OUTPUT_SPEC[key]
        obs.update(spec["observers"])
        needs_eval = needs_eval or spec.get("needs_eval", False)
        needs_cost = needs_cost or spec.get("needs_cost", False)
    return obs, needs_eval, needs_cost
```

**observer key → class 解析**在 `session/_session.py` 中（执行层），不在 `report/` 中：

```python
# src/session/_session.py
_OBSERVER_MAP = {
    "qsnr": QSNRObserver,
    "mse": MSEObserver,
    "histogram": HistogramObserver,
    "distribution": DistributionObserver,
}
```

#### _converters.py

```
核心函数:
  session_results_to_viz_dict(results: List[SessionResult]) → dict
    (吸收 pipeline/report.py _results_to_viz_dict)

  session_results_to_nested_viz_dict(results: List[SessionResult], configs: List[QuantConfig]) → dict
    (吸收 pipeline/report.py _results_to_nested_viz_dict)

  extract_metric_per_layer(report, metric) → Dict[str, float]
    (从 pipeline/runner.py extract_metric_per_layer 迁移，修复 C5)
```

#### _study_report.py

**StudyReport 类：**

```
__init__(results: Dict[str, List[SessionResult]])
  - SessionResult 从 session/ 导入（模块级 import，方向 report → session ✓）

print_summary() → None
to_serializable() → dict
save(output_dir: str, config: Optional[dict] = None) → None
  - 调用 _registry 中注册的 table/figure 生成器
  - 统一 viz 函数签名为 fn(data, output_dir, **kwargs)，删除 _call_table 的 TypeError fallback（修复 C4）

from_file(path) → StudyReport (classmethod)
  - 从 results.json 重新加载
```

**测试:**
- test_report_spec.py (~15 tests): resolve_outputs 各种组合，PRESETS 完整
- test_study_report.py (~18 tests): save / from_file / print_summary / to_serializable

**涉及文件:**
- NEW: `src/report/`（6 个文件）
- REMOVE: `src/pipeline/report.py`

---

### Task 4: `Session` 执行单元

**文件**: `src/session/_session.py`（新建，~220 行，吸收 pipeline/runner.py 逻辑 + C2 修复）

```
Session.__init__(model, config: QuantConfig, *, keep_fp32=True)
  - calibrator 从 config.calibrator 读取，不暴露独立参数（H2）
  - 不提前调用 to_op_config()；在 run() 中按需执行

Session.run(calib_data, *, eval_data=None, eval_fn=None,
            outputs: str|list[str] = "default") → SessionResult

  内部流程:
    1. 解析 outputs → (observer_keys, needs_eval, needs_cost)
       (调用 report/_spec.py resolve_outputs)
    2. observer_keys → observer_classes (本地 _OBSERVER_MAP 解析)
    3. [if transform="smoothquant"]:
         sq_transforms = SmoothQuantTransform.from_model_calibration(...)
         model = fuse_smoothquant_weights(model, sq_transforms)
         → per-layer OpQuantConfig (SQ 在 input role, weight=Identity)
    4. op_cfg = config.to_op_config()
       (若 configs 含 per-layer SQ info，在此合并)
    5. qs = QuantSession(model, op_cfg, keep_fp32=keep_fp32)
    6. [if transform="prescale"]:
         ── 两步翻译 Step 2 ──
         qs.initialize_pre_scales(calib_data, ...)
         (创建 PreScaleTransform，替换 cfg 中的 Identity)
         [if lsq_steps > 0]:
           opt = LayerwiseScaleOptimizer(num_steps=lsq_steps, ...)
           qs.optimize_scales(opt, calib_data, eval_fn=eval_fn)
    7. Calibrate: with qs.calibrate(): eval_fn(qs, calib_data)
    8. [if observer_classes]: Analyze with 推导出的 observers
    9. [if needs_eval]: Evaluate
       - fp32_metrics = eval_fn(qs.fp32_model, eval_data)   ← C2: 直接用 qs.fp32_model
       - quant_metrics = eval_fn(qs, eval_data)
       - delta = {...}
   10. [if needs_cost]: cost = qs.estimate_cost()
   11. return SessionResult(...)
```

**SessionResult dataclass**（定义在 `session/_session.py`，与 Session 同居）:

```
@dataclass
class SessionResult:
    name: str
    config: QuantConfig
    fp32_metrics: dict | None
    quant_metrics: dict | None
    delta: dict | None
    qsnr_per_layer: dict[str, float]
    mse_per_layer: dict[str, float]
    observers_data: dict           # raw observer output per layer
    cost: Any = None
    cost_fp32: Any = None
    sq_transforms: dict | None = None  # 缓存，供 PerLayerOpt 复用 (修复 C1)
```

**测试:** test_session.py (~24 tests)
  - 单 config → SessionResult（smoke test）
  - outputs=["histogram"] → 只创建 HistogramObserver
  - transform="prescale" + lsq_steps=0（静态 prescale）
  - transform="prescale" + lsq_steps=100（LSQ 路径）
  - transform="smoothquant" 路径（SQ 标定 + 融合 + per-layer cfg）
  - transform≠"prescale" 且 lsq_steps>0 → ValueError
  - eval_data 默认 = calib_data
  - keep_fp32=False 时 evaluate 报错
  - outputs="all" → 17 个 key 全部覆盖

**涉及文件:**
- NEW: `src/session/_session.py`（Session + SessionResult）
- MODIFY: `src/session/__init__.py`（导出 Session, SessionResult）
- REMOVE: `src/pipeline/runner.py`（ExperimentRunner → Session 吸收）
- REMOVE: `src/pipeline/config.py`（→ QuantConfig 吸收）
- RENAME: `src/session/_session.py` → `src/session/_quant.py`（QuantSession，减名冲突）

---

### Task 5: `Study` 聚合层 + `per_layer_optimal` 工具

**文件 5a**: `src/session/_study.py`（新建，~70 行）

```
Study.__init__(configs: list[QuantConfig], *, model: nn.Module)
Study.run(calib_data, *, eval_data=None, eval_fn=None,
          outputs="default", model_factory=None) → StudyReport

  内部:
    results: Dict[str, List[SessionResult]] = {}
    for cfg in self._configs:
        model = model_factory(cfg) if model_factory else copy.deepcopy(self._model)
        session = Session(model, cfg)               ← 创建 Session
        result = session.run(calib_data, ...)        ← 委托执行
        results[cfg.name] = [result]
    from src.report import StudyReport               ← lazy import（D1 方案）
    return StudyReport(results)
```

**设计关键:**
- Study 内部只做 for 循环 + Session 创建。零量化逻辑、零 transform 感知。
- `StudyReport` 通过 lazy import 导入（解决 D1 循环依赖）
- 模块级 `session/` 不依赖 `report/`

**文件 5b**: `src/session/_per_layer_opt.py`（新建，~90 行，H1）

```
per_layer_optimal(
    part_results: List[SessionResult],
    calib_data,
    fp32_model,                    # 用于 SQ 重标定
    eval_fn,
    *,
    sq_transforms: dict | None = None,  # 缓存复用（C1 修复）
) → SessionResult:
    """按层选最优 transform 并重跑。"""
    1. 按 format 分组结果（None/Hadamard/SmoothQuant 变体）
    2. 对每组: _compute_best_transform_per_layer(variant_qsnr)
    3. 构建 per-layer OpQuantConfig（按层选最优 transform）
    4. 对 SQ-winning layers: 重新 fuse_weights（使用缓存的 sq_transforms）
    5. 创建 Session(opt_model, per_layer_cfg).run(...)
    6. return SessionResult

(从 pipeline/format_study.py L540-643 迁移，80 行)
```

**测试:**
- test_study.py (~12 tests): 多 config → StudyReport，model_factory，outputs 传递，空 configs
- test_per_layer_opt.py (~8 tests): 按层选最优，SQ 缓存复用，边界情况

**涉及文件:**
- NEW: `src/session/_study.py`
- NEW: `src/session/_per_layer_opt.py`
- MODIFY: `src/session/__init__.py`（导出 Study, per_layer_optimal）
- REMOVE: `src/pipeline/format_study.py`

---

### Task 6: 清理、兼容、代码缺陷修复

#### 6a. 删除 pipeline/

```
rm -rf src/pipeline/
```

`src/pipeline/study_config.py` → `src/session/study_config.py`（保留作为预设数据）

#### 6b. 代码缺陷修复

| 缺陷 | 位置 | 修复 |
|------|------|------|
| C1 SQ 重复计算 | 原 `format_study.py:398,582` | `SessionResult.sq_transforms` 字段缓存 SQ 结果；`per_layer_optimal()` 接收 `sq_transforms` 参数复用 |
| C2 fp32 deepcopy 冗余 | 原 `runner.py:136` | `Session.run()` 使用 `qs.fp32_model`，不做额外 deepcopy |
| C3 `_resolve_transform` 不完整 | 原 `config.py:43-59` | 整个函数被 `QuantConfig.to_op_config()` 的 §2.1 规则表替代，不再存在 |
| C4 `_call_table` TypeError fallback | 原 `report.py:112` | 统一 viz 函数签名为 `fn(data, output_dir, **kwargs)`，删除 try/except TypeError |
| C5 `extract_metric_per_layer` 鸭子类型 | 原 `runner.py:40` | 迁移到 `report/_converters.py`，`Report.to_dataframe()` 保证返回类型稳定 |

#### 6c. 测试更新

```
- 修复 pipeline 相关测试的 import（→ session/report/transform）
- 测试逻辑不变，只改 import 路径
- 全量: pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
```

#### 6d. 示例更新

```
- examples/ 中 from src.pipeline import ... → 更新为新路径
```

---

### Task 7: 文档更新

- `CLAUDE.md`:
  - 工具层描述更正：`calibration analysis cost onnx`（移除 pipeline，移除 viz 的工具定位）
  - 新增 `report/` 和 `viz/` 的输出层描述
  - 快速参考表更新
- `docs/architecture/INDEX.md` → 已添加（Task 实施前完成也可）
- `docs/status/CURRENT.md` → 更新进度

---

## 执行顺序

```
Task 1 (QuantConfig) ──┐
                       ├──→ Task 3 (report/) ──→ Task 4 (Session) ──→ Task 5a (Study)
Task 2 (SQ helpers) ───┘                                              Task 5b (per_layer_opt)
                                                                          │
                                                                          ▼
                                                                     Task 6 (清理+C1-C5)
                                                                          │
                                                                          ▼
                                                                     Task 7 (文档)
```

Task 1 + 2 可并行。
Task 3 依赖 Task 1（SessionResult 类型定义）。
Task 4 依赖 Task 1（QuantConfig）+ Task 3（resolve_outputs）。
Task 5 依赖 Task 4（Session + SessionResult）。
Task 6 依赖 Task 5。
Task 7 在所有 code 完成后。

---

## 风险

| 风险 | 缓解 |
|------|------|
| pipeline 测试大量断裂 | 先读所有 pipeline 测试，评估影响面；测试逻辑不变，只改 import |
| QuantConfig 字段遗漏 | 对照 STUDY_CONFIG 所有 key + pipeline/config.py resolve_config 参数 + runner.py ExperimentResult 字段逐一检查 |
| D1 lazy import 边缘情况 | 仅 `Study.run()` 一处使用，标准 Python 模式，低风险 |
| D2 prescale 两步机制回归 | Task 4 测试覆盖 prescale 路径（静态 + LSQ） |
| 向后兼容 | `feature/refactor-src` 未发布，破坏性变更可接受 |
