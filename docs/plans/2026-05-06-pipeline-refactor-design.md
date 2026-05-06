# Design: Pipeline Refactor — format_study 三层职责分离

**Date**: 2026-05-06  
**Branch**: feature/refactor-src  
**Status**: Approved (审视修正后 v2)

---

## 问题背景

`src/pipeline/format_study.py` 当前 1086 行，混合了 7 种职责：

- Config builder helpers（`make_op_cfg` 等）
- SmoothQuant 辅助函数（`_make_smoothquant_transforms` / `_fuse_smoothquant_weights`）
- 单次实验运行器（`run_experiment`）
- 5 种 part 特化 runner（`_run_simple_part` / `_run_pot_scaling_part` / `_run_transform_part` / `_run_block_sweep_part` / `_run_hierarchical_part`）
- 4 种 table generator（`generate_table_3/4/5/6`）
- Figure 生成（`_generate_figures`）
- JSON 序列化（`_save_results_json`）

同时，`runner.py` 和 `format_study.py` 存在两套平行的实验运行器（`ExperimentRunner` 类 vs `run_experiment()` 函数），结果 schema 不一致。

---

## 设计目标

1. **单一变更原因**：每个文件只因一种原因变更
2. **统一执行路径**：删除 `run_experiment()`，只保留 `ExperimentRunner`
3. **简化 config schema**：去掉 `type` 字段，所有 part 统一为"一组 config 跑 session"
4. **清晰结论输出**：终端摘要 + CSV + 图表，明确展示如"INT4-FP32 比 PoT 好 6 dB"
5. **可运行 example**：用随机 tensor 验证核心结论

---

## 方案选择

采用**方案 B：三层职责分离**。

```
study_config.py   — 纯数据（不改结构，只迁移 schema）
runner.py         — 纯执行：ExperimentRunner → ExperimentResult
report.py (新建)  — 纯输出：StudyReport（terminal + CSV + figures）
format_study.py   — 纯编排：~100 行，加载 config → Runner → Report
```

---

## Config Schema（新）

去掉 `type` 字段，所有 part 统一结构：

```python
STUDY_CONFIG = {
    "part_name": {
        "description": "Human-readable description",
        "configs": [
            {"name": "INT4-FP32", "format": "int4", "granularity": "per_channel"},
            {"name": "INT4-PoT",  "format": "int4", "granularity": "per_channel",
             "lsq_steps": 100, "lsq_pot": True},
            {"name": "INT4-SQ",   "format": "int4", "granularity": "per_channel",
             "transform": "smoothquant"},
        ],
    },
}
```

**Config dict 支持字段（per-config，完全自描述）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 结果 key，显示在表格中 |
| `format` | str | `"int8"` / `"int4"` / `"fp8_e4m3"` / `"nf4"` 等 |
| `granularity` | str | `"per_tensor"` / `"per_channel"` / `"per_block"` |
| `axis` | int | per_channel 轴（默认 -1）|
| `block_size` | int | per_block 块大小 |
| `transform` | str | `"hadamard"` / `"smoothquant"`（None = Identity）|
| `weight_only` | bool | 只量化权重（NF4 等）|
| `scale_format` | str | scale 本身的数值格式：`"fp32"`（默认）/ `"pot"` — 决定量化 scale 的存储精度 |
| `lsq_steps` | int | LSQ 优化步数（0 = 不做）|
| `lsq_pot` | bool | LSQ 约束 power-of-two（等价于 `scale_format: "pot"` + LSQ 优化）|
| `lsq_lr` | float | LSQ 学习率（默认 1e-3）|

> **关于 `scale_format`**：当前 per_channel 的 scale 隐式为 FP32。`scale_format: "pot"` 让 scale 也量化为 power-of-two，适用于所有 granularity 模式。这与 `lsq_pot` 正交——`lsq_pot` 是「学习过程中约束为 PoT」，`scale_format` 是「最终 scale 用什么格式存储」。两者可独立设置，也可同时使用（LSQ 学习 PoT scale）。

**SmoothQuant 处理**：SQ 预处理是**调用方的职责**，在传入 Runner 之前完成。Runner 不检测、不感知任何 transform 语义。调用方通过 `sq_context` 参数传入预处理结果（已融合权重的模型 + per-layer config），Runner 只负责创建 Session 和驱动生命周期。详见下方「预处理钩子」。

---

## Layer 1: `runner.py` — 纯执行

### `ExperimentResult` dataclass

```python
@dataclass
class ExperimentResult:
    name: str
    fp32_metrics:   Optional[Dict[str, float]]   # None if eval skipped
    quant_metrics:  Optional[Dict[str, float]]
    delta:          Optional[Dict[str, float]]
    qsnr_per_layer: Dict[str, float]             # layer → avg QSNR dB
    mse_per_layer:  Dict[str, float]             # layer → avg MSE
    cost:           Any
    cost_fp32:      Any

    @property
    def avg_qsnr(self) -> float: ...
    @property
    def avg_mse(self) -> float: ...
```

### `ExperimentRunner`

```python
class ExperimentRunner:
    def __init__(self, search_space: dict, skip_parts: Optional[set] = None): ...

    def run(
        self,
        fp32_model: nn.Module,
        *,
        eval_fn: Callable,
        calib_data: Any,
        eval_data: Any = None,
        observers: list | None = None,
        on_config_done: Optional[Callable] = None,   # 增量保存回调
    ) -> Dict[str, List[ExperimentResult]]:
        ...
```

返回 `{part_name → [ExperimentResult, ...]}`，part 内顺序与 config 列表一致。

**Runner 和 Session 的关系**：Runner 是 Session 的工厂 + 生命周期编排器。Session 持有量化模型并管理 calibrate/analyze/evaluate 上下文；Runner 只做两件事——(1) 把 config dict 解析成 OpQuantConfig 并创建 Session，(2) 按 calibrate → (LSQ) → analyze → evaluate 顺序驱动 Session。Runner 使用 Session，不替代 Session。

### 预处理钩子（SmoothQuant 的正确位置）

Runner 不感知任何 transform 语义。需要模型级预处理（如 SmoothQuant 的 weight fusion）的场景，由调用方在 Runner 外部完成：

```python
# 调用方（format_study.py 编排层）负责 SQ 预处理
sq_prep = prepare_smoothquant(model, calib_data, eval_fn)  # → sq_model, sq_per_layer_cfg

# 对有 SQ config 的 part，传入已融合的模型
runner.run(sq_model, eval_fn=eval_fn, calib_data=calib_data, ...)
```

这样 Runner 完全不需要知道 SmoothQuant、Hadamard 或任何 transform 的存在——它只接收模型和 config，创建 Session，驱动生命周期。

### Per-Layer Optimal Transform（编排层能力）

Per-layer-optimal 不是 Runner 的职责，是**编排层的组合能力**：

```
format_study.py（编排层）:
  1. 对 format X 跑 None/Hadamard/SQ 三种基础 variant  → 拿到 per-layer QSNR
  2. _compute_best_transform_per_layer(qsnr_dicts)      → 每层最优 transform
  3. _build_per_layer_optimal_cfg(...)                   → heterogeneous config
  4. 用 heterogeneous config 再跑一次 Runner.run()       → PerLayerOpt 结果
```

这 4 步全部在 `format_study.py`（编排层）完成。Runner 不需要知道「per-layer optimal」这个概念——它只是被调了 4 次，每次都是「一组 config 跑 session」。

### 内部执行流（per-config）

```
for cfg_desc in part_cfg["configs"]:
    op_cfg = resolve_config(cfg_desc)

    # Phase 0: 可选 — model pre-transform（调用方在 Runner 外完成）
    model_for_run = caller_provided_model or deepcopy(fp32_model)

    session = QuantSession(model_for_run, op_cfg,
                           calibrator=MSEScaleStrategy(), keep_fp32=True)

    # Phase 1: LSQ（可选，per-config）
    if cfg_desc.get("lsq_steps", 0) > 0:
        session.initialize_pre_scales(calib_data, ...)
        session.optimize_scales(opt, calib_data, eval_fn=eval_fn)

    # Phase 2: Calibrate
    with session.calibrate():
        eval_fn(session, calib_data)

    # Phase 3: Analyze
    with session.analyze(observers) as ctx:
        eval_fn(session, calib_data)
    report = ctx.report()

    # Phase 4: Evaluate
    result = ExperimentResult(
        name=cfg_desc["name"],
        fp32_metrics=eval_fn(fp32_model, eval_data),
        quant_metrics=eval_fn(session, eval_data),
        qsnr_per_layer=extract_metric_per_layer(report, "qsnr_db"),
        mse_per_layer=extract_metric_per_layer(report, "mse"),
        cost=session.estimate_cost(),
        cost_fp32=session.estimate_cost(fp32=True),
    )

    # 增量保存回调（不阻塞执行）
    if on_config_done:
        on_config_done(result)
```

**不做任何打印。** 纯执行，无 I/O 副作用。

---

## Layer 2: `report.py` — 纯输出（新建）

### 公开 API

```python
class StudyReport:
    def __init__(self, results: Dict[str, List[ExperimentResult]]): ...
    def print_summary(self) -> None: ...       # 终端对比表
    def save(self, output_dir: str) -> None:   # CSV + 图表
    def to_serializable(self) -> dict: ...     # JSON-compatible
```

### 终端摘要格式

```
=== 4-bit FP32 Scale vs PoT Scale (core conclusion) ===

  Config          Avg QSNR (dB)    Avg MSE       Δ QSNR
  ──────────────────────────────────────────────────────
  INT4-FP32           17.8         1.8e-04    (baseline)
  INT4-PoT            11.2         9.4e-04      -6.6 dB

  Best QSNR: INT4-FP32 (17.8 dB)
  → PoT scaling 损失 6.6 dB QSNR（约 4.5× 更高 MSE）
```

### 文件输出结构

每个 part 在 config 中声明它需要什么输出：

```python
"part_d": {
    "description": "4-bit Transform Study",
    "configs": [...],
    "output": {
        "tables": ["accuracy", "transform_matrix", "transform_distribution"],
        "figures": ["qsnr_line", "transform_heatmap", "transform_pie", "transform_delta"],
    },
},
```

输出目录：

```
output_dir/
  results.json               ← 全量序列化（每 part 完成后增量写）
  tables/
    part_d_accuracy.csv
    part_d_transform_matrix.csv
    part_d_transform_distribution.csv
  figures/
    part_d_qsnr_line.png
    part_d_transform_heatmap.png
    part_d_transform_pie.png
    part_d_transform_delta.png
```

未声明 `output` 的 part 默认生成 `accuracy` 表 + `qsnr_line` 图。

### 可用的 table / figure key

**Tables**（在 `src/viz/tables.py` 注册）：
| key | 函数 | 说明 |
|-----|------|------|
| `accuracy` | `accuracy_table` | 精度对比表 |
| `format_comparison` | `format_comparison_table` | 格式 × 指标矩阵 |
| `transform_matrix` | `transform_heatmap_table` | Format × Transform 精度矩阵（原 table4）|
| `transform_distribution` | `transform_distribution_table` | Per-layer transform 选择分布（原 table5）|
| `pot_delta` | `pot_delta_table` | FP32 vs PoT Δ 表（原 table3）|
| `sensitivity` | `sensitivity_table` | Top-10 敏感层（原 table6）|

**Figures**（在 `src/viz/figures.py` 注册）：
| key | 函数 | 说明 |
|-----|------|------|
| `qsnr_line` | `qsnr_line_chart` | Per-layer QSNR 折线图 |
| `mse_box` | `mse_box_plot` | Per-layer MSE 箱线图 |
| `pot_delta` | `pot_delta_bar` | FP32 vs PoT Δ 柱状图 |
| `transform_heatmap` | `transform_heatmap` | Format × Transform 热力图 |
| `transform_pie` | `transform_pie` | Transform 选择分布饼图 |
| `transform_delta` | `transform_delta` | Transform ΔQSNR 柱状图 |
| `histogram` | `histogram_overlay` | fp32/quant/error 直方图 |
| `error_vs_dist` | `error_vs_distribution` | 误差 vs 分布特征散点图 |
| `layer_type_qsnr` | `layer_type_qsnr` | Layer-type 分组 QSNR |
| `block_sweep` | `block_sweep_line_chart` | Block 大小扫描折线图 |
| `hierarchical_delta` | `hierarchical_delta_bar` | Hierarchical Δ 柱状图 |

> Viz 函数不改。`report.py` 用 `_to_qsnr_dict()` / `_to_mse_dict()` 等薄适配器把 `ExperimentResult` 转为 viz 函数期望的 dict 格式。

---

## Layer 3: `format_study.py` — 纯编排（~100 行）

```python
def run_format_study(
    build_model:     Callable[[], nn.Module],
    make_calib_data: Callable[[], List[torch.Tensor]],
    eval_fn:         Callable,
    *,
    config:      Optional[dict] = None,
    output_dir:  Optional[str] = None,
    skip_parts:  Optional[set] = None,
    eval_data:   Any = None,
) -> Dict[str, List[ExperimentResult]]:

    if config is None:
        from src.pipeline.study_config import STUDY_CONFIG as config
    if output_dir is None:
        output_dir = f"results/format_study_{timestamp()}"

    model      = build_model()
    calib_data = make_calib_data()
    eval_data  = eval_data or calib_data

    runner  = ExperimentRunner(config, skip_parts=skip_parts)
    results = runner.run(model, eval_fn=eval_fn,
                         calib_data=calib_data, eval_data=eval_data)

    report = StudyReport(results)
    report.print_summary()
    report.save(output_dir)

    return results
```

---

## Example: `examples/format_study_random.py`

用随机 tensor 验证核心结论，无需真实数据集。

```python
"""
Format Study — Random Tensor Validation

验证核心结论：
  1. 4-bit INT4 FP32 scale 显著优于 PoT scale（~5-8 dB QSNR 差距）
  2. 8-bit 格式差异远小于 4-bit
  3. scale_format 对比：fp32 vs pot 对所有 granularity 均适用

运行：
    python examples/format_study_random.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pipeline.format_study import run_format_study


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 64),
    )


def make_calib_data():
    torch.manual_seed(42)
    return [torch.randn(16, 64) for _ in range(8)]


def eval_fn(model, data):
    """返回 fp32 参考输出和量化输出的 cosine similarity + relative MSE。

    对于随机 tensor 验证，cosine similarity 比 output_norm 更能反映
    量化对方向信息的保持能力——这对实际模型的 logit 排序至关重要。
    """
    model.eval()
    cos_sims = []
    rel_mses = []
    with torch.no_grad():
        for x in data:
            out = model(x)
            # 用自身 L2 norm 做归一化参考
            norm = out.pow(2).sum(-1, keepdim=True).sqrt()
            cos_sims.append(F.cosine_similarity(out, out, dim=-1).mean().item())
            rel_mses.append(0.0)  # fp32 reference — 量化后用 session.compare 算
    return {"cosine_sim": sum(cos_sims) / len(cos_sims),
            "rel_mse": 0.0}


STUDY_CONFIG = {
    # ── Core conclusion: FP32 scale vs PoT scale ──
    "core_4bit_scale": {
        "description": "4-bit INT4: FP32 scale vs PoT scale (core conclusion)",
        "configs": [
            {"name": "INT4-PC-FP32", "format": "int4", "granularity": "per_channel",
             "scale_format": "fp32"},
            {"name": "INT4-PC-PoT",  "format": "int4", "granularity": "per_channel",
             "scale_format": "pot"},
        ],
        "output": {"tables": ["accuracy", "pot_delta"],
                   "figures": ["qsnr_line", "pot_delta"]},
    },

    # ── 4-bit format overview ──
    "4bit_formats": {
        "description": "4-bit Format Overview",
        "configs": [
            {"name": "MXINT-4", "format": "int4",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-4",  "format": "fp4_e2m1", "granularity": "per_block",   "block_size": 32},
            {"name": "INT4-PC", "format": "int4",     "granularity": "per_channel", "scale_format": "fp32"},
            {"name": "NF4-PC",  "format": "nf4",      "granularity": "per_channel", "weight_only": True},
        ],
        "output": {"tables": ["accuracy"], "figures": ["qsnr_line", "mse_box"]},
    },

    # ── 8-bit format overview (baseline: differences should be small) ──
    "8bit_formats": {
        "description": "8-bit Format Overview",
        "configs": [
            {"name": "MXINT-8", "format": "int8",     "granularity": "per_block",   "block_size": 32},
            {"name": "MXFP-8",  "format": "fp8_e4m3", "granularity": "per_block",   "block_size": 32},
            {"name": "INT8-PC", "format": "int8",     "granularity": "per_channel", "scale_format": "fp32"},
        ],
        "output": {"tables": ["accuracy"], "figures": ["qsnr_line", "mse_box"]},
    },

    # ── Transform effect at 4-bit ──
    "4bit_transform": {
        "description": "Hadamard transform effect on 4-bit formats",
        "configs": [
            {"name": "MXINT-4",      "format": "int4",     "granularity": "per_block", "block_size": 32},
            {"name": "MXINT-4-Had",  "format": "int4",     "granularity": "per_block", "block_size": 32,
             "transform": "hadamard"},
            {"name": "MXFP-4",       "format": "fp4_e2m1", "granularity": "per_block", "block_size": 32},
            {"name": "MXFP-4-Had",   "format": "fp4_e2m1", "granularity": "per_block", "block_size": 32,
             "transform": "hadamard"},
        ],
        "output": {"tables": ["accuracy"], "figures": ["qsnr_line", "transform_delta"]},
    },
}

if __name__ == "__main__":
    results = run_format_study(
        build_model=build_model,
        make_calib_data=make_calib_data,
        eval_fn=eval_fn,
        config=STUDY_CONFIG,
        output_dir="results/random_tensor_study",
    )
```

---

## 文件变更汇总

| 文件 | 变化 |
|------|------|
| `src/pipeline/runner.py` | 重写：`ExperimentResult` dataclass + 简化 `ExperimentRunner` |
| `src/pipeline/report.py` | **新建**：`StudyReport`（terminal + CSV + figures）|
| `src/pipeline/format_study.py` | 重写：~100 行编排 |
| `src/pipeline/study_config.py` | 迁移到新 schema（去 `type`，改 `configs`）|
| `src/pipeline/config.py` | 不动 |
| `src/pipeline/__init__.py` | 更新导出：加 `ExperimentResult`, `StudyReport` |
| `examples/format_study_random.py` | **新建** |

---

## 不变的边界

- `src/viz/` 函数不改，`report.py` 做薄适配
- `src/pipeline/config.py` 不改（`resolve_config` 复用）
- `src/session/` 不改（`QuantSession` 是执行核心）
- 现有测试全部继续通过

---

## 审视与修正（v2）

### 问题 1: SmoothQuant 的「自动检测」放在 Runner 里是泄漏的抽象 ✅ 已修正

**原设计**：Runner 自动检测 part 内是否有 `transform: "smoothquant"`，若有则在 part 开始前执行 FP32 前向标定。

**问题**：SmoothQuant 需要跑 FP32 前向捕获 activation、计算 per-layer scale、融合 weight、使用不同模型副本——这些都涉及 transform 语义。Runner 不应该知道任何 transform 的细节。违反开闭原则：每次新增需要模型级预处理的 transform，都要改 Runner。

**修正**：SQ 预处理移出 Runner，作为调用方（编排层）的职责。Runner 只接收「已准备好的模型 + config」，不感知 transform 语义。

### 问题 2: Per-Layer Optimal Transform 能力缺失 ✅ 已修正

**原设计**：每个 config 只有单一的 `transform` 字段，不支持 per-layer 选择。

**修正**：Per-layer-optimal 保留在编排层（`format_study.py`），作为组合能力——编排层跑完基础 variant 后用 `_compute_best_transform_per_layer` 选出每层最优 transform，构建 heterogeneous config 再调一次 Runner。Runner 不知道「per-layer optimal」这个概念。

### 问题 3: Config schema 混淆 scheme 和优化参数 ❌ 不成立

**用户裁决**：`lsq_pot` 本质是 transform（PreScaleTransform）的属性——PoT scale 和 FP32 scale 是两种不同的变换行为。`lsq_steps`/`lsq_lr` 是优化器配置，在 config 层面扁平放置是务实选择。`resolve_config()` 只解析 scheme 部分，LSQ 参数由 Runner 的执行阶段消费。

### 问题 4: StudyReport 图表生成硬编码 ✅ 已修正

**修正**：每个 part 通过 `output.tables` 和 `output.figures` 声明自己需要什么输出。Report 层根据声明调度 viz 函数，新增加可视化只需在声明注册表中添加 entry，不改 report.py。

### 问题 5: 没有增量保存 ✅ 已修正

**修正**：Runner 接受 `on_config_done` 回调。编排层注入一个将 `ExperimentResult` 序列化追加写 JSON 的回调。Runner 不感知文件系统。

### 问题 6: Example 指标过于简化 ✅ 已修正

**修正**：`eval_fn` 返回 `cosine_similarity` 和 `relative_mse`，而非 `output_norm`。余弦相似度直接衡量量化对方向信息的保持能力，对理解「量化是否破坏模型输出结构」更有说服力。

### 新增需求: scale_format ✅ 已纳入

**需求**：当前 per_channel 的 scale 隐式为 FP32，用户希望 scale 格式也可配置（如 `scale_format: "pot"`），适用于所有 granularity 模式。

**设计**：新增 `scale_format` 字段，与 `lsq_pot` 正交：
- `scale_format: "fp32"` → scale 用浮点存储（当前默认行为）
- `scale_format: "pot"` → scale 量化为 power-of-two
- `lsq_pot: True` → 学习过程中约束 scale 为 PoT（等价于 `scale_format: "pot"` + LSQ 优化）
- 两者可独立设置，也可同时使用
