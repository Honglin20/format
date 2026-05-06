# Design: Pipeline Refactor — format_study 三层职责分离

**Date**: 2026-05-06  
**Branch**: feature/refactor-src  
**Status**: Approved

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
| `lsq_steps` | int | LSQ 优化步数（0 = 不做）|
| `lsq_pot` | bool | LSQ 约束 power-of-two |
| `lsq_lr` | float | LSQ 学习率（默认 1e-3）|

**SmoothQuant 处理**：Runner 自动检测 part 内是否有 `transform: "smoothquant"`，若有则在 part 开始前执行一次 FP32 前向标定，对用户透明。

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
    ) -> Dict[str, List[ExperimentResult]]:
        ...
```

返回 `{part_name → [ExperimentResult, ...]}`，part 内顺序与 config 列表一致。

### 内部执行流

```
for part_name, part_cfg in search_space.items():

    # SmoothQuant 预处理：该 part 有任意 smoothquant config 就执行一次
    sq_ctx = _prepare_sq(fp32_model, calib_data, eval_fn)
        if any(c.get("transform") == "smoothquant" for c in configs) else None

    for cfg_desc in part_cfg["configs"]:
        model_for_run, op_cfg = _resolve(cfg_desc, fp32_model, sq_ctx)
        session = QuantSession(deepcopy(model_for_run), op_cfg,
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
        fp32_m  = eval_fn(fp32_model, eval_data)
        quant_m = eval_fn(session,    eval_data)

        yield ExperimentResult(name, fp32_m, quant_m, ...)
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

```
output_dir/
  results.json               ← 全量序列化（增量写）
  tables/
    {part_name}.csv
  figures/
    {part_name}_qsnr.png     ← qsnr_line_chart
    {part_name}_mse.png      ← mse_box_plot
```

### 与 viz 层的关系

`report.py` 是 viz 函数的薄适配层，viz 函数不感知 `ExperimentResult`：

```python
def _to_qsnr_dict(results: List[ExperimentResult]) -> dict:
    return {r.name: {"qsnr_per_layer": r.qsnr_per_layer} for r in results}

qsnr_line_chart(_to_qsnr_dict(part_results), ...)
```

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

## Example: `examples/07_format_study_random.py`

用随机 tensor 验证核心结论，无需真实数据集。

```python
"""
Format Study — Random Tensor Validation

用随机生成的 tensor 验证核心结论：
  4-bit INT4 FP32 scaling 显著优于 PoT scaling（~5-8 dB QSNR 差距）
  8-bit 格式差异较小

运行：
    python examples/07_format_study_random.py
"""
import torch
import torch.nn as nn
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
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x in data:
            total += model(x).pow(2).mean().item()
    return {"output_norm": total / len(data)}

STUDY_CONFIG = {
    "4bit_fp32_vs_pot": {
        "description": "4-bit FP32 Scale vs PoT Scale (core conclusion)",
        "configs": [
            {"name": "INT4-FP32", "format": "int4", "granularity": "per_channel"},
            {"name": "INT4-PoT",  "format": "int4", "granularity": "per_channel",
             "lsq_steps": 50, "lsq_pot": True},
        ],
    },
    "4bit_formats": {
        "description": "4-bit Format Overview",
        "configs": [
            {"name": "MXINT-4", "format": "int4",     "granularity": "per_block",   "block_size": 32},
            {"name": "INT4-PC", "format": "int4",     "granularity": "per_channel"},
            {"name": "NF4-PC",  "format": "nf4",      "granularity": "per_channel", "weight_only": True},
        ],
    },
    "8bit_formats": {
        "description": "8-bit Format Overview",
        "configs": [
            {"name": "MXINT-8", "format": "int8",     "granularity": "per_block",   "block_size": 32},
            {"name": "INT8-PC", "format": "int8",     "granularity": "per_channel"},
            {"name": "MXFP-8",  "format": "fp8_e4m3", "granularity": "per_block",   "block_size": 32},
        ],
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
| `examples/07_format_study_random.py` | **新建** |

---

## 不变的边界

- `src/viz/` 函数不改，`report.py` 做薄适配
- `src/pipeline/config.py` 不改（`resolve_config` 复用）
- `src/session/` 不改（`QuantSession` 是执行核心）
- 现有测试全部继续通过
