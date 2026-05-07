# Experiment Pipeline + Viz 重构设计

> **决策日期**: 2026-04-29
> **范围**: `src/pipeline/`（新建）+ `src/viz/`（新建）+ `examples/experiment_format_study.py`（重构）

## 目标

将 `examples/experiment_format_study.py` 中的实验管线逻辑和可视化代码从 examples 中抽离到 `src/` 下，使其可作为库被多个项目复用。

核心设计原则：
1. **IoC（控制反转）**：框架不掌控推理/评估循环，用户代码通过一个 `eval_fn` 回调注入
2. **Config as Data**：搜索空间从代码中剥离为纯数据定义
3. **关注点分离**：pipeline（实验调度）、viz（图表）、session（量化）三层互不依赖

## 模块结构

```
src/
├── pipeline/                    # 实验管线（纯逻辑，不依赖 matplotlib）
│   ├── __init__.py
│   ├── config.py                # 搜索空间解析 + OpQuantConfig 工厂
│   ├── runner.py                # ExperimentRunner（网格搜索调度）
│   ├── protocol.py              # EvalFn 协议类型
│   └── studies/                 # 各实验的搜索空间（纯数据）
│       ├── __init__.py
│       └── format_study.py      # Format Study 搜索空间
│
├── viz/                         # 可视化基础能力（可复用纯函数）
│   ├── __init__.py
│   ├── theme.py                 # 调色板 / 字体 / 样式常量
│   ├── figures.py               # 图表生成（fig1-fig11）
│   ├── tables.py                # 表格生成（table1-table6）
│   └── save.py                  # 文件保存工具
│
├── session.py                   # 已有，不改
├── calibration/                 # 已有，不改
├── analysis/                    # 已有，不改
└── ...

examples/
├── experiment_format_study.py   # 变薄：CLI + 组装 pipeline + viz（~200 行）
└── test_format_study_verification.py
```

**边界约束**：
- `src/pipeline/` → 不 import `matplotlib`、`seaborn`
- `src/viz/` → 只接收数据返回 chart，不依赖 pipeline 或 session
- `examples/` → 唯一同时 import pipeline 和 viz 的模块

## Search Space（Config as Data）

搜索空间用纯 Python dict 定义，不依赖任何框架对象。`config.py` 提供 resolver 将字符串描述符解析为 `OpQuantConfig`。

```python
# src/pipeline/studies/format_study.py

FORMAT_STUDY = {
    "part_a_8bit": {
        "description": "8-bit Format Comparison",
        "configs": {
            "MXINT-8": {"format": "int8",      "granularity": "per_block",  "block_size": 32},
            "MXFP-8":  {"format": "fp8_e4m3",  "granularity": "per_block",  "block_size": 32},
            "INT8-PC": {"format": "int8",      "granularity": "per_channel", "axis": 0},
        },
        "calibrator": "mse",
    },
    "part_b_4bit": {
        "description": "4-bit Format Comparison",
        "configs": {
            "MXINT-4": {"format": "int4",      "granularity": "per_block",  "block_size": 32},
            "MXFP-4":  {"format": "fp4_e2m1",  "granularity": "per_block",  "block_size": 32},
            "INT4-PC": {"format": "int4",      "granularity": "per_channel", "axis": 0},
            "NF4-PC":  {"format": "nf4",       "granularity": "per_channel", "axis": 0, "weight_only": True},
        },
        "calibrator": "mse",
    },
    # ... part_c, part_d, block_sweep 同理
}
```

`config.py` 中的 `resolve_config(desc: dict) -> OpQuantConfig` 负责将每个描述符解析为可用的配置对象。用户也可以跳过字符串描述符，直接在搜索空间中传入 `OpQuantConfig` 实例。

## ExperimentRunner

薄调度层。只做网格搜索的外层循环，**不掌控推理循环**。用户通过一个 `eval_fn` 回调注入所有模型交互逻辑。

```python
# src/pipeline/protocol.py
from typing import Protocol, Dict

class EvalFn(Protocol):
    def __call__(self, model: nn.Module, data: Any) -> Dict[str, float]: ...
```

```python
# src/pipeline/runner.py
class ExperimentRunner:
    """遍历搜索空间中每个 config，执行 quantize→calibrate→analyze→compare"""

    def __init__(self, search_space: dict):
        self._search_space = search_space

    def run(
        self,
        fp32_model: nn.Module,
        *,
        eval_fn: Callable[[nn.Module, Any], Dict[str, float]],
        calib_data: Any = None,
        analyze_data: Any = None,
        eval_data: Any = None,
        observers: list | None = None,
    ) -> Dict[str, dict]:
        """对搜索空间中每个 config 执行完整流程。

        流程：
        1. QuantSession(model, cfg) — 量化模型
        2. session.calibrate() + eval_fn(session, calib_data) — 校准
        3. session.analyze() + eval_fn(session, analyze_data) — 分析
        4. eval_fn(fp32, eval_data) vs eval_fn(session, eval_data) — 对比

        eval_fn 在校准和分析阶段也被调用（触发 forward hooks），
        但只使用其 forward 副作用，返回的 metrics 被忽略。

        Returns:
            {config_name: {"fp32": dict, "quant": dict, "delta": dict, "report": Report}}
        """
```

**关键设计决策**：整个实验只需要一个用户提供的函数 `eval_fn`。它在三个地方被调用：
- **校准阶段**：`eval_fn(session, calib_data)` — 只利用 forward 副作用触发 hooks，忽略返回值
- **分析阶段**：`eval_fn(session, analyze_data)` — 同上
- **评估阶段**：`eval_fn(fp32, eval_data)` 和 `eval_fn(session, eval_data)` — 使用返回值计算 delta

如果 `calib_data` 或 `analyze_data` 为 `None`，跳过对应阶段。

## `src/viz/` — 可视化基础能力

纯函数库，所有函数接收数据返回 matplotlib Figure 或 CSV 路径。不依赖 pipeline、session、或任何实验逻辑。

```python
# src/viz/theme.py
FORMAT_COLORS: dict       # 格式→颜色映射
TRANSFORM_COLORS: dict    # Transform→颜色映射
HIST_COLORS: dict         # 直方图颜色
FALLBACK_CYCLE: list      # 回退调色板

# src/viz/figures.py
def qsnr_bar_chart(results: dict, *, title: str, colors: dict, output_dir: str) -> Figure
def mse_box_plot(results: dict, *, title: str, colors: dict, output_dir: str) -> Figure
def transform_heatmap(part_d: dict, *, colors: dict, output_dir: str) -> Figure
def transform_pie(part_d: dict, *, colors: dict, output_dir: str) -> Figure
def transform_delta(part_d: dict, *, colors: dict, output_dir: str) -> Figure
def histogram_overlay(fp32: Tensor, quant: Tensor, *, ...) -> Figure
def error_vs_distribution(results: dict, *, ...) -> Figure
def layer_type_qsnr(results: dict, *, ...) -> Figure

# src/viz/tables.py
def accuracy_table(results: dict, *, title: str, output_dir: str, filename: str) -> str
def format_comparison_table(results: dict, *, title: str, output_dir: str) -> str

# src/viz/save.py
def save_figure(fig: Figure, output_dir: str, name: str) -> str
def save_table(csv_path: str, output_dir: str) -> str
```

从当前 `examples/experiment_format_study.py` 中抽出的函数（`generate_table_1` ~ `generate_table_6`、`plot_fig1` ~ `plot_fig11`），削去硬编码的 title 和颜色，变为参数化纯函数。

## 用户在自己的项目中使用

```python
from src.pipeline import ExperimentRunner

# 1. 定义搜索空间（可以放项目自己的配置文件里）
MY_STUDY = {
    "8bit": {
        "configs": {
            "int8":  {"format": "int8", "granularity": "per_channel", "axis": 0},
            "fp8":   {"format": "fp8_e4m3", "granularity": "per_channel", "axis": 0},
        },
    },
}

# 2. 一行跑实验（eval_fn 是已有的，不改）
runner = ExperimentRunner(MY_STUDY)
results = runner.run(
    fp32_model=model,
    eval_fn=my_existing_eval_fn,
    calib_data=calib_data,
    eval_data=test_data,
)

# 3. 查看结果
for cfg_name, r in results.items():
    print(f"{cfg_name}: {r['fp32']} → {r['quant']} (delta={r['delta']})")
    r["report"].print_summary()
```

## 重构步骤概要

1. **创建 `src/pipeline/`** — `protocol.py` + `config.py` + `runner.py` + `studies/format_study.py`
2. **创建 `src/viz/`** — 从 `experiment_format_study.py` 抽图表和表格函数
3. **重构 `examples/experiment_format_study.py`** — 删除被抽离的代码，改为 import pipeline + viz，保留 CLI 入口
4. **验证** — 全量测试 + format study 端到端运行
