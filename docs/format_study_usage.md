# Format Study 使用指南

系统化量化格式精度研究：8-bit 对比、4-bit 对比、FP32 vs PoT scaling、Transform 效果，产出 6 张表格和 11 张图表。

## 程序化调用

```python
from src.pipeline.format_study import run_format_study

results = run_format_study(
    build_model=my_build_fn,
    make_calib_data=my_calib_fn,
    make_eval_loader=my_loader_fn,
    eval_fn=my_eval_fn,
)
```

## 需要提供的四个函数

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def build_model() -> nn.Module:
    """每次调用返回一个新的 FP32 模型实例。"""
    return MyModel()

def make_calib_data() -> list[torch.Tensor]:
    """返回校准数据列表，每项为一个 batch tensor。"""
    return [torch.randn(16, 128) for _ in range(16)]

def make_eval_loader() -> DataLoader:
    """返回评估 DataLoader，每个 batch yield (input, label) tuple。"""
    ...

def eval_fn(model: nn.Module, dataloader: DataLoader) -> dict[str, float]:
    """运行推理，返回指标字典，例如 {"accuracy": 0.92}。"""
    ...
```

参见 `examples/experiment_format_study.py` 查看完整的默认实现。

## 修改搜索空间

编辑 `src/pipeline/studies/format_study.py` 中的 `FORMAT_STUDY` dict：

```python
FORMAT_STUDY = {
    "part_a_8bit": {
        "configs": {
            # 格式名 → descriptor dict
            "MXINT-8": {"format": "int8",     "granularity": "per_block",   "block_size": 32},
            "INT8-PC":  {"format": "int8",     "granularity": "per_channel", "axis": 0},
            "NF4-PC":   {"format": "nf4",      "granularity": "per_channel", "axis": 0, "weight_only": True},
            # 添加 Hadamard transform：
            "INT8-Had": {"format": "int8",     "granularity": "per_channel", "axis": 0, "transform": "hadamard"},
        },
    },
    # 添加新的 part：
    "my_custom_part": {
        "configs": {
            "fp8-blk16": {"format": "fp8_e4m3", "granularity": "per_block", "block_size": 16},
        },
    },
}
```

descriptor 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `format` | str | `"int8"` / `"int4"` / `"fp8_e4m3"` / `"fp4_e2m1"` / `"nf4"` |
| `granularity` | str | `"per_tensor"` / `"per_channel"` / `"per_block"` |
| `axis` | int | per_channel / per_block 的 axis（默认 -1）|
| `block_size` | int | per_block 必填 |
| `transform` | str / None | `"hadamard"` 或 `None`（默认无 transform）|
| `weight_only` | bool | `True` 则只量化 weight，input / output 不量化（NF4 场景）|

## 其他参数

```python
run_format_study(
    ...,
    build_conv_model=my_conv_fn,     # 额外跑 Part D Conv2d 验证（可选）
    output_dir="results/my_study",   # 指定输出目录
    skip_parts={"C": True, "D": True},  # 跳过不需要的 Part
)
```

## 输出

```
output_dir/
├── results.json       # 精度 + per-layer QSNR/MSE（可用于重绘）
├── figures/           # 11 张 PNG
└── tables/            # 6 张 CSV
```

从已有结果重绘（不重跑实验）：

```python
from src.pipeline.format_study import plot_from_results
plot_from_results("results/my_study/results.json")
```
