# Format Study 使用指南

系统化量化格式精度研究：8-bit 对比、4-bit 对比、FP32 vs PoT scaling、Transform 效果，产出多张表格和图表。

## 程序化调用

```python
from src.session import Study, QuantConfig

configs = [
    QuantConfig(name="int8", w_format="int8", w_granularity="per_block", w_block_size=32),
    QuantConfig(name="int4", w_format="int4", w_granularity="per_channel"),
]

study = Study(configs, model=model)
report = study.run(calib_data, eval_fn=eval_fn, eval_data=eval_loader)
report.print_summary()
report.save("results/my_study/")
```

## 需要提供的函数

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

参见 `examples/format_study_random.py` 查看完整的参考实现。

## 配置搜索空间

通过 `QuantConfig` 字段定义每个配置：

```python
from src.session import QuantConfig

configs = [
    # 基本格式对比
    QuantConfig(name="MXINT-8", w_format="int8", w_granularity="per_block", w_block_size=32),
    QuantConfig(name="INT8-PC",  w_format="int8", w_granularity="per_channel"),
    QuantConfig(name="NF4-PC",   w_format="nf4",  w_granularity="per_channel", weight_only=True),
    # 加 Hadamard transform
    QuantConfig(name="INT8-Had", w_format="int8", w_granularity="per_channel", transform="hadamard"),
    # 4-bit
    QuantConfig(name="fp4-blk16", w_format="fp4_e2m1", w_granularity="per_block", w_block_size=16),
    # SmoothQuant
    QuantConfig(name="INT8-SQ", w_format="int8", transform="smoothquant", sq_alpha=0.5),
    # LSQ (prescale)
    QuantConfig(name="INT8-LSQ", w_format="int8", transform="prescale", lsq_steps=100),
]
```

主要字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `w_format` | str | `"int8"` / `"int4"` / `"fp8_e4m3"` / `"fp4_e2m1"` / `"nf4"` / ... |
| `a_format` | str / None | 独立 activation 格式（None = 同 weight），wXaY mixed-precision |
| `w_granularity` | str | `"per_tensor"` / `"per_channel"` / `"per_block"` |
| `w_block_size` | int / None | per_block 必填 |
| `transform` | str | `"none"` / `"hadamard"` / `"smoothquant"` / `"prescale"` |
| `weight_only` | bool | `True` 则只量化 weight（NF4 场景）|
| `calibrator` | str | `"mse"` / `"max"` / `"percentile"` / `"kl"` |
| `scale_storage` | str | `"fp32"` / `"pot"` |

预设配置在 `src/session/study_config.py` 的 `STUDY_CONFIG` 中：

```python
from src.session.study_config import STUDY_CONFIG
# STUDY_CONFIG["part_a"], STUDY_CONFIG["part_b"], ...
```

## 输出

```
output_dir/
├── results.json       # 精度 + per-layer QSNR/MSE（可用于重绘）
├── figures/           # PNG 图表
└── tables/            # CSV 表格
```

从已有结果重绘（不重跑实验）：

```python
from src.report import StudyReport
StudyReport.from_file("results/my_study/results.json").save("results/regen/")
```

## PerLayerOpt

按层选最优 transform 的后处理：

```python
from src.session import per_layer_optimal

results = study.run(calib_data, eval_fn=eval_fn)
opt_result = per_layer_optimal(results, calib_data, fp32_model, eval_fn=eval_fn)
```
