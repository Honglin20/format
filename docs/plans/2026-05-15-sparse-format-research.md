# Sparse Format Research: 实验设计与分析框架

**日期**: 2026-05-15
**状态**: 设计中
**依赖**: ADR-012（BANK + Sparse mask + Static sparse + outlier_format，已实施）

## 1. 目标

系统性评估 sparse 量化（outlier-based, two-group）相比于 MXINT（int4 + PER_BLOCK）在 4-bit 精度下的表现。回答以下核心研究问题：

1. **QSNR 对比**: sparse 在不同 granularity mode 下对随机/真实张量的 QSNR 是否优于 MXINT？
2. **Ratio 特性**: sparse ratio 变化如何影响 QSNR、有效比特宽、以及两者之间的 trade-off？
3. **Bank 甜点**: BANK 粒度下，bank_size 的最优值如何随张量维度和分布变化？
4. **泛化能力 (L4)**: 静态 sparse mask 从校准集到测试集的泛化 gap 有多大？

## 2. 目录结构

```
research/sparse/
├── README.md                    # 实验说明 + 如何运行 + 结果索引
├── configs/
│   └── experiments.py           # 实验配置（声明式，所有实验参数在此定义）
├── experiments/
│   ├── l1_baseline.py           # L1: Sparse vs MXINT QSNR 对比
│   ├── l2_ratio_sweep.py        # L2: Sparse ratio → QSNR + 有效比特宽
│   ├── l3_bank_sweetspot.py     # L3: Bank 粒度甜点分析
│   └── l4_real_model.py         # L4: 真实模型泛化（接口预留，仅骨架）
├── viz/
│   ├── l1_viz.py                # L1 可视化（standalone，读 JSON 结果）
│   ├── l2_viz.py                # L2 可视化
│   ├── l3_viz.py                # L3 可视化
│   └── common.py                # 共享绘图工具（风格、色板、字体）
├── results/                     # 实验结果（JSON，版本化）
│   └── .gitkeep
└── figures/                     # 输出图表（PNG/SVG）
    └── .gitkeep
```

**原则**：
- 实验脚本调用 `quantize()` + 自行计算 QSNR/MSE → 输出 JSON 到 `results/`
- 可视化脚本只读 JSON → 输出 PNG/SVG 到 `figures/`
- 实验和可视化解耦：改颜色/布局不需要重跑实验
- L4 只写接口骨架 + docstring，不实现具体逻辑

## 3. 四层实验设计

### 3.1 L1: Sparse vs MXINT QSNR 基础对比

**目标**: 回答 "sparse 在什么 granularity 下 QSNR 优于 MXINT？"

**自变量**:
- `format`: int4（固定）
- `granularity`: PER_TENSOR, PER_CHANNEL, PER_BLOCK, BANK（4 种）× {dense, sparse}（2 种）
  - dense: `outlier_ratio=0.0`
  - sparse: `outlier_ratio=0.1`
- `distribution`: normal, lognormal(σ=1), powerlaw(α=2.5), real_weight, real_activation
- `tensor_shape`: [(256,256), (512,128), (64,1024)] 覆盖不同的宽高比
- `seed`: 5 个（报告 mean ± std）

**因变量**: QSNR (dB), MSE

**特殊处理**:
- PER_BLOCK 静态 sparse 当前 raise NotImplementedError → 跳过 dense/sparse 对比，只标注
- real_weight/activation 从 `scripts/mnist_hadamard_study.py` 的 MLP 模型提取

**输出**:
- `results/l1_baseline.json`
- `figures/l1_qsnr_comparison.png` — grouped bar chart, x=granularity, hue=dense/sparse, facet=distribution
- `figures/l1_qsnr_heatmap.png` — heatmap, x=granularity, y=tensor_shape, color=ΔQSNR(sparse-dense)

### 3.2 L2: Sparse Ratio 特性曲线

**目标**: 回答 "outlier_ratio 如何影响 QSNR 和有效比特宽？"

**自变量**:
- `format`: int4
- `granularity`: PER_TENSOR, PER_CHANNEL, BANK（PER_BLOCK 标注为不可用）
- `outlier_ratio`: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]（8 个点）
- `distribution`: normal, lognormal, powerlaw
- `tensor_shape`: (256, 256)
- `seed`: 5 个

**因变量**:
- QSNR (dB)
- 有效比特宽 `b_eff`（按以下公式计算）

**有效比特宽模型**:
```
b_eff = b×(1-r) + b_o×r + 1 + 2×s/group_size

其中:
  b        = 主格式每元素比特（int4 = 4）
  b_o      = outlier 格式每元素比特（默认 = b）
  r        = outlier_ratio
  1        = mask 开销（1 bit/element）
  s        = scale 比特宽度（pot = 8 bits, fp32 = 32 bits）
  group_size = granularity 组大小:
    PER_TENSOR: N_total
    PER_CHANNEL: N_per_channel
    BANK: bank_size
```

**输出**:
- `results/l2_ratio_sweep.json`
- `figures/l2_qsnr_vs_ratio.png` — line plot, x=outlier_ratio, y=QSNR, hue=granularity
- `figures/l2_bitwidth_vs_ratio.png` — line plot, x=outlier_ratio, y=b_eff, hue=granularity
- `figures/l2_qsnr_vs_bitwidth.png` — scatter/line, x=b_eff, y=QSNR, hue=granularity（Pareto 前沿）
- `figures/l2_per_distribution.png` — facet by distribution, 3 subplots

### 3.3 L3: Bank 粒度甜点分析

**目标**: 回答 "给定张量形状，bank_size 取多少最优？"

**自变量**:
- `format`: int4
- `granularity_mode`: BANK + sparse (outlier_ratio=0.1)
- `bank_size`: [8, 16, 32, 64, 128, 256]（6 个点）
- `tensor_dim`: 张量沿 bank_axis 的维度大小 [64, 128, 256, 512, 1024, 2048]（6 个点）
- `distribution`: normal, lognormal
- `seed`: 5 个

**固定**: tensor 另一维度 = 256（形状如 (256, D_bank_axis)）

**因变量**:
- QSNR (dB)
- 有效比特宽
- QSNR / b_eff（单位比特的效率）

**输出**:
- `results/l3_bank_sweetspot.json`
- `figures/l3_bank_heatmap.png` — 2D heatmap, x=bank_size, y=tensor_dim, color=QSNR
- `figures/l3_bank_efficiency.png` — 同 heatmap 但 color=QSNR/b_eff
- `figures/l3_bank_lineplot.png` — line plot, x=bank_size, y=QSNR, hue=tensor_dim (facet by distribution)

### 3.4 L4: 真实模型泛化（接口预留）

**目标**: 测量静态 sparse mask 从校准集到测试集的 QSNR gap。

**设计思路**（仅接口，不实现）:
1. 从真实模型提取某一层的 weight / activation
2. 用 calibration samples (S=1,2,4,8,16,32) 生成 mask
3. 用 hold-out test samples 测量 QSNR
4. 报告 `QSNR_gap = QSNR_calib - QSNR_test`

**预留接口**:
```python
# experiments/l4_real_model.py (skeleton only)
def run_l4_real_model(
    model_path: str,
    layer_name: str,
    calib_samples: List[int] = [1, 2, 4, 8, 16, 32],
    test_samples: int = 100,
    outlier_ratio: float = 0.1,
    granularity: str = "bank",
    bank_size: int = 16,
    output_dir: str = "research/sparse/results",
) -> dict:
    """Measure mask generalization gap on real model tensors.

    This is a skeleton — implement when you have a target model
    and calibration pipeline ready.

    Args:
        model_path: Path to the saved model or model loading function.
        layer_name: Name of the target layer to extract tensors from.
        calib_samples: List of calibration sample counts to sweep.
        test_samples: Number of hold-out test samples.
        outlier_ratio: Fraction of elements marked as outliers.
        granularity: Granularity mode string.
        bank_size: Bank size (only used for BANK granularity).
        output_dir: Directory to write results JSON.

    Returns:
        Dict with keys: calib_qsnr, test_qsnr, gap, mask_stability.
    """
    raise NotImplementedError(
        "L4 real model generalization is not yet implemented. "
        "Implement when you have a target model and calibration data ready."
    )
```

## 4. Session API 使用规范

所有实验必须通过以下 Session 层 API 完成：

### 4.1 核心 API

```python
# 张量级量化（单次调用）
from src.quantize.elemwise import quantize
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.formats.base import FormatBase

# 构造 scheme
fmt = FormatBase.from_str("int4")
g = GranularitySpec(
    mode=GranularityMode.BANK,
    bank_size=16,
    bank_axis=-1,
    outlier_ratio=0.1,
)
scheme = QuantScheme(format=fmt, granularity=g, scale_storage="pot")

# 量化
x_q = quantize(x, scheme)

# 静态 sparse 路径（pre-computed mask + scales）
from src.quantize._sparse_mask import compute_sparse_mask
mask = compute_sparse_mask(x_calib, fmt, g, outlier_ratio=0.1)
# amax_n, amax_o 从校准数据预计算
x_q = quantize(x, scheme, mask=mask, scale=amax_n, scale_o=amax_o)

# QSNR 计算（直接计算，不依赖 observer pipeline）
def compute_qsnr(x_fp32: torch.Tensor, x_quant: torch.Tensor) -> float:
    num = x_fp32.pow(2).mean()
    den = (x_fp32 - x_quant).pow(2).mean().clamp_min(1e-30)
    return (10 * torch.log10(num / den)).item()
```

### 4.2 禁止使用的 API

- 不使用 `run_quantization()` / `Session`（这些是模型级 pipeline，实验是张量级）
- 不使用 `QSNRObserver` / `MSEObserver`（设计用于模型内 hook，张量实验直接计算更简单）
- 不 import `mx` 包

### 4.3 实验脚本模板

```python
"""L1: Sparse vs MXINT QSNR baseline comparison."""
import json
import torch
from pathlib import Path
from itertools import product
from src.quantize.elemwise import quantize
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec, GranularityMode
from src.formats.base import FormatBase

def run_l1(config: dict, output_dir: str) -> dict:
    results = []
    for params in _generate_sweep(config):
        qsnr_vals = []
        for seed in range(config["n_seeds"]):
            torch.manual_seed(seed)
            x = _generate_tensor(params)
            x_q = quantize(x, params["scheme"])
            qsnr_vals.append(compute_qsnr(x, x_q))
        results.append({**params, "qsnr_mean": ..., "qsnr_std": ...})
    # Write JSON
    out = Path(output_dir) / "l1_baseline.json"
    out.write_text(json.dumps(results, indent=2))
    return results
```

## 5. 有效比特宽模型

### 5.1 通用公式

```
b_eff = (1-r)*b + r*b_o + b_mask + b_scale

其中:
  b_mask = 1 bit/element  （固定开销）
  b_scale = s / group_size × 2  （normal + outlier 各一组 scale）

  s = 8 (pot) or 32 (fp32)
  group_size 取决于 granularity:
    PER_TENSOR:  N = prod(shape)
    PER_CHANNEL: N_per_channel = N / C
    BANK:        bank_size
    PER_BLOCK:   block_size  （但 PER_BLOCK 静态 sparse 暂不可用）
```

### 5.2 示例计算

对于 (256, 256) tensor, int4, r=0.1, pot scale:

| Granularity | group_size | b_scale | b_eff |
|-------------|-----------|---------|-------|
| PER_TENSOR  | 65536 | 2×8/65536 ≈ 0.0002 | 4 + 0 + 1 + 0.0002 = **5.00** |
| PER_CHANNEL | 256 | 2×8/256 = 0.0625 | 4 + 0 + 1 + 0.0625 = **5.06** |
| BANK(16)    | 16 | 2×8/16 = 1.0 | 4 + 0 + 1 + 1.0 = **6.00** |
| MXINT baseline | 32 | 8/32 = 0.25 | 4 + 0.25 = **4.25** |

> 关键洞察: BANK 粒度 sparse 的 scale 开销最显著（每个 bank 只有 16 个元素但需要 2 组 8-bit scale），有效比特宽达到 6.00 bpw——比 MXINT 的 4.25 bpw 高出 41%。公平对比需要在相同 b_eff 下进行。

### 5.3 公平对比策略

对于 L1/L2，除了原始对比（各自默认参数），还需做 **b_eff 对齐对比**：
- 计算 sparse 方案的 b_eff
- 找到 b_eff 最接近的 MXINT block_size 变体（如 block_size=16 → 4.5 bpw, block_size=8 → 5.0 bpw）
- 在对齐的 b_eff 下比较 QSNR

## 6. 可视化规范

### 6.1 共享风格 (viz/common.py)

```python
# 色板
COLORS = {
    "mxint": "#2196F3",        # 蓝色 — baseline
    "sparse_per_tensor": "#FF9800",
    "sparse_per_channel": "#4CAF50",
    "sparse_bank": "#E91E63",
    "sparse_per_block": "#9C27B0",
}

# 输出尺寸
FIG_SINGLE = (8, 5)
FIG_DOUBLE = (12, 5)
FIG_SQUARE = (8, 8)

# 字体
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
```

### 6.2 各层图表清单

| 层 | 图表 | 类型 | 文件名 |
|----|------|------|--------|
| L1 | QSNR grouped bar | grouped bar + error bar | `l1_qsnr_comparison.png` |
| L1 | ΔQSNR heatmap | 2D heatmap | `l1_qsnr_heatmap.png` |
| L2 | QSNR vs ratio | line + error band | `l2_qsnr_vs_ratio.png` |
| L2 | b_eff vs ratio | line | `l2_bitwidth_vs_ratio.png` |
| L2 | QSNR vs b_eff | Pareto scatter | `l2_qsnr_vs_bitwidth.png` |
| L2 | Per-distribution facet | faceted lines | `l2_per_distribution.png` |
| L3 | Bank sweet spot | 2D heatmap | `l3_bank_heatmap.png` |
| L3 | Bank efficiency | 2D heatmap | `l3_bank_efficiency.png` |
| L3 | Bank line plot | faceted lines | `l3_bank_lineplot.png` |

### 6.3 输出表格

每个实验脚本同时输出 markdown 表格（打印到 stdout + 写入结果目录），例如 L1:

```
| Granularity     | Mode   | QSNR (dB)        | MSE              |
|-----------------|--------|------------------|------------------|
| PER_TENSOR      | dense  | 12.34 ± 0.56    | 0.058 ± 0.003   |
| PER_TENSOR      | sparse | 14.21 ± 0.42    | 0.038 ± 0.002   |
| PER_CHANNEL     | dense  | 18.90 ± 0.31    | 0.013 ± 0.001   |
| PER_CHANNEL     | sparse | 19.45 ± 0.28    | 0.011 ± 0.001   |
| BANK(16)        | dense  | 16.78 ± 0.44    | 0.021 ± 0.002   |
| BANK(16)        | sparse | 18.12 ± 0.37    | 0.015 ± 0.001   |
| PER_BLOCK(32)   | MXINT  | 15.50 ± 0.50    | 0.028 ± 0.002   |
```

## 7. 已知限制与假设

| 限制 | 影响 | 缓解 |
|------|------|------|
| PER_BLOCK 静态 sparse 不可用 | L1/L2 缺少 PER_BLOCK sparse 数据点 | 标注不可用，用 PER_BLOCK dense 作为 MXINT baseline |
| 随机张量 ≠ 真实分布 | synthetic 实验结论可能不适用于真实模型 | L4 接口预留给真实模型验证 |
| 单张量实验无跨层交互 | 不考虑层间误差累积 | 这是刻意设计——先解耦分析，后续再做端到端 |
| int4 仅一种格式 | 不覆盖 fp4/nf4 | 按用户决策聚焦 int4；格式扩展通过 config 参数支持 |
| pot scale 默认 | fp32 scale 对 BANK 影响更大（b_eff 公式变化） | 设为可配置参数，不硬编码 |

## 8. 后续扩展方向（不纳入当前范围）

- **outlier_format 消融**: int4+int8 / int4+fp8 混合精度
- **Transform 交互**: Hadamard × Sparse 2×2 矩阵
- **Error decomposition**: normal 组 vs outlier 组误差分解
- **Oracle mask**: 后验最优 mask 作为理论上界
- **fp4_e2m1 / nf4**: 扩展格式对比
- **端到端模型评估**: 用 Session API 跑完整模型 quantize + eval
