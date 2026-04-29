# Format Study 使用指南

`examples/experiment_format_study.py` 是一个系统化的量化格式精度研究实验，比较 8 种格式 × 3 种粒度 × 3 种 Transform 的组合，产出 6 张表格和 11 张图表。

## 快速开始

```bash
PYTHONPATH=. python examples/experiment_format_study.py
```

运行后在 `results/` 下生成带时间戳的输出目录，包含所有表格、图表和 JSON 结果。

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-o, --output-dir` | `results/<timestamp>/` | 输出目录 |
| `--seed` | `42` | 随机种子（可复现） |
| `--calib-samples` | `256` | 校准样本数 |
| `--eval-samples` | `512` | 评估样本数 |
| `--batch-size` | `16` | 校准和评估的 batch size |
| `--skip-part-a` | — | 跳过 Part A |
| `--skip-part-b` | — | 跳过 Part B |
| `--skip-part-c` | — | 跳过 Part C |
| `--skip-part-d` | — | 跳过 Part D（含 Conv2d） |
| `--skip-part-d-conv` | — | 只跳过 Part D 的 Conv2d 部分 |
| `--plot-from RESULTS_JSON` | — | 从已有结果 JSON 重新生成图表（跳过实验） |

示例：

```bash
# 自定义输出和种子
PYTHONPATH=. python examples/experiment_format_study.py -o results/my_study --seed 1234

# 只跑 4-bit 对比 (Part B)
PYTHONPATH=. python examples/experiment_format_study.py --skip-part-a --skip-part-c --skip-part-d

# 从已有结果重新出图
PYTHONPATH=. python examples/experiment_format_study.py --plot-from results/2026-04-29_120000/results.json -o results/regenerated
```

## 实验矩阵

实验分四个部分（Part A / B / C / D），每部分产出特定的表格和图表。

### Part A: 8-bit 格式对比

比较三种 8-bit 格式，全部使用 PoT（power-of-two）scaling：

| 格式 | 类型 | 粒度 | 说明 |
|---|---|---|---|
| MXINT-8 | 对称整数 | per_block(32) | block-wise 共享 exponent |
| MXFP-8 | 浮点 (e4m3) | per_block(32) | OCP FP8 block-wise |
| INT8-PC | 对称整数 | per_channel(axis=0) | 每列独立 scale |

**产出**：Table 1（QSNR/MSE 精度汇总）+ Figure 1-3（逐层 QSNR、MSE、直方图）

### Part B: 4-bit 格式对比

比较四种 4-bit 格式：

| 格式 | 类型 | 粒度 | 说明 |
|---|---|---|---|
| MXINT-4 | 对称整数 | per_block(32) | block-wise |
| MXFP-4 | 浮点 (e2m1) | per_block(32) | MX FP4 block-wise |
| INT4-PC | 对称整数 | per_channel(axis=0) | 每列独立 |
| NF4-PC | 查找表 | per_channel(axis=0) | QLoRA 非均匀 LUT |

**产出**：Table 2 + Figure 4-6（逐层 QSNR、MSE、NF4 独有 LUT 分布图）

### Part C: FP32 vs PoT Scaling

比较 INT8-PC 和 INT4-PC 在 FP32 scaling 和 PoT scaling 下的精度差异。

| Scaling 方式 | 说明 |
|---|---|
| FP32 | scale 保留为任意浮点值 |
| PoT | scale 投影到最近的 2 的幂次（bit-shift，硬件友好） |

**产出**：Table 3-4 + Figure 7-8

### Part D: Transform 研究

在 4-bit INT4-PC 和 NF4-PC 上评估 SmoothQuant 和 Hadamard 变换的效果：

| Transform | 说明 |
|---|---|
| None | 无变换（baseline） |
| SmoothQuant | (x/s) @ (W\*s)，平滑 activation outlier → weight |
| Hadamard | FWHT 正交旋转，分散 outlier 能量 |

额外验证 MLP 和 Conv2d 两种架构（Conv2d 验证 `channel_axis=1` 路径）。

**产出**：Table 5（MLP）+ Table 6（Conv2d）+ Figure 9-11（Transform 对比热力图、分布指纹、SmoothQuant scale 分布）

## 自定义模型

替换 `experiment_format_study.py` 顶部的四个函数即可在自己的模型上运行：

```python
def build_model() -> nn.Module:
    """返回 FP32 参考模型的新实例。"""
    return MyModel()

def make_calib_data(num_samples=256, batch_size=16) -> List[torch.Tensor]:
    """返回校准数据列表（每个元素一个 batch）。"""
    return [torch.randn(batch_size, input_dim) for _ in range(num_samples // batch_size)]

def make_eval_loader(num_samples=512, batch_size=16) -> DataLoader:
    """返回评估 DataLoader（yield (input, label)）。"""
    ...

def eval_fn(model, dataloader) -> Dict[str, float]:
    """自定义评估函数，返回 dict[str, float]。"""
    ...
```

## 输出文件

每次运行在 `<output-dir>/` 下生成：

```
<output-dir>/
├── results.json        # 完整结果（可用 --plot-from 重绘）
├── report.txt          # 文本报告
├── table_1_*.csv       # 6 张 CSV 表格
├── figure_1_*.png      # 11 张 PNG 图表
└── experiment.log      # 运行日志
```

## 验证脚本

`examples/test_format_study_verification.py` 提供独立的理论验证：

```bash
PYTHONPATH=. python -m pytest examples/test_format_study_verification.py -v
```

12 个测试分三层递进验证：
- **Layer 1** (001-006): Format.quantize() 逐元素 bit-exact 正确性
- **Layer 2** (007-009): Linear 算子 + SmoothQuant + Hadamard
- **Layer 3** (010-012): QuantSession calibrate → analyze → compare 全流程

每个测试对应 `docs/verification/NNN-name.md` 中的手工推导文档，遵循"先理论推导、后代码验证"方法论。所有测试使用固定微型张量（W: 3×2, x: 2×2），确保结果可手算复现。
