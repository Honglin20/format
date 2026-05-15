# 4bit 量化格式对比分析 — 设计文档

**日期**: 2026-05-14
**状态**: 设计中
**模型**: Shakespeare character-level MiniGPT (d=192, n_heads=3, n_layers=4)
**预训练权重**: `scripts/weights/shakespeare_gpt.pt`

## 目标

回答核心问题：**4bit 下数据格式应该怎么做？**

通过系统化对比分析 MXINT / MXFP / NF4 + granularity × sparse 交叉扫描，找出最优 4bit 量化策略。

## 四部分分析

### Part 1 — MXINT 精度对比 (W8A8 / W4A8 / W4A4)

MX 风格（per_block + block_size=32）整数格式在不同位宽组合下的精度对比。

| Config | Weight | Activation | Granularity |
|--------|--------|-----------|-------------|
| W8A8 | int8 | int8 | per_block(32) |
| W4A8 | int4 | int8 | per_block(32) |
| W4A4 | int4 | int4 | per_block(32) |

输出：Markdown 表格 `[Config, PPL, ΔPPL, QSNR_local, QSNR_accum, Model Size]`

### Part 2 — 精度下降根因分析 + 可视化

四阶段闭环（复用 ADR-010 框架）：

1. **Diagnose** — `result.diagnose` 逐层 per-role QSNR，定位 top-K 问题层
2. **Characterize** — `result.characterize` 6-规则分类器，区分 weight vs input 问题
3. **Visualize** — 8bit vs 4bit 分布叠加直方图、per-channel QSNR 柱状图、量化误差热图
4. **Attribute** — 归因表 `[Layer, Role, DegradationType, QSNR_8bit, QSNR_4bit, Δ]`

数据来源：Tiny Shakespeare calibration data（4 batch），observer 自动采集。

### Part 3 — MXFP / NF4 对比

相同 4bit 位宽下不同数据格式对比：

| Config | Weight | Activation |
|--------|--------|-----------|
| MXINT-4 | int4 | int4 |
| MXFP-4 | fp4_e2m1 | fp4_e2m1 |
| NF4-W | nf4 | - (weight_only) |
| NF4-WA | nf4 | nf4 |
| MXFP-8 | fp8_e4m3 | fp8_e4m3 (上界参考) |

分析点：FP4 非均匀分布适应性、NF4 对激活值的适用性、结合 Part 2 分布特征解释差异。

### Part 4 — Granularity × Sparse 交叉扫描

`3 granularity × 5 outlier_ratio = 15 configs`，全部 int4：

```
per_tensor  × {0, 0.01, 0.05, 0.1, 0.2}
per_channel × {0, 0.01, 0.05, 0.1, 0.2}
per_block   × {0, 0.01, 0.05, 0.1, 0.2}
```

输出：Pivot markdown 表格（行=granularity, 列=outlier_ratio, 值=PPL/QSNR），标注最佳 sparse 度。

### 综合结论

回答：最佳 4bit 格式、最优 granularity+sparse 组合、是否需要 mixed-precision、不同层类型的策略差异。

## 技术方案

- **API**: `Study` + `QuantConfig`（高层 API，不重复定义模型结构）
- **模型加载**: `MiniGPT` + `torch.load("scripts/weights/shakespeare_gpt.pt")`
- **校准数据**: Tiny Shakespeare train set，4 batch
- **评估指标**: Perplexity（validation set）
- **可视化**: `result.plot` 方法 + `src/viz/figures.py` standalone 函数
- **输出**: 单个 Python 脚本，运行后输出 markdown 表格 + 保存 PNG 图表

## 文件规划

- `scripts/4bit_format_analysis.py` — 主分析脚本（唯一新增文件）
