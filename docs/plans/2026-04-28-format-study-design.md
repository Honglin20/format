# Quantization Format Precision Study — 实验设计

**日期**: 2026-04-28
**分支**: feature/refactor-src
**目标**: 系统性对比 MXINT/MXFP/INT-per-channel/NF4-per-channel 在 8-bit 和 4-bit 下的推理精度与层级量化误差，并评估 SmoothQuant/Hadamard transform 在 4-bit 下的精度增益。

---

## 实验矩阵

### Part A: 8-bit 格式对比（3 配置）

| 配置名 | Format | Granularity | Transform |
|--------|--------|-------------|------------|
| MXINT-8 | `int8` | per_block(32) | Identity |
| MXFP-8 | `fp8_e4m3` | per_block(32) | Identity |
| INT8-PC | `int8` | per_channel | Identity |

### Part B: 4-bit 格式对比（4 配置）

| 配置名 | Format | Granularity | Transform |
|--------|--------|-------------|------------|
| MXINT-4 | `int4` | per_block(32) | Identity |
| MXFP-4 | `fp4_e2m1` | per_block(32) | Identity |
| INT4-PC | `int4` | per_channel | Identity |
| NF4-PC | `nf4` | per_channel | Identity |

### Part C: FP32 vs PoT scaling 对比（4 配置）

对 INT per-channel 分别在 8-bit 和 4-bit 下，对比浮点 scale 与 PoT scale 的精度差异。

| 配置名 | Format | Granularity | PreScale pot | Transform |
|--------|--------|-------------|-------------|------------|
| INT8-PC-FP32 | `int8` | per_channel | False | PreScale |
| INT8-PC-PoT | `int8` | per_channel | True | PreScale |
| INT4-PC-FP32 | `int4` | per_channel | False | PreScale |
| INT4-PC-PoT | `int4` | per_channel | True | PreScale |

> PreScale 的 scale 通过 LSQ（LayerwiseScaleOptimizer）优化得到。

### Part D: Transform 对 4-bit 精度的提升（16 配置）

对 4 种 4-bit 格式，各测试 4 种 transform 策略：

| Transform 策略 | 说明 |
|----------------|------|
| Identity (None) | baseline |
| SmoothQuant | alpha=0.5, from_calibration() |
| Hadamard | FWHT rotation along last dim |
| Per-layer optimal | 每层独立选 QSNR 最高的 transform（从上述 3 种中选） |

**Part D 共 4 格式 × 4 策略 = 16 个模型运行。**

Per-layer optimal 的实现方式：
1. 对每个格式分别跑 Identity / SmoothQuant / Hadamard 三个 session
2. 对每层对比三种 transform 下的 QSNR，选最高者
3. 构建 heterogeneous config（每层不同 transform），跑第四次
4. 报告该 heterogeneous 模型的 E2E 精度

---

## 实验控制变量

| 控制项 | 取值 | 说明 |
|--------|------|------|
| Calibration 策略 | MSE (ScaleStrategy.MSE) | 统一控制，排除 scale 计算方式的混杂 |
| Block size | 32（默认）；额外扫 64, 128 | MX 格式 block size 敏感性作为子分析 |
| 默认 scaling | PoT (pot=True) | Part A/B/D 统一使用 PoT scaling |
| 默认 transform | Identity | Part A/B 无 transform；Part C 为 PreScale |
| FP32 Baseline | 所有 Part 均包含 | 精度上界参照 |
| QAT | 不启用（PTQ only） | 所有实验为 inference-only |

---

## 数据采集

每个配置通过 `AnalysisContext` 挂载以下 Observer：

| Observer | 采集数据 |
|----------|---------|
| QSNRObserver | 每层每个量化点的 QSNR (dB) |
| MSEObserver | 每层每个量化点的 MSE |
| HistogramObserver | fp32/quant/error 三通道直方图 |
| DistributionObserver | 峰度/偏度/稀疏率/动态范围/离群比等 13 维特征 |

额外采集：
- E2E 推理精度（用户自定义指标）
- LayerSensitivity 排序
- ErrorByDistribution 相关性分析

---

## 输出产物

### 表格（6 张）

| 表 | 内容 |
|----|------|
| Table 1 | Part A: 8-bit 格式 × 推理精度 + 全层平均 QSNR/MSE |
| Table 2 | Part B: 4-bit 格式 × 推理精度 + 全层平均 QSNR/MSE |
| Table 3 | Part C: FP32 vs PoT scaling 精度差值表（8-bit + 4-bit 并排） |
| Table 4 | Part D: 格式 × Transform 矩阵（E2E 精度 + 平均 QSNR） |
| Table 5 | Part D: 每层最优 transform 选择分布（各 transform 被选中的层数/比例） |
| Table 6 | 层级敏感度 Top-10 排名 |

### 图片（11 张）

| 图 | 内容 | 类型 |
|----|------|------|
| Fig 1 | 8-bit per-layer QSNR 折线图（3 条线） | 折线图 |
| Fig 2 | 4-bit per-layer QSNR 折线图（4 条线） | 折线图 |
| Fig 3 | 8-bit per-layer MSE 箱线图（3 组） | 箱线图 |
| Fig 4 | 4-bit per-layer MSE 箱线图（4 组） | 箱线图 |
| Fig 5 | FP32 vs PoT ΔQSNR 柱状图（8bit + 4bit 并排） | 柱状图 |
| Fig 6 | 关键层 fp32/quant/error 三通道直方图（3-5 层） | 直方图叠加 |
| Fig 7 | Part D: 格式×Transform 热力图 | 热力图 |
| Fig 8 | Part D: Per-layer optimal transform 选择分布饼图 | 饼图 |
| Fig 9 | Part D: Transform ΔQSNR vs baseline 柱状图 | 柱状图 |
| Fig 10 | QSNR vs 分布特征散点图矩阵 | 散点图 |
| Fig 11 | Layer-type 分组 QSNR 对比 | 分组柱状图 |

---

## 实现方式

- 单一 Python 脚本 `examples/experiment_format_study.py` 驱动全部实验
- 使用 `QuantSession` 管理每个实验配置
- `AnalysisContext` + 4 Observer 采集层级数据
- `matplotlib` + `seaborn` 出图
- 模型和指标通过用户自定义函数注入（模型无关）

---

## 预期实验周期（估算）

| Part | 配置数 | 单次耗时估计 | 总耗时估计 |
|------|--------|------------|-----------|
| A (8-bit) | 3 | 取决于模型 | ~3× |
| B (4-bit) | 4 | 同上 | ~4× |
| C (PoT) | 4 | 含 LSQ 优化 | ~4× (+ LSQ) |
| D (Transform) | 16 | 含 transform 计算 | ~16× |
| 额外 sweep | block_size×2 | 6 | ~6× |
| **合计** | **~33 次模型运行** | | |
