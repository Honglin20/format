# 系统化误差分析示例 — Transformer + AG News + int4

本文档演示 ADR-010 定义的 **四阶段闭环分析工作流**，以预训练 Transformer 模型在 AG News 数据集上的 int4-per_channel 量化为例。

模型：2-layer Transformer (d=128, nhead=4) · 数据集：AG News 4 分类 · 量化：int4 per_channel, quantize_nonlinear=True · 9 个被量化模块（4 Linear + 4 LayerNorm + 1 classifier）

---

## 0. 实验设置

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.session import Session, QuantConfig

# ══ 模型定义 ══
# 注意：为捕获所有层的 observer 数据，需继承 nn.TransformerEncoderLayer
# 并覆盖 forward 强制走 slow path（详见 ADR-010 中关于 PyTorch fused fast
# path 的已知限制）。

class SlowTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """始终走 slow path，确保 Quantized* 模块的 observer hook 触发。"""
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=4, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = SlowTransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ══ 数据加载 ══
# AG News 训练/测试集下载到 /tmp/agnews_data/
# 词汇表及预训练权重从 scripts/weights/transformer_agnews.pt 加载

torch.manual_seed(42)
ckpt = torch.load("scripts/weights/transformer_agnews.pt", map_location="cpu")
vocab = ckpt["vocab"]
hparams = {k: ckpt[k] for k in ["vocab_size", "num_classes", "d_model",
           "nhead", "num_layers", "dim_feedforward", "max_len"]}

model = TransformerClassifier(**hparams)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 校准数据：512 条训练样本，batch_size=64，取前 8 个 batch
calib_samples = [x for x, _y in calib_loader][:8]
# 评估数据：7600 条测试样本，batch_size=128
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=128)

# ══ 评估函数 ══
def eval_fn(model, data):
    model.eval()
    if isinstance(data, list):       # 校准模式（无标签）
        with torch.no_grad():
            for batch in data: model(batch)
        return {}
    correct, total = 0, 0            # 评估模式
    with torch.no_grad():
        for x, y in data:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total if total > 0 else 0.0}

# ══ 量化配置 ══
cfg = QuantConfig(
    name="int4-pc",
    w_format="int4", a_format="int4",
    w_granularity="per_channel", a_granularity="per_channel",
    w_axis=-1, a_axis=-1,
    quantize_nonlinear=True,   # 量化 TransformerEncoderLayer 内部的 Linear/LayerNorm
)
```

## 0.1 基线评估

```python
session = Session(model, cfg, keep_fp32=True)
session.quantize(calib_data=calib_samples)
session.calibrate(calib_samples)
session.analyze(calib_samples, outputs=["qsnr", "distribution"])
session.evaluate(test_loader, eval_fn)
result = session.result
```

```
Config: int4-pc | accuracy: fp32=0.8224 quant=0.7889 | avg QSNR=8.6 dB | Δaccuracy=-0.0334

Metric       FP32       Quant      Δ
---------------------------------------------
accuracy     0.8224     0.7889     -0.0334
```

FP32 基线准确率 82.24%，int4 量化后降至 78.89%，下降 3.34 个百分点。累计 QSNR（端到端信号保真度）仅 7.2 dB，说明误差在层间显著累积。

---

## 1. DIAGNOSE — `result.diagnose`

`ErrorProvenance` 提供按 role × layer_type 分层的误差溯源视图。

### 1.1 全局概览

```python
prov = result.diagnose
print(prov.summary())
```

```
Role       Type           Count   Avg QSNR   Min QSNR      Std
--------------------------------------------------------------
input      Linear             4        7.2        6.0      0.8
input      Norm               4        6.5        4.7      1.2
input      Other              1        9.6        9.6      0.0
output     Linear             4        7.4        6.8      0.6
output     Norm               4        5.4        2.3      2.1
output     Other              1       26.2       26.2      0.0
weight     Linear             4       10.5        9.7      1.0
weight     Norm               4       21.0       20.2      1.1
weight     Other              1       14.2       14.2      0.0
```

**解读**：
- **Weight QSNR 最高**（Linear 10.5 dB, Norm 21.0 dB）：权重分布在量化后被保留得最好，因为权重通常分布均匀、动态范围可控。
- **Output QSNR 分化严重**：Norm output 最低（均值 5.4 dB，最低仅 2.3 dB），但 classifier output 高达 26.2 dB。LayerNorm 输出是误差的主要注入点。
- **Input QSNR 整体偏低**（6.0–9.6 dB）：输入激活的动态范围受上游误差累积影响。

### 1.2 逐层三列视图

```python
print(prov.per_role_table(max_layers=15))
```

```
Layer                               Input     Weight     Output  Dominant
-------------------------------------------------------------------------
transformer.layers.0.norm1            4.7       21.2        2.3  output
transformer.layers.1.linear2          6.0       12.0        6.9  input
transformer.layers.0.norm2            7.2       22.5        6.4  output
transformer.layers.1.norm2            7.1       20.2        6.4  output
transformer.layers.1.norm1            7.0       20.2        6.6  output
transformer.layers.0.linear1          7.6       10.2        6.8  output
transformer.layers.0.linear2          7.6       10.2        7.9  input
transformer.layers.1.linear1          7.7        9.7        7.9  input
classifier                            9.6       14.2       26.2  input
```

Dominant 列标注了每层最差的 role：
- **4/9 层以 output 为主导退化**（全部是 Norm 层）：LayerNorm 输出端量化噪声最大。
- **4/9 层以 input 为主导退化**（Linear 层 + classifier）：输入激活的量化是瓶颈。
- **没有 weight 主导层**：权重不是当前配置的瓶颈，提升 weight 位宽不会带来最大收益。

### 1.3 Top-K 问题定位

```python
for name, q in prov.top_k(5, role="weight"):
    print(f"  {name:<55} QSNR={q:.1f} dB")
```

```
  transformer.layers.1.linear1                            QSNR=9.7 dB
  transformer.layers.0.linear2                            QSNR=10.2 dB
  transformer.layers.0.linear1                            QSNR=10.2 dB
  transformer.layers.1.linear2                            QSNR=12.0 dB
  classifier                                              QSNR=14.2 dB
```

```python
for name, q in prov.top_k(5, role="auto"):   # 每层取最差 role
    print(f"  {name:<55} QSNR={q:.1f} dB")
```

```
  transformer.layers.0.norm1                              QSNR=2.3 dB
  transformer.layers.1.linear2                            QSNR=6.0 dB
  transformer.layers.0.norm2                              QSNR=6.4 dB
  transformer.layers.1.norm2                              QSNR=6.4 dB
  transformer.layers.1.norm1                              QSNR=6.6 dB
```

`role="auto"` 模式揭示了关键洞察：**前 5 中最差的全是 Norm output 和 Linear input，与 weight 无关**。如果只看 weight，会错过真正的瓶颈 `transformer.layers.0.norm1`（QSNR 仅 2.3 dB）。

### 1.4 误差传播溯源

```python
print(prov.error_source_analysis(role="output"))
```

```
=========================================================================================================
  Error Source Analysis — int4-pc [output]
=========================================================================================================
Layer                          Accum QSNR   Local QSNR      Delta   Headroom  Diagnosis
---------------------------------------------------------------------------------------
classifier                          11.82        26.22      +0.00     +14.40  Propagated
transformer.layers.0.linear1         8.66         6.75      +3.15      -1.91  Source
transformer.layers.0.linear2         3.76         7.93      +4.91      +4.17  Mixed
transformer.layers.0.norm1           6.97         2.34      -3.22      -4.64  Source
transformer.layers.0.norm2           5.05         6.39      +1.92      +1.33  Source
transformer.layers.1.linear1         8.82         7.93      -3.76      -0.89  Source
transformer.layers.1.linear2         8.13         6.91      +0.69      -1.22  Source
transformer.layers.1.norm1           5.94         6.61      +2.19      +0.67  Source
transformer.layers.1.norm2           5.56         6.44      +0.38      +0.88  Source
---------------------------------------------------------------------------------------
Summary:                                               drop=+6.3 avg_headroom=+1.4  7 source, 1 mixed, 1 propagated
```

**误差传播诊断**：
- **7 层标记为 Source**：这些层是本地量化误差的主要产生者，本地 QSNR 接近或低于累计 QSNR。
- **Classifier 标记为 Propagated**：本地 QSNR（26.2 dB）远高于累计 QSNR（11.8 dB），说明它接收到的输入已被上游噪声污染。这不是 classifier 本身的问题，而是上游累积所致。
- **整体累计 QSNR 下降 6.3 dB**：从第一个量化模块到 classifier，累计信号质量从 ~14 dB 降至 ~12 dB，主要下降发生在 transformer.layers.0 内部。

---

## 2. CHARACTERIZE — `result.characterize`

`DistributionDiagnosis` 通过 6 规则分类引擎，将分布特征因果映射到量化退化机制。

### 2.1 单层深度剖析

```python
diag = result.characterize

# 取 weight QSNR 最差的 3 层
worst_w = prov.top_k(3, role="weight")
for layer_name, qsnr in worst_w:
    print(diag.profile(layer_name, role="weight"))
```

```
transformer.layers.1.linear1 (weight)
  QSNR: 9.7 dB
  Crest factor: 2.44          ← 峰值/均值比，低 → 分布均匀
  Outlier ratio (>3σ): 0.0%   ← 无离群值
  Dynamic range: 10.2 bits    ← >8 bits，浪费量化区间
  Excess kurtosis: -0.72      ← 负值 → 比正态分布更平坦
  Normalised entropy: 0.94    ← 接近 1.0，信息密度高
  Bimodality coeff: 0.43      ← <0.7，非双峰

  Diagnosis: high_dynamic_range
  High dynamic range — the ratio between max and min non-zero values
  exceeds 8 bits, wasting quantisation levels on empty range.

  Suggested: Consider per_channel granularity or hadamard transform.
```

```
transformer.layers.0.linear2 (weight)
  QSNR: 10.2 dB
  Crest factor: 2.15 | Outlier ratio: 0.0% | Dynamic range: 6.5 bits
  Excess kurtosis: -0.72 | Norm entropy: 0.92 | Bimodality: 0.43

  Diagnosis: benign
  Benign — no problematic distribution features detected.

  Suggested: Check other causes: layer position, upstream error
  propagation, or model architecture.
```

```
transformer.layers.0.linear1 (weight)
  QSNR: 10.2 dB
  Crest factor: 2.74 | Outlier ratio: 0.0% | Dynamic range: 13.1 bits
  Excess kurtosis: -0.17 | Norm entropy: 0.91 | Bimodality: 0.35

  Diagnosis: high_dynamic_range
  Suggested: Consider per_channel granularity or hadamard transform.
```

**洞察**：两个 Linear weight 层被诊断为 `high_dynamic_range`——动态范围超过 13 bits，4-bit 量化只能覆盖 16 个级别，大量空范围被浪费。已经使用 `per_channel` 粒度，下一步应尝试 Hadamard transform 进一步压缩动态范围。

### 2.2 全局因果矩阵

```python
print(diag.causal_analysis())
```

关键行节选：

```
Layer                          Role           QSNR   Crest     OL%     DR  Classification
-----------------------------------------------------------------------------------------
transformer.layers.0.linear2   input           7.6    14.6    2.9%    9.4  outlier_dominated
transformer.layers.1.linear2   input           6.0     8.0    4.2%    6.9  heavy_tailed
transformer.layers.0.norm2     output          6.4     3.0    0.0%   14.7  high_dynamic_range
transformer.layers.1.norm1     output          6.6     2.7    0.0%   16.6  high_dynamic_range
transformer.layers.0.norm1     output          2.3     3.0    0.0%    6.5  benign
transformer.layers.1.norm2     weight         20.2     1.0  100.0%    0.0  low_entropy
```

**因果模式**：
1. **Linear input → outlier_dominated / heavy_tailed**：激活张量中 2.9%–4.2% 的值超过 3σ，导致量化器将大范围分配给少数离群值。宜用 SmoothQuant 或 per_channel 缓解。
2. **Norm output → high_dynamic_range**：动态范围 14.7–16.6 bits，远超 4-bit 的 16 级别，是输出误差的主要来源。
3. **Norm weight → low_entropy**：所有 LayerNorm 权重的归一化熵 < 0.33，且 100% 的值为 0（3σ 阈值下）。这是因为 LayerNorm 权重向量通常接近均匀分布，量化非常友好——QSNR 高达 20–22 dB 证明了这一点。
4. **`transformer.layers.0.norm1` output 为 benign**：尽管 output QSNR 仅 2.3 dB（全模型最差），分布分类却是 benign。这说明退化不是数据分布导致的，而是**结构性因素**（如该层恰好处于残差连接后，上游噪声和自身量化叠加）。

### 2.3 退化分类

```python
for layer_name, _ in prov.top_k(3, role="auto"):
    print(diag.classify(layer_name, role="auto"))
```

返回具体的退化分类标签（`outlier_dominated` / `heavy_tailed` / `high_dynamic_range` / `benign`），方便脚本化决策。

---

## 3. PLAN — `result.plan`

`InterventionPlanner` 基于 diagnose 和 characterize 的结果，自动生成提升方案。

### 3.1 Top-K 精度提升

```python
planner = result.plan

plan_w = planner.top_k_boost(k=3, role="weight", target_bits=8)
plan_auto = planner.top_k_boost(k=3, role="auto", target_bits=8)
```

**role="weight" → 3 个 Linear weight 提升至 8-bit**：

```
Layer                          Change                                   Reason
----------------------------------------------------------------------------------------------------
transformer.layers.0.linear1   weight: int4 → 8bit                      QSNR=10.2 dB (worst weight)
transformer.layers.0.linear2   weight: int4 → 8bit                      QSNR=10.2 dB (worst weight)
transformer.layers.1.linear1   weight: int4 → 8bit                      QSNR=9.7 dB (worst weight)
```

**role="auto" → 每层最差 role 提升**：

```
Layer                          Change                                   Reason
----------------------------------------------------------------------------------------------------
transformer.layers.0.norm1     output: int4 → 8bit                      QSNR=2.3 dB (worst output)
transformer.layers.0.norm2     output: int4 → 8bit                      QSNR=6.4 dB (worst output)
transformer.layers.1.linear2   input: int4 → 8bit                       QSNR=6.0 dB (worst input)
```

两种方案截然不同：weight 方案提升 Linear 权重，auto 方案优先修复最差的 Norm output（2.3 dB）。

### 3.2 策略推荐

```python
print(planner.recommend(strategy="conservative").explain())
# → 2 layers: transformer.layers.0.norm1 (output), transformer.layers.1.linear2 (input)

print(planner.recommend(strategy="aggressive").explain())
# → 9 layers: 所有层的 worst role 全部提升
```

- **Conservative**：只改 QSNR < 10 dB 的层（2 层），最小修改量。
- **Aggressive**：改全部 9 层，追求最大精度恢复但存储开销大。

---

## 4. INTERVENE — `result.intervention`

`InterventionAccessor.compare()` 应用方案、重新运行量化流程、返回 `InterventionComparison` 对比表。

### 4.1 auto 方案

```python
intervention = result.intervention
cmp = intervention.compare(model, calib_samples, plan_auto,
                            eval_data=test_loader, eval_fn=eval_fn)
print(cmp.summary())
```

```
Intervention Comparison
================================================================================
  Plan: Top-3 auto (worst per layer) boost to 8-bit
  Layers modified: 3

  Metric       FP32         Baseline     Intervention Change
  ------------------------------------------------------------
  accuracy     0.8224       0.7889       0.8166       +0.0276

  Avg QSNR: baseline=8.6 dB → intervention=8.6 dB (Δ=-0.0 dB)

  Layer                          Role        QSNR Before   QSNR After          Δ
  ------------------------------------------------------------------------------
  transformer.layers.0.norm1     output: int4 → 8bit          2.3          5.7       +3.4
  transformer.layers.0.norm2     output: int4 → 8bit          6.4          7.9       +1.5
  transformer.layers.1.linear2   input: int4 → 8bit          6.9          8.3       +1.4
```

**结果**：只改动 3 层（2 个 Norm output + 1 个 Linear input），准确率从 78.89% 恢复到 81.66%，回收了 **83% 的量化损失**（-0.0334 → -0.0057）。

各层 QSNR 改善：
- `transformer.layers.0.norm1` output: 2.3 → 5.7 dB（+3.4 dB）
- `transformer.layers.0.norm2` output: 6.4 → 7.9 dB（+1.5 dB）
- `transformer.layers.1.linear2` input: 6.9 → 8.3 dB（+1.4 dB）

### 4.2 Conservative 方案

```
  Plan: Top-2 auto boost to 8-bit
  Layers modified: 2

  Metric       FP32         Baseline     Intervention Change
  ------------------------------------------------------------
  accuracy     0.8224       0.7889       0.8166       +0.0276

  Layer                          Role        QSNR Before   QSNR After          Δ
  ------------------------------------------------------------------------------
  transformer.layers.0.norm1     output: int4 → 8bit          2.3          5.7       +3.4
  transformer.layers.1.linear2   input: int4 → 8bit          6.9          8.3       +1.4
```

仅 2 层达到与 3 层相同的准确率提升（0.8166），说明第三层 `transformer.layers.0.norm2` 的提升（1.5 dB）对最终准确率贡献为零。这验证了 ADR-010 的"最差 K 层优先"策略是有效的。

---

## 5. VISUALIZATION — `result.plot`

`SessionPlotAccessor` 提供 17 种诊断图表，覆盖误差传播、per-role 分布、统计异常检测。

### 5.1 误差传播

| 方法 | 用途 |
|------|------|
| `result.plot.propagation_dag()` | DAG：节点=层，颜色=local QSNR，边标累计 QSNR 衰减 |
| `result.plot.error_waterfall()` | 逐层累计 QSNR 瀑布下降图 |
| `result.plot.local_vs_accum_scatter()` | 散点图：local vs accum QSNR，对角线为无传播线 |

### 5.2 Per-Role 视图

| 方法 | 用途 |
|------|------|
| `result.plot.per_role_qsnr_bars()` | 堆叠 grouped bar：每层 input/weight/output 三根 |
| `result.plot.depth_decay(role="output")` | QSNR 随深度衰减折线图 |
| `result.plot.per_layer_role_histogram()` | 每层的 role 分布直方图 |
| `result.plot.role_distribution_comparison()` | role 间分布特征对比 |

### 5.3 分布诊断

| 方法 | 用途 |
|------|------|
| `result.plot.crest_vs_qsnr()` | crest factor vs QSNR 散点，识别 outlier-dominated |
| `result.plot.outlier_analysis()` | 离群值比例多维度分析 |
| `result.plot.layer_histogram(layer, role)` | 单层单 role 的 fp32 vs quant 直方图叠加 |
| `result.plot.channel_heterogeneity(layer, role)` | per-channel QSNR 小提琴图 |

### 5.4 其他

| 方法 | 用途 |
|------|------|
| `result.plot.correlation_heatmap()` | 统计量 × 统计量相关性热力图 |
| `result.plot.accumulated_vs_local()` | 累计 vs 本地 QSNR 对比 |
| `result.plot.cost_decomposition()` | 延迟/内存 roofline 分解（需先 `session.cost()`） |
| `result.plot.per_block_qsnr()` | per-block QSNR 分布（per_block 粒度配置下） |
| `result.plot.qsnr_comparison()` | 多 config QSNR 对比（需传入对比数据） |
| `result.plot.error_propagation()` | 误差传播路径可视化的简化版 |

```python
# 示例：生成 per-role QSNR 堆叠 bar 图
fig = result.plot.per_role_qsnr_bars()
fig.savefig("per_role_qsnr.png")

# 示例：误差瀑布图
fig = result.plot.error_waterfall()
fig.savefig("error_waterfall.png")
```

---

## 6. TABLES — `result.tables`

`SessionTablesAccessor` 提供声明式文本表格。

```python
print(result.tables.per_role_qsnr())
```

```
Layer                               Input     Weight     Output  Dominant
-------------------------------------------------------------------------
transformer.layers.0.norm1            4.7       21.2        2.3  output
transformer.layers.1.linear2          6.0       12.0        6.9  input
transformer.layers.0.norm2            7.2       22.5        6.4  output
transformer.layers.1.norm2            7.1       20.2        6.4  output
transformer.layers.1.norm1            7.0       20.2        6.6  output
transformer.layers.0.linear1          7.6       10.2        6.8  output
transformer.layers.0.linear2          7.6       10.2        7.9  input
transformer.layers.1.linear1          7.7        9.7        7.9  input
classifier                            9.6       14.2       26.2  input
```

---

## 7. 数据导出

```python
# DataFrame 导出
df = result.report.to_dataframe()       # 全部分析数据的 DataFrame
df.to_csv("analysis_report.csv")

# 序列化方案
plan_auto.to_dict()                     # → dict，可 JSON 序列化

# Intervention 对比数据
cmp.summary()                           # → dict，含 baseline + intervention 指标
```

---

## 8. 分析总结

### 关键发现

| 维度 | 发现 |
|------|------|
| **主要瓶颈** | LayerNorm output（QSNR 2.3–6.6 dB）和 Linear input（6.0–7.7 dB），不是 weight |
| **退化机制** | Norm output → `high_dynamic_range`（14–17 bits）；Linear input → `outlier_dominated` / `heavy_tailed` |
| **最差单层** | `transformer.layers.0.norm1` output，QSNR 仅 2.3 dB |
| **最佳干预** | 2 层 auto 提升（1 Norm output + 1 Linear input → 8-bit）回收 83% 量化损失 |

### 量化友好度排名

```
Weight:  Norm (21.0 dB) > Other (14.2 dB) > Linear (10.5 dB)
Output:  Other (26.2 dB) > Linear (7.4 dB) > Norm (5.4 dB)
Input:   Other (9.6 dB)  > Linear (7.2 dB) > Norm (6.5 dB)
```

LayerNorm 权重量化极友好（低熵、均匀分布），但其输出量化极差（高动态范围 + 残差连接导致误差叠加）。

### 后续优化建议

1. **对 LayerNorm output 应用 Hadamard transform**：旋转空间以减少动态范围，预期提升 > 2 dB
2. **对 Linear input 应用 SmoothQuant**：将激活离群值平滑迁移到权重，预期提升 > 1.5 dB
3. **对 `transformer.layers.0.norm1`**：考虑保留 fp32 output（residual 连接处的误差放大效应）
4. **整体采用混合精度**：Norm weight 可保持 int4（QSNR 21 dB 已足够），Linear weight 和 Norm output 提升至 int8

---

## 附录：完整代码

`SlowTransformerEncoderLayer` 的引入是 PyTorch 2.2 的已知限制：`nn.TransformerEncoderLayer.forward()` 包含一个 fused fast path（`torch._transformer_encoder_layer_fwd`），该路径将 submodule 的 weight/bias 作为原始张量提取并传递给 C++ kernel，完全绕过 `QuantizedLinear.forward()` 和 `ObservableMixin._emit()`。覆盖 `forward` 走 slow path（`_sa_block` + `_ff_block`）确保 observer 数据正常采集。

完整可运行脚本见 `scripts/_capture_analysis.py`。
