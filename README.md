# microxcaling — 可组合量化框架

基于 [microsoft/microxcaling](https://github.com/microsoft/microxcaling) 的增量式重建。将量化分解为**格式 × 粒度 × 变换**三个正交轴，通过 `QuantConfig` 一个 dataclass 控制一切。

**特性**：三轴正交组合 · 全算子覆盖（Linear/Conv/Norm/Activation/Softmax/Pool） · MX per-block 位精确等价 · 4 种校准策略 · 4 种 Transform · LSQ 可学习量化 · ONNX 导出 · 误差分析 · 性能估算

## 安装

```bash
pip install -r requirements.txt
```

## 30 秒快速体验

```python
import torch
import torch.nn as nn
from src.session import Session, QuantConfig

# 准备模型和校准数据
model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

def eval_fn(m, data):
    return {"loss": sum(m(batch).sum() for batch in data).item()}

# 一行配置 + 一键运行
result = Session(model, QuantConfig(name="int8", w_format="int8")).run(
    calib_data, eval_fn=eval_fn)

# 查看结果
print(result.summary())                     # 一行摘要
print(result.accuracy_table())              # 精度对比表
for name, qsnr in result.top_k_qsnr(3):     # QSNR 最差的 3 层
    print(f"  {name}: {qsnr:.1f} dB")
```

## 核心概念

### 三轴正交模型

量化的每一个环节被拆成三个独立、可任意组合的轴：

```
输入 x ──→ [Transform] ──→ [Format × Granularity] ──→ [Inverse Transform] ──→ x_q
```

| 轴 | 回答的问题 | 例子 |
|----|-----------|------|
| **Format**（格式） | 用什么数字表示？ | int8, fp8_e4m3, fp4_e2m1, nf4 |
| **Granularity**（粒度） | scale 多细？一份 scale 管多少元素？ | per_tensor, per_channel, per_block(32) |
| **Transform**（变换） | 量化前/后对数据做什么预处理？ | 无, Hadamard, SmoothQuant, PreScale |

三个轴完全正交——任意 Format 可以配任意 Granularity，再加任意 Transform，不需要改任何核心代码。

### Session 管道

一次量化实验经过一条固定的管道：

```
quantize() → calibrate() → analyze() → evaluate() → cost() → result
   构建量化模型   计算scale    误差分析     精度评估     性能估算    汇总结果
```

你可以一次调用 `Session.run()` 跑完整条管道，也可以分步调用每一步来检查和调试。

---

# 使用教程

## 1. 基础量化

最简单的入口：只量化权重的 INT8。

```python
import torch
import torch.nn as nn
from src.session import Session, QuantConfig

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

cfg = QuantConfig(w_format="int8", weight_only=True)
result = Session(model, cfg).run(calib_data)

print(result.summary())
# Config: (unnamed) | avg QSNR=35.2 dB
```

`weight_only=True` 表示只量化权重，激活保持 fp32。这是风险最低的量化方式。

## 2. 配置不同格式

换量化格式只需要改 `w_format` / `a_format` 字符串。所有支持格式：

| 格式 | 注册名 | 位宽 | 适用场景 |
|------|--------|------|---------|
| INT8 | `"int8"` | 8 | 通用，兼容性最好 |
| INT4 | `"int4"` | 4 | 高压缩比，需校准 |
| INT2 | `"int2"` | 2 | 极限压缩，精度损失大 |
| FP8 E4M3 | `"fp8_e4m3"` | 8 | OCP 标准，训练/推理通用 |
| FP8 E5M2 | `"fp8_e5m2"` | 8 | 更大动态范围 |
| FP6 E3M2 | `"fp6_e3m2"` | 6 | MX 规格 |
| FP6 E2M3 | `"fp6_e2m3"` | 6 | MX 规格，更高精度 |
| FP4 E2M1 | `"fp4_e2m1"` | 4 | MX 规格，极限压缩 |
| NF4 | `"nf4"` | 4 | QLoRA 正态优化 LUT |
| BF16 | `"bfloat16"` | 16 | 硬件捷径，几乎无损失 |
| FP16 | `"float16"` | 16 | IEEE 半精度 |

同时量化权重和激活（W/A 全量化）：

```python
import torch
import torch.nn as nn
from src.session import Session, QuantConfig

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

# INT4 权重 + INT8 激活
cfg = QuantConfig(w_format="int4", a_format="int8")
result = Session(model, cfg).run(calib_data)

# FP8 全量化
cfg = QuantConfig(w_format="fp8_e4m3", a_format="fp8_e4m3")

# NF4 权重
cfg = QuantConfig(w_format="nf4", weight_only=True)
```

## 3. 配置不同粒度

粒度决定了量化 scale 的精细程度。粒度越细，精度越高，但 scale 开销越大。

```python
import torch
import torch.nn as nn
from src.session import Session, QuantConfig

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

# per_tensor：整个张量共享一个 scale（最粗糙但最省）
cfg = QuantConfig(w_format="int8", w_granularity="per_tensor")

# per_channel：每个通道一个 scale（CNN/Linear 权重标准做法）
cfg = QuantConfig(w_format="int8", w_granularity="per_channel")

# per_block：每 32 个元素共享一个指数（MX 规格）
cfg = QuantConfig(
    w_format="fp4_e2m1", w_granularity="per_block", w_block_size=32,
    a_format="fp4_e2m1", a_granularity="per_block", a_block_size=32,
)
result = Session(model, cfg).run(calib_data)
```

**粒度选择建议**：

| 场景 | 推荐粒度 |
|------|---------|
| 原型验证、最低风险 | per_tensor |
| 权重部署（CNN/Transformer） | per_channel |
| 激活量化、MX 规格 | per_block(32) |

MX per_block 格式的 scale 在推理时动态计算，因此 **`calibrate()` 会自动跳过**。

## 4. 查看量化结果

`Session.run()` 返回的 `SessionResult` 提供多层级的查看方式。

```python
import torch
import torch.nn as nn
from src.session import Session, QuantConfig

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

def eval_fn(m, data):
    return {"loss": sum(m(batch).sum() for batch in data).item()}

cfg = QuantConfig(name="int8", w_format="int8", w_granularity="per_channel")
result = Session(model, cfg).run(calib_data, eval_fn=eval_fn)

# 1. 一行摘要
print(result.summary())
# Config: int8 | loss: fp32=14.2341 quant=15.1076 | avg QSNR=34.2 dB

# 2. 精度对比表
print(result.accuracy_table())
# Metric       FP32       Quant      Δ
# ---------------------------------------
# loss         14.2341    15.1076    +0.8735

# 3. 质量最差的层（定位问题）
for name, qsnr in result.top_k_qsnr(3):
    print(f"  {name}: {qsnr:.1f} dB")

# 4. 逐层 DataFrame
df = result.layer_report()
print(df.sort_values("qsnr_db").head(5))
```

**Result 对象属性速查**：

| 属性/方法 | 返回 | 说明 |
|-----------|------|------|
| `result.summary()` | `str` | 一行摘要 |
| `result.accuracy_table()` | `str` | FP32 vs Quant 对比表 |
| `result.top_k_qsnr(k)` | `List[Tuple]` | QSNR 最差的 k 层 |
| `result.layer_report()` | `DataFrame` | 逐层 QSNR/MSE |
| `result.qsnr_per_layer` | `dict` | `{层名: QSNR dB}` |
| `result.mse_per_layer` | `dict` | `{层名: MSE}` |
| `result.fp32_metrics` | `dict` | 原始模型指标 |
| `result.quant_metrics` | `dict` | 量化模型指标 |
| `result.delta` | `dict` | 精度损失（fp32 - quant） |

## 5. 多配置对比与可视化

对比多个量化配置时，`Study` → `StudyReport` 是你的核心工具链。设计理念：**先收集数据，后消费数据**——report 对象不绑定任何特定的输出格式，你可以终端打印、调用内置图表、导出 DataFrame、或保存到磁盘，按需组合。

### 5.1 终端对比表

```python
import torch
import torch.nn as nn
from src.session import QuantConfig, Study

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

def eval_fn(m, data):
    return {"loss": sum(m(batch).sum() for batch in data).item()}

configs = [
    QuantConfig(name="int8", w_format="int8"),
    QuantConfig(name="int4", w_format="int4"),
    QuantConfig(name="fp4-mx", w_format="fp4_e2m1", w_granularity="per_block",
                w_block_size=32, a_format="fp4_e2m1", a_granularity="per_block",
                a_block_size=32),
]

report = Study(configs, model=model).run(calib_data, eval_fn=eval_fn)
report.print_summary()

# 输出：
# ======================================================================
#   Part: int8
# ======================================================================
#   Config                    Avg QSNR     Avg MSE      Acc Delta
#   ----------------------------------------------------------------
#   int8                      34.21        0.001200     loss=-0.0300, acc=-0.0200
#   int4                      22.50        0.005000     loss=-0.1000, acc=-0.1100
#   ...
```

`Study` 对每个 config 创建一个独立的 `Session`，model 自动 deepcopy 互不干扰。

### 5.2 内置图表

`StudyReport` 通过 `.plot` 访问器提供内置图表，每个方法返回 `matplotlib.Figure`——你可以 `plt.show()` 或 `savefig()`。

```python
# QSNR 逐层对比折线图（每个 config 一条线）
fig = report.plot.qsnr_comparison()
fig.savefig("qsnr_comparison.png", dpi=150)

# Crest Factor vs QSNR 散点图（需要 DistributionObserver + QSNRObserver 数据）
fig = report.plot.crest_vs_qsnr(role="input")
fig.savefig("crest_vs_qsnr_input.png", dpi=150)
```

图表内置优雅降级：如果数据不足（比如没开 DistributionObserver），会显示提示文字而不是崩溃。

要启用完整分析数据，通过 `outputs` 参数指定需要的 observer：

```python
# 打开 QSNR + MSE + Distribution 三个 observer
report = Study(configs, model=model).run(
    calib_data, eval_fn=eval_fn,
    outputs=["qsnr", "error_dist"]   # qsnr → QSNRObserver, error_dist → DistributionObserver + MSEObserver
)
```

### 5.3 导出 DataFrame 自定义分析

`report.to_dataframe()` 将所有结果展开为一张**整洁 DataFrame**——每行一个 `(part, config, format, layer, role)`，列包含所有 observer 采集的指标（`qsnr_db`、`mse`、`crest_factor`、`peak`、`rms`、`mean`、`std`、`skewness`、`kurtosis`……）。

DataFrame 是通用数据交换格式——你不需要学框架的绘图 API，用自己熟悉的工具分析：

```python
df = report.to_dataframe()

# 用 pandas 做任意分析
print(df.groupby("config")["qsnr_db"].mean())           # 每个配置的平均 QSNR
print(df[df["qsnr_db"] < 20][["layer", "role", "qsnr_db"]])  # 找出质量差的层

# 对接任意可视化库
import seaborn as sns

sns.boxplot(data=df, x="config", y="qsnr_db")           # QSNR 分布
sns.scatterplot(data=df, x="crest_factor", y="qsnr_db",  # Crest vs QSNR
                hue="config", style="role")
```

### 5.4 保存到磁盘

`report.save("results/")` 一键生成标准输出。只生成有数据的文件——没跑 eval_fn 就不产生 accuracy.csv，没开 DistributionObserver 就不产生 crest 散点图。

```
results/
├── results.json                    # 完整结构化数据
├── tables/
│   └── accuracy.csv                # 精度对比表（需要 eval_fn）
└── figures/
    ├── qsnr_comparison.png         # QSNR 逐层对比（需要 qsnr_db）
    ├── crest_vs_qsnr_input.png     # Crest vs QSNR 散点（需要 crest_factor）
    └── crest_vs_qsnr_weight.png
```

```python
report.save("results/")

# 从保存的结果重新加载
from src.report import StudyReport
reloaded = StudyReport.from_file("results/")
```

## 6. ONNX 导出

量化后一行导出 ONNX，部署到推理引擎。

```python
import torch
import torch.nn as nn
from src.session import Session, QuantConfig

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

cfg = QuantConfig(w_format="int8", w_granularity="per_channel")
session = Session(model, cfg).quantize(calib_data=calib_data)

# 一行导出
session.qmodel.export_onnx(torch.randn(1, 128), "model.onnx")
```

导出的 ONNX 图中：
- **int/fp8 格式** → 标准 `QuantizeLinear` / `DequantizeLinear` 节点
- **MX per_block 格式** → `com.microxscaling::MxQuantize` 自定义算子

也可以通过底层 API 导出：

```python
from src.onnx import export_quantized_model
export_quantized_model(session.qmodel, torch.randn(1, 128), "model.onnx")
```

## 7. Transform：处理 Outlier

当激活或权重中有 outlier 导致量化误差大时，用 Transform 在量化前做预处理。

```python
import torch
import torch.nn as nn
from src.session import Session, QuantConfig

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

# SmoothQuant：把激活 outlier 平滑到权重
cfg = QuantConfig(w_format="int8", transform="smoothquant")

# Hadamard：正交旋转分散 outlier
cfg = QuantConfig(w_format="int8", transform="hadamard")

# PreScale：在量化和反量化之间插入可学习 scale
cfg = QuantConfig(w_format="int8", transform="prescale")

session = Session(model, cfg).quantize(calib_data=calib_data)
```

| Transform | 原理 | 适用场景 |
|-----------|------|---------|
| `"none"` | 不做处理 | 默认，大多数情况 |
| `"hadamard"` | Hadamard 正交旋转，O(n log n) | 激活有聚集 outlier |
| `"smoothquant"` | 激活平滑因子迁移到权重 | LLM 激活 outlier 严重时 |
| `"prescale"` | 前置可学习 scale（+ LSQ 优化） | 需要极致精度，可配合 `lsq_steps` |

**SmoothQuant 在使用时需要注意**：`session.quantize(calib_data=...)` 必须传入校准数据，因为 SmoothQuant 需要用校准数据计算平滑因子并融合到权重。

## 8. 分步链式 API

除了 `run()` 一键跑完，你也可以分步执行来检查每一阶段。

```python
import torch
import torch.nn as nn
from src.session import Session, QuantConfig

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

def eval_fn(m, data):
    return {"loss": sum(m(batch).sum() for batch in data).item()}

cfg = QuantConfig(name="int8", w_format="int8", w_granularity="per_channel")
session = Session(model, cfg)

# 分步执行
session.quantize(calib_data=calib_data)   # 构建量化模型
# session.qmodel 现在可用：output = session.qmodel(batch)

session.calibrate(calib_data)             # 计算 scale（MX per_block 自动跳过）
session.analyze(calib_data)               # 误差分析（QSNR / MSE）
session.evaluate(calib_data, eval_fn)     # 精度评估
session.cost()                            # 性能估算

result = session.result
print(result.summary())
```

所有步骤方法返回 `self`，可以链式调用：

```python
result = (Session(model, cfg).quantize(calib_data=calib_data)
          .calibrate(calib_data).analyze(calib_data)
          .evaluate(calib_data, eval_fn).cost().result)
```

**推理模式切换**：

```python
session = Session(model, cfg).quantize()
output = session(x)            # 量化模型推理

session.use_fp32()
output = session(x)            # 切换回 fp32 模型推理

session.use_quant()
output = session(x)            # 再切回量化模型

print(session.mode)            # "quant" 或 "fp32"
```

**Session 不会修改原始模型**：`quantize()` 内部对模型做 deepcopy，原模型不变。同一个模型可以安全地传给多个 Session。

---

# 高层 API 参考

## Session

单次量化实验的完整生命周期管理。

| 方法 | 说明 |
|------|------|
| `Session(model, cfg)` | 构造。`model` 是 `nn.Module`，`cfg` 是 `QuantConfig` |
| `.quantize(*, calib_data=None)` | 构建量化模型。`calib_data` 在 SmoothQuant/Prescale 时必传 |
| `.calibrate(calib_data, *, eval_fn=None)` | 校准 scale。MX per_block 自动跳过 |
| `.analyze(calib_data, *, outputs="default", eval_fn=None)` | 误差分析。`outputs` = `"default"` / `"all"` / `["qsnr", "mse"]` |
| `.evaluate(eval_data, eval_fn)` | 精度评估。`eval_fn(model, data) -> Dict[str, float]` |
| `.cost()` | 性能估算（延迟 + 内存） |
| `.run(calib_data, *, eval_data, eval_fn, outputs)` | 一键跑完整条管道 |
| `session(x)` | 量化模型推理（`__call__`） |
| `.use_fp32()` / `.use_quant()` | 切换推理模式 |
| `.qmodel` | 量化后的 `nn.Module` |
| `.fp32_model` | 原始模型（`keep_fp32=True` 时） |
| `.mode` | 当前推理模式（`"fp32"` 或 `"quant"`） |
| `.result` | 构建 `SessionResult` |

## QuantConfig 完整字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | `""` | 配置名 |
| `w_format` | `str` | `"int8"` | 权重格式 |
| `w_granularity` | `str` | `"per_tensor"` | per_tensor / per_channel / per_block |
| `w_block_size` | `int\|None` | `None` | per_block 的 block 大小 |
| `w_axis` | `int` | `-1` | 权重量化轴 |
| `a_format` | `str\|None` | `None` | 激活格式（None = 同权重） |
| `a_granularity` | `str` | `"per_tensor"` | 激活粒度 |
| `a_block_size` | `int\|None` | `None` | 激活 per_block 大小 |
| `a_axis` | `int` | `-1` | 激活量化轴 |
| `transform` | `str` | `"none"` | none / hadamard / smoothquant / prescale |
| `sq_alpha` | `float` | `0.5` | SmoothQuant 平滑强度 |
| `prescale_init` | `str` | `"ones"` | prescale 初始化：ones / amax / pot_amax |
| `prescale_pot` | `bool` | `False` | prescale 投影到 2 的幂 |
| `prescale_granularity` | `str\|None` | `None` | None = 跟随 a_granularity |
| `lsq_steps` | `int` | `0` | LSQ 优化步数（>0 需 transform="prescale"） |
| `lsq_lr` | `float` | `1e-3` | LSQ 学习率 |
| `scale_storage` | `str` | `"fp32"` | scale 存储格式：fp32 / pot |
| `calibrator` | `str` | `"mse"` | 校准策略：mse / max / percentile / kl |
| `storage_bits` | `int` | `0` | Element-wise 存储位宽（16=bf16，0=禁用） |
| `storage_kind` | `str` | `"bfloat"` | 存储类型：bfloat / fp |
| `weight_only` | `bool` | `False` | 仅量化权重 |
| `quantize_nonlinear` | `bool` | `True` | False = 非线性算子保持 fp32 |

## Study

多配置聚合对比。

```python
from src.session import Study, QuantConfig

configs = [
    QuantConfig(name="int8", w_format="int8"),
    QuantConfig(name="int4", w_format="int4"),
]

study = Study(configs, model=model)
report = study.run(calib_data, eval_fn=eval_fn, outputs="all")
report.print_summary()
report.save("results/")
```

`Study` 对每个 config 创建一个独立的 `Session`，model 自动 deepcopy 互不干扰。返回的 `StudyReport` 支持 `print_summary()`、`save()` 和 `from_file()` 重新加载。

## SessionResult

单次实验的结果对象。

| 方法/属性 | 返回 | 说明 |
|-----------|------|------|
| `.summary()` | `str` | 单行摘要 |
| `.accuracy_table()` | `str` | FP32 vs Quant 对比表 |
| `.top_k_qsnr(k=10)` | `List[(name, dB)]` | QSNR 最差的 k 层 |
| `.layer_report()` | `DataFrame` | 逐层 QSNR + MSE |
| `.qsnr_per_layer` | `dict` | `{层名: float}` |
| `.mse_per_layer` | `dict` | `{层名: float}` |
| `.fp32_metrics` | `dict` | `eval_fn` 在 fp32 模型上的输出 |
| `.quant_metrics` | `dict` | `eval_fn` 在量化模型上的输出 |
| `.delta` | `dict` | 精度差（fp32 - quant） |

---

# 进阶主题

## 校准策略

校准决定了每个量化层的 scale。四种策略通过 `QuantConfig.calibrator` 选择。

| 策略 | 配置值 | 原理 | 适用场景 |
|------|--------|------|---------|
| MSE | `"mse"` | 网格搜索最小化 MSE（默认） | 通用，鲁棒 |
| Max | `"max"` | scale = max(\|x\|) | 快速、保守 |
| Percentile | `"percentile"` | scale = N 分位数（q=99） | 有 outlier 但不想丢太多信息 |
| KL | `"kl"` | 最小化 KL 散度 | 分布敏感任务（分类） |

```python
import torch
import torch.nn as nn
from src.session import Session, QuantConfig

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

# 比较不同校准策略
for cal in ["mse", "max", "percentile", "kl"]:
    cfg = QuantConfig(name=cal, w_format="int4", calibrator=cal)
    result = Session(model, cfg).run(calib_data)
    print(f"{cal}: {result.summary()}")
```

## LSQ：可学习量化

当静态校准不够时，用 LSQ（Learned Step Size Quantization）通过梯度下降学习最优 pre-scale。

```python
import torch
import torch.nn as nn
from src.session import Session, QuantConfig

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

# LSQ：prescale transform + lsq_steps > 0
cfg = QuantConfig(
    w_format="int4", w_granularity="per_channel",
    transform="prescale",
    lsq_steps=100,      # 每层 100 步梯度优化
    lsq_lr=1e-3,
    prescale_init="amax",
)
result = Session(model, cfg).run(calib_data)
```

LSQ 使用逐层（layer-wise）BRECQ 式优化：先跑前面层的量化输入，再梯度优化当前层的 pre-scale，最小化量化输出和 fp32 输出的 MSE。

## 误差分析体系

框架内置 4 种 Observer，在 `analyze()` 阶段自动挂载，记录每层每阶段的量化误差。

| Observer | Output Key | 测量内容 |
|----------|------------|---------|
| `QSNRObserver` | `"qsnr"` | 量化信噪比 `10*log10(\|\|fp32\|\|² / \|\|fp32-quant\|\|²)` dB |
| `MSEObserver` | `"mse"` | 均方误差 |
| `HistogramObserver` | `"histogram"` | fp32 / quant / error 三通道直方图 |
| `DistributionObserver` | `"distribution"` | 统计指纹：mean/std/skewness/kurtosis + peak/rms/crest_factor + 稀疏度/outlier/动态范围 |

通过 `outputs` 控制打开哪些分析：

```python
# 只做 QSNR（默认）
result = Session(model, cfg).run(calib_data, outputs="default")

# 全部打开
result = Session(model, cfg).run(calib_data, outputs="all")

# 自定义
result = Session(model, cfg).run(calib_data, outputs=["qsnr", "histogram"])
```

### Observer 的底层用法

Observer 可以脱离 Session 独立使用，直接挂载到任意模型上：

```python
from src.analysis import QSNRObserver, MSEObserver, AnalysisContext

observers = [QSNRObserver(), MSEObserver()]
with AnalysisContext(model, observers) as ctx:
    for batch in calib_data:
        model(batch)
report = ctx.report()
```

### 分布分析与误差关联

拿到 observer 数据后，可以对分布指纹做聚合分析：

```python
from src.analysis.correlation import (
    DistributionProfile,      # 按 role 聚合分布统计（p50/p25/p75…）
    DistributionTaxonomy,     # 将层自动分为 8 种分布类型
    ErrorByDistribution,      # 关联动态范围与量化误差
    LayerSensitivity,         # 按层类型/阈值找出最敏感的层
)

# 从分析报告构建分布画像
profile = DistributionProfile.from_report(report)
print(profile.by_role("input"))   # 输入激活的统计摘要
profile.print_profile()

# 自动分类层到分布类型
taxonomy = DistributionTaxonomy.from_report(report)
taxonomy.print_taxonomy()           # 终端表格
taxonomy.print_taxonomy(ascii_plots=True)  # ASCII 分布图
exemplars = taxonomy.get_exemplars("heavy-tailed", n=3)  # 典型层

# 找出误差最大的层
eb = ErrorByDistribution(report)
for layer, role, qsnr in eb.rank_layers(by="qsnr_db", k=5, ascending=True):
    print(f"{layer}/{role}: {qsnr:.1f} dB")

# 按动态范围分组统计
groups = eb.group_by_range(role="input", bins=[0, 4, 7, 999])
for name, info in groups.items():
    print(f"{name}: avg_qsnr={info['avg_qsnr']:.1f} dB, {info['verdict']}")

# 最敏感的层
sens = LayerSensitivity(report)
for layer, role, mse in sens.topk(k=5, metric="mse"):
    print(f"{layer}/{role}: MSE={mse:.6f}")
```

## 自定义格式

一行注册新格式，不改核心代码。

```python
from src.formats import register_format, register_float_format, register_int_format

# 注册自定义浮点格式
register_float_format("fp5_e3m1", ebits=3, mbits=1)

# 注册自定义整数格式
register_int_format("int6", bits=6)

# 注册任意 FormatBase 子类
from src.formats.base import FormatBase
register_format("my_format", MyFormatInstance)

# 之后就可在 QuantConfig 中使用
cfg = QuantConfig(w_format="fp5_e3m1")
```

自动解析：任何 `fp<N>_e<E>m<M>` 或 `int<N>` 字符串在首次使用时自动注册，无需手动调用。

## 性能估算

`cost()` 使用 Roofline 模型估计延迟和内存占用。

```python
import torch
import torch.nn as nn
from src.session import Session, QuantConfig
from src.cost import DeviceSpec, analyze_model_cost

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

# 方式 1：Session 内置
session = Session(model, QuantConfig(w_format="int8")).quantize().cost()
print(session.result.cost.print_summary())

# 方式 2：独立调用
fp32_cost = analyze_model_cost(model, {"x": (1, 128)}, DeviceSpec.a100())
```

---

# 底层 API

日常使用推荐上面的 `Session` + `QuantConfig`。需要精细控制时，使用底层 API。

## quantize_model：只要模块替换

```python
import torch
import torch.nn as nn
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.session import quantize_model

model = nn.Linear(128, 256)

# 构建底层配置
scheme = QuantScheme.per_channel("int8", axis=0)
cfg = OpQuantConfig(weight=scheme)

# 替换模块 → 量化模型
qmodel = quantize_model(model, cfg)
output = qmodel(torch.randn(4, 128))  # 自动走量化路径
qmodel.export_onnx(torch.randn(1, 128), "model.onnx")
```

`quantize_model` 只做模块替换（`nn.Linear` → `QuantizedLinear` 等），不提供工作流（校准、分析、评估）。适合集成到自己的训练/推理脚本中。

## OpQuantConfig：算子级逐角色配置

为每个 tensor 角色（input/weight/output/grad_*）单独指定 QuantScheme。

```python
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.formats.base import FormatBase

# 方案 A：用 static factory
w_scheme = QuantScheme.per_channel("int4", axis=0)
a_scheme = QuantScheme.per_tensor("int8")

cfg = OpQuantConfig(input=a_scheme, weight=w_scheme)
# storage / output / grad_* 全部为 None（不量化）

# 方案 B：带 storage（两级模型）
from src.formats.bf16_fp16 import BFloat16Format

storage = QuantScheme.per_tensor(BFloat16Format())
cfg = OpQuantConfig(input=a_scheme, weight=w_scheme, storage=storage)
```

## 三种入口对比

| 入口 | 输入 | 适用场景 |
|------|------|---------|
| `Session(model, QuantConfig)` | 字符串字段，IDE 补全 | **推荐**：日常量化实验 |
| `Study(configs, model)` | `List[QuantConfig]` | 多配置精度对比 |
| `quantize_model(model, OpQuantConfig)` | 对象 | 集成到自定义脚本 |

---

# 与 Microsoft MX 的位精确等价性

本框架与 [microsoft/microxcaling](https://github.com/microsoft/microxcaling) 的 MX 推理输出**位精确等价**（`torch.equal`），已通过全算子验证。

```python
import torch
import torch.nn as nn
import mx
from mx.specs import apply_mx_specs
from src.session import Session, QuantConfig

linear = nn.Linear(64, 128, bias=True).eval()
x = torch.randn(4, 64)

# ═══ 本框架 ═══
cfg = QuantConfig(name="w8a8", w_format="int8", w_granularity="per_block",
                  w_block_size=32, a_format="int8", a_granularity="per_block",
                  a_block_size=32, storage_bits=16, storage_kind="bfloat")
session_out = Session(linear, cfg).quantize()(x)

# ═══ 微软 MX ═══
mx_specs = apply_mx_specs({"bfloat": 16, "w_elem_format": "int8",
                            "a_elem_format": "int8", "block_size": 32})
mx_out = mx.linear(x, linear.weight, linear.bias, mx_specs=mx_specs)

assert torch.equal(session_out, mx_out)  # bit-exact ✓
```

等价性覆盖范围：
- **21 种模块** + **10 种 inline op**（matmul、add、softmax 等）
- **8 种格式**（int8/int4/int2/fp8_e4m3/fp8_e5m2/fp6_e3m2/fp6_e2m3/fp4_e2m1）
- **3 种 storage**（bfloat16 / float16 / disabled）
- **forward + backward**（STE 梯度对齐，4/5 规格 bit-exact）

---

# FAQ

### Session 会修改我的原始模型吗？

不会。`quantize()` 内部对模型做 deepcopy，原始模型保持不变。同一个模型可以安全地传给多个 Session。

### MX 格式需要 calibrate 吗？

不需要。MX per_block 的 scale（shared exponent）在推理时动态计算，`calibrate()` 会自动检测并跳过。

### 如何只量化权重，不量化激活？

`QuantConfig(weight_only=True)`。此时只有权重被量化，激活保持 fp32。

### 如何只量化 MatMul 算子，保持 Norm/Activation 为 fp32？

`QuantConfig(quantize_nonlinear=False)`。Linear 和 Conv 正常量化，但 Norm、Activation、Pool 保持 fp32。

### 量化后的模型能做 backward 训练吗？

可以。QAT（Quantization-Aware Training）通过 STE（Straight-Through Estimator）支持。配置 backward 角色（`grad_input`、`grad_weight`）后，`loss.backward()` 正常通过量化算子。

### 支持哪些 ONNX opset？

默认 opset 17。Int/fp8 格式导出为标准 QDQ 节点，MX per_block 导出为 `com.microxscaling::MxQuantize` 自定义算子。注意导出的 scale 是占位符常量（1.0），图结构有效但不可直接推理。

### 可以自定义量化格式吗？

可以。调用 `register_format("my_format", instance)` 注册，然后在 `QuantConfig(w_format="my_format")` 中使用。也支持自动解析：任何 `fp<N>_e<E>m<M>` 字符串在首次使用时自动注册。

---

# 更多文档

→ **[docs/INDEX.md](docs/INDEX.md)** — 完整文档导航

| 文档 | 内容 |
|------|------|
| [架构决策文档](docs/architecture/) | ADR-001 ~ ADR-008，理解设计意图 |
| [开发规范](docs/standards/) | API 设计、新增格式/Transform/Observer |
| [开发流程](docs/workflow/) | 功能生命周期、分支策略、Commit 规范 |
| [公式参考](docs/reference/) | 量化公式的权威定义 |
| [数学验证](docs/verification/) | 量化测试的数学推导 |
| [Review 记录](docs/reviews/) | 历史缺陷排查 |
