# microxcaling — 可组合量化框架

基于 [microsoft/microxcaling](https://github.com/microsoft/microxcaling) 的增量式重建。`mx/` 为只读参考，新代码全部在 `src/`。

## 核心概念

量化方案由**三轴**正交组合：`format × granularity × transform`

```
QuantScheme = format（数值格式） × granularity（量化粒度） × transform（前后变换）
```

算子级配置 `OpQuantConfig` 为每个 tensor 角色（input / weight / output / grad_*）各绑定一个 scheme。用户通过 `QuantConfig` dataclass 配置，IDE 自动补全所有字段。

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 全自动 pipeline

```python
import torch.nn as nn
from src.session import Session, QuantConfig

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

def eval_fn(m, data):
    return {"loss": sum(m(batch).sum() for batch in data).item()}

cfg = QuantConfig(name="int8-pc", w_format="int8",
                  w_granularity="per_channel", calibrator="percentile")

result = Session(model, cfg).run(calib_data, eval_fn=eval_fn)

# 查看结果
print(result.summary())                    # 一行摘要
print(result.accuracy_table())             # 精度对比表
for name, qsnr in result.top_k_qsnr(3):    # QSNR 最差的 3 层
    print(f"  {name}: {qsnr:.1f} dB")
```

### 2. 分步链式 API

```python
session = Session(model, cfg)

session.quantize(calib_data=calib_data)    # 构建量化模型
# session.qmodel 现在可用：output = session.qmodel(x)

session.calibrate(calib_data)              # MX per_block 自动跳过
session.analyze(calib_data, outputs="default")
session.evaluate(calib_data, eval_fn)
session.cost()

result = session.result

# 链式一行等价写法：
result = (Session(model, cfg).quantize().calibrate(calib_data)
          .analyze(calib_data).evaluate(calib_data, eval_fn).cost().result)
```

### 3. MX 格式 — 跳过校准直接推理

```python
cfg = QuantConfig(
    w_format="fp4_e2m1", w_granularity="per_block", w_block_size=32,
    a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
)
session = Session(model, cfg).quantize()
output = session(torch.randn(4, 128))  # MX scale 动态计算，无需 calibrate()
```

### 4. 只量化 MatMul 算子

```python
cfg = QuantConfig(
    w_format="int8",
    quantize_nonlinear=False,  # norm / activation / pool 保持 fp32
)
session = Session(model, cfg).quantize()
```

### 5. 多配置对比

```python
from src.session import Study

configs = [
    QuantConfig(name="int8", w_format="int8"),
    QuantConfig(name="int4", w_format="int4"),
    QuantConfig(name="fp4-mx", w_format="fp4_e2m1", w_granularity="per_block",
                w_block_size=32, a_format="fp4_e2m1", a_granularity="per_block",
                a_block_size=32),
]
report = Study(configs, model=model).run(calib_data, eval_fn=eval_fn, outputs="all")
report.save("results/")
```

### 6. Element-wise 存储量化

```python
cfg = QuantConfig(
    w_format="int8", w_granularity="per_channel",
    storage_bits=16, storage_kind="bfloat",  # 所有张量先过 bfloat16
)
```

## QuantConfig 完整字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | `""` | 配置名 |
| `w_format` | `str` | `"int8"` | 权重格式 |
| `w_granularity` | `str` | `"per_tensor"` | per_tensor / per_channel / per_block |
| `w_block_size` | `int\|None` | `None` | per_block 的 block 大小 |
| `w_axis` | `int` | `-1` | 权重量化轴 |
| `a_format` | `str\|None` | `None` | 激活格式（None = 同权重） |
| `a_granularity` | `str` | `"per_tensor"` | 激活量化粒度 |
| `a_block_size` | `int\|None` | `None` | 激活 per_block 大小 |
| `a_axis` | `int` | `-1` | 激活量化轴 |
| `transform` | `str` | `"none"` | none / hadamard / smoothquant / prescale |
| `sq_alpha` | `float` | `0.5` | SmoothQuant 平滑强度 |
| `prescale_init` | `str` | `"ones"` | prescale 初始化：ones / amax / pot_amax |
| `prescale_pot` | `bool` | `False` | prescale 投影到 2 的幂 |
| `prescale_granularity` | `str\|None` | `None` | None = 跟随 a_granularity |
| `lsq_steps` | `int` | `0` | LSQ 步数（>0 需 transform="prescale"） |
| `lsq_lr` | `float` | `1e-3` | LSQ 学习率 |
| `scale_storage` | `str` | `"fp32"` | Scale 存储格式：fp32 / pot |
| `calibrator` | `str` | `"mse"` | 校准策略：mse / max / percentile / kl |
| `storage_bits` | `int` | `0` | Element-wise 存储位宽（0=禁用） |
| `storage_kind` | `str` | `"bfloat"` | 存储类型：bfloat / fp |
| `weight_only` | `bool` | `False` | 仅量化权重 |
| `quantize_nonlinear` | `bool` | `True` | False = 非线性算子保持 fp32 |

## 支持的数值格式

| 格式 | 注册名 | 说明 |
|------|--------|------|
| int8 / int4 / int2 | `"int8"` `"int4"` `"int2"` | 对称整数 |
| fp8_e4m3 / fp8_e5m2 | `"fp8_e4m3"` `"fp8_e5m2"` | OCP FP8 |
| fp6_e3m2 / fp6_e2m3 | `"fp6_e3m2"` `"fp6_e2m3"` | MX FP6 |
| fp4_e2m1 | `"fp4_e2m1"` (`"fp4"`) | MX FP4 |
| nf4 | `"nf4"` | QLoRA 正态优化 4-bit LUT |
| bfloat16 / float16 | `"bfloat16"` `"float16"` | 硬件快捷路径 |

添加新格式：[`register_format()`](src/formats/registry.py)，不改核心代码。

## 项目结构

```
src/
├── formats/         # FormatBase 及各格式实现
├── scheme/          # QuantScheme、GranularitySpec、OpQuantConfig
├── quantize/        # 核心量化函数
├── ops/             # 量化算子（Linear / Conv / Norm / Activation 等）
├── session/         # 驱动层：QuantConfig / Session / Study / quantize_model
├── calibration/     # 校准管线（策略 + CalibrationSession + LSQ）
├── analysis/        # 误差分析（AnalysisContext / Observer / e2e comparison）
├── observer/        # Observer 横切基础设施
├── transform/       # Hadamard / SmoothQuant / PreScale 变换
├── report/          # 输出层：SessionReport / StudyReport
├── cost/            # 性能估算
├── viz/             # 可视化
└── onnx/            # ONNX 导出
mx/                  # 原始 microsoft/microxcaling（只读）
```

## Transform 支持

| Transform | 效果 | 使用方式 |
|-----------|------|---------|
| `"none"` | 无变换 | 默认 |
| `"hadamard"` | Hadamard 正交旋转，分散 outlier | `transform="hadamard"` |
| `"smoothquant"` | 平滑 activation outlier → weight | `transform="smoothquant"` |
| `"prescale"` | 可学习前置 scale（+ 可选 LSQ） | `transform="prescale"` + `lsq_steps=N` |

## 低层 API

日常使用推荐上面的 `Session` + `QuantConfig`。需要精细控制时使用：

```python
from src.session import QuantSession, quantize_model
from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec

# 直接构造 OpQuantConfig
fmt = FormatBase.from_str("int8")
scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_channel(axis=0))
cfg = OpQuantConfig(input=scheme, weight=scheme)

# 方式 A：QuantSession（工作流控制）
qs = QuantSession(model, cfg)
with qs.calibrate():
    for batch in calib_data:
        qs(batch)
qs.export_onnx("model.onnx", dummy_input=torch.randn(1, 128))

# 方式 B：quantize_model（只做模块替换，不要工作流）
qmodel = quantize_model(model, cfg)
output = qmodel(x)
```

## API 层级总结

| 入口 | 输入 | 适用场景 |
|------|------|---------|
| `Session(model, QuantConfig)` | 字符串字段，IDE 补全 | **推荐**：日常量化实验 |
| `Study(configs, model)` | `List[QuantConfig]` | 多配置精度对比 |
| `QuantSession(model, OpQuantConfig)` | 对象 | 精细工作流控制 |
| `quantize_model(model, OpQuantConfig)` | 对象 | 只要模块替换，不要工作流 |

## 测试

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
# 2034 passed
```

## 文档索引

→ **[docs/INDEX.md](docs/INDEX.md)** — 完整文档导航

| 文档 | 内容 |
|------|------|
| [快速开始详解](docs/reference/quickstart-details.md) | Transform、Calibration、QAT、ONNX、LSQ 等进阶用法 |
| [架构决策文档](docs/architecture/) | ADR-001 ~ ADR-008 |
| [开发规范](docs/INDEX.md) | 开发准则、规范、流程、Phase 计划 |
