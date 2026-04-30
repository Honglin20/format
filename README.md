# microxcaling — 可组合量化框架

基于 [microsoft/microxcaling](https://github.com/microsoft/microxcaling) 的增量式重建。`mx/` 为只读参考，新代码全部在 `src/`。

## 核心概念

量化方案由**三轴**组合：`format × granularity × transform`

```
QuantScheme = format（数值格式） × granularity（量化粒度） × transform（前后变换）
```

算子级配置由 `OpQuantConfig` 管理，每个 tensor 角色（input / weight / output / grad_*）各自绑定一个 scheme pipeline。

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```python
from src.session import QuantSession
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.analysis.observers import QSNRObserver, MSEObserver
from src.calibration.strategies import PercentileScaleStrategy

# 1. 定义配置
scheme = QuantScheme(
    format=FormatBase.from_str("int8"),
    granularity=GranularitySpec.per_tensor(),
)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

# 2. 初始化 Session（自动量化模型）
session = QuantSession(
    model, cfg,
    calibrator=PercentileScaleStrategy(q=99.0),
    observers=[QSNRObserver(), MSEObserver()],
)

# 3. 校准
with session.calibrate():
    for batch in calib_loader:
        session(batch)

# 4. 层级误差分析
with session.analyze() as ctx:
    for batch in eval_loader:
        session(batch)
report = ctx.report()

# 5. 端到端精度对比
result = session.compare(eval_loader, my_eval_fn)
print(f"fp32: {result['fp32']}, quant: {result['quant']}, delta: {result['delta']}")

# 6. ONNX 导出
session.export_onnx("model.onnx")
```

> 更详细的配置方式、分层配置、Transform、QAT、LSQ 等见 [快速开始详解](docs/refs/quickstart-details.md)。

## 支持的数值格式

| 格式 | 注册名 | 备注 |
|---|---|---|
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
├── formats/         # FormatBase 及各格式实现（int / fp / nf4 / lookup / bf16）
├── scheme/          # QuantScheme、GranularitySpec、OpQuantConfig
├── quantize/        # 核心量化函数
├── ops/             # 量化算子（Linear / Conv / Norm / Activation 等）
├── analysis/        # 误差分析（AnalysisContext / Observer / e2e comparison）
├── mapping/         # quantize_model 一键量化入口
├── calibration/     # Calibration 管线（策略 + Session + LSQ）
├── transform/       # Hadamard / SmoothQuant / PreScale 变换
├── onnx/            # ONNX 导出
├── context/         # QuantizeContext（inline op 截获）
└── session.py       # QuantSession 统一 API
mx/                  # 原始 microsoft/microxcaling（只读）
```

## 测试

```bash
pytest src/tests/ -q
# 1305 passed, 0 xfail
```

所有等价性测试使用 `torch.equal`（bit-exact）。

## 文档索引

| 文档 | 内容 |
|---|---|
| [快速开始详解](docs/refs/quickstart-details.md) | 配置方式、Transform、Calibration、QAT、ONNX 导出、LSQ |
| [架构决策文档](docs/architecture/) | ADR-001 ~ ADR-007（三轴方案、Observer、ONNX、OpQuantConfig 等） |
| [P6 Cost Model 公式](docs/refs/p6-cost-model-formulas.md) | 延迟/显存估算公式 |
| [Format Study 实验](docs/refs/format_study_usage.md) | 量化格式精度对比实验 |
