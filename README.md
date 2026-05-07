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
from src.session import Session, QuantConfig

# 1. 定义配置（一个 dataclass，IDE 自动补全）
cfg = QuantConfig(
    name="int8-pc",
    w_format="int8",
    w_granularity="per_channel",
    calibrator="percentile",
)

# 2. 运行 Session（一行完成 calibrate → analyze → evaluate）
result = Session(model, cfg).run(calib_data, eval_fn=my_eval)
print(f"fp32: {result.fp32_metrics}, quant: {result.quant_metrics}")

# 3. 多配置对比
from src.session import Study
study = Study([cfg_a, cfg_b, cfg_c], model=model)
report = study.run(calib_data, eval_fn=my_eval, outputs="all")
report.save("results/")
```

### 低层精细控制

```python
from src.session import QuantSession
from src.scheme.op_config import OpQuantConfig

qs = QuantSession(model, cfg.to_op_config(), calibrator=PercentileScaleStrategy(q=99.0))
with qs.calibrate():
    for batch in calib_loader:
        qs(batch)
qs.export_onnx("model.onnx")
```

> 详细配置方式、Transform、LSQ、ONNX 导出等见 [快速开始详解](docs/reference/quickstart-details.md)。

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
├── session/         # 驱动层：QuantConfig / Session / Study / quantize_model
├── calibration/     # Calibration 管线（策略 + Session + LSQ）
├── analysis/        # 误差分析（AnalysisContext / Observer / e2e comparison）
├── observer/        # Observer 横切基础设施
├── transform/       # Hadamard / SmoothQuant / PreScale 变换
├── report/          # 输出层：SessionReport / StudyReport（output-driven）
├── cost/            # Coarse Model 性能估算
├── viz/             # 可视化（图表 / 表格）
└── onnx/            # ONNX 导出
mx/                  # 原始 microsoft/microxcaling（只读）
```

## 测试

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
# 1671 passed
```

所有等价性测试使用 `torch.equal`（bit-exact）。

## 文档索引

→ **[docs/INDEX.md](docs/INDEX.md)** — 完整文档导航

| 文档 | 内容 |
|---|---|
| [快速开始详解](docs/reference/quickstart-details.md) | 配置方式、Transform、Calibration、QAT、ONNX 导出、LSQ |
| [架构决策文档](docs/architecture/) | ADR-001 ~ ADR-008（三轴方案、Observer、ONNX、OpQuantConfig、Session 统一入口 等） |
| [P6 Cost Model 公式](docs/reference/p6-cost-model-formulas.md) | 延迟/显存估算公式 |
| [Format Study 实验](docs/reference/format_study_usage.md) | 量化格式精度对比实验 |
| [开发规范](docs/INDEX.md) | 开发准则、规范、流程、Phase 计划 |
