# microxcaling — 可组合量化框架

基于 [microsoft/microxcaling](https://github.com/microsoft/microxcaling) 的增量式重建，在 `src/` 中提供高扩展性的张量级量化库。`mx/` 保留为只读参考。

## 核心设计

量化方案由三轴组合描述：

```
QuantScheme = format × granularity × transform
```

| 轴 | 描述 | 示例 |
|---|---|---|
| `format` | 数值格式（int8 / fp8 / nf4 / bf16 等） | `FormatBase.from_str("nf4")` |
| `granularity` | 量化粒度（per_tensor / per_channel / per_block） | `GranularitySpec.per_block(32)` |
| `transform` | 量化前后变换（Hadamard / SmoothQuant 等） | `HadamardTransform()` |

算子级配置由 `OpQuantConfig` 描述，每个 tensor 角色接一个 scheme pipeline（forward + QAT backward 共 12 个角色）。

---

## 安装

```bash
pip install -r requirements.txt
```

---

## 快速开始

以下是一个完整工作流：**定义配置 → 量化模型 → 校准 → 推理 → 误差分析 → 导出**。

> **推荐**：对于大多数使用场景，推荐使用 `QuantSession` 统一 API（见下方），它将以下 6 个步骤封装为单一对象。独立 API 仍可用于需要更细粒度控制的场景。

### QuantSession：统一 API（推荐）

`QuantSession` 将量化、校准、误差分析、端到端精度对比、ONNX 导出封装为单一对象：

```python
from src.session import QuantSession
from src.analysis.observers import QSNRObserver, MSEObserver
from src.calibration.strategies import PercentileScaleStrategy

# 初始化 — 自动量化模型，保留 fp32 副本用于对比
session = QuantSession(
    model, cfg,
    calibrator=PercentileScaleStrategy(q=99.0),
    observers=[QSNRObserver(), MSEObserver()],
)
session.eval()

# 校准（scales 自动写入模型 buffer）
with session.calibrate():
    for batch in calib_loader:
        session(batch)

# 层级误差分析
with session.analyze() as ctx:
    for batch in eval_loader:
        session(batch)
report = ctx.report()

# 端到端精度对比（自动执行 fp32 baseline）
result = session.compare(eval_loader, my_eval_fn)
print(f"fp32: {result['fp32']}, quant: {result['quant']}, delta: {result['delta']}")

# 或手动控制循环（更灵活）
cmp = session.comparator()
with cmp:
    for inputs, labels in eval_loader:
        session.use_fp32()
        fp32_out = session(inputs)
        session.use_quant()
        q_out = session(inputs)
        cmp.record(fp32_out, q_out, labels)
result = cmp.evaluate(my_eval_fn, directions={"acc": "higher"})

# ONNX 导出（自动使用上次推理的输入作为 dummy_input）
session.export_onnx("model.onnx")

# 对比多个量化配置
from src.analysis.e2e import compare_sessions
results = compare_sessions({"int8": s1, "fp4": s2, "nf4": s3}, eval_loader)
```

| 方法 | 返回 | 说明 |
|---|---|---|
| `session(x)` | Tensor | 推理（默认使用量化模型，`use_fp32()` 切换） |
| `session.calibrate()` | `CalibrationSession` | 上下文管理器，退出时自动写入 scales |
| `session.analyze()` | `AnalysisContext` | 上下文管理器，退出后 `ctx.report()` 获取层级报告 |
| `session.compare(dl, fn)` | dict | 自动模式：fp32 vs quant + delta |
| `session.comparator()` | `Comparator` | 手动模式：用户控制循环 + 自定义指标 |
| `session.export_onnx(path)` | — | 导出 ONNX（自动使用上次推理输入） |
| `session.clear_scales()` | list | 清除所有 `_output_scale` buffer |

---

### 1. 定义量化配置（OpQuantConfig）

```python
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig

# ── 方式 A：统一配置（所有算子用同一 scheme）─────────────────
fmt = FormatBase.from_str("int8")
scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
cfg = OpQuantConfig(
    input=(scheme,),
    weight=(scheme,),
    output=(scheme,),
)

# ── 方式 B：分层配置（不同层不同精度）────────────────────────
# 量化模型时用 dict 映射
cfg_dict = {
    "encoder.*":  OpQuantConfig(input=(int8,), weight=(int8,), output=(int8,)),
    "decoder.0":  OpQuantConfig(input=(fp8,), weight=(fp8,), output=(fp8,)),
    "decoder.*":  OpQuantConfig(input=(int4,), weight=(int4,), output=(int4,)),
}

# ── 方式 C：MX block-wise 量化 ─────────────────────────────────
mx_fmt = FormatBase.from_str("fp4_e2m1")
mx_scheme = QuantScheme(mx_fmt, GranularitySpec.per_block(32))
mx_cfg = OpQuantConfig(input=(mx_scheme,), weight=(mx_scheme,), output=(mx_scheme,))

# ── 方式 D：NF4 查找表格式（适合 weight-only）───────────────────
nf4_fmt = FormatBase.from_str("nf4")  # QLoRA 正态优化 4-bit LUT
nf4_scheme = QuantScheme(nf4_fmt, GranularitySpec.per_channel(axis=0))
nf4_cfg = OpQuantConfig(weight=(nf4_scheme,))
```

**OpQuantConfig 完整字段**（每个字段是 `tuple[QuantScheme, ...]` 的 pipeline）：

| 角色 | 字段 | 说明 |
|---|---|---|
| 输入 | `input` | 激活输入量化 |
| 权重 | `weight` | 权重量化（weight-only 只填此项） |
| 偏置 | `bias` | 偏置量化 |
| 输出 | `output` | 输出量化（Linear 最多 2 步：matmul 后 + bias 后） |
| 梯度输出 | `grad_output` | backward 中 grad_output 量化 |
| 梯度输入 | `grad_input` | backward 中 grad_input 量化 |
| 梯度权重 | `grad_weight` | backward 中 grad_weight 量化 |
| 梯度偏置 | `grad_bias` | backward 中 grad_bias 量化 |
| 输入（grad weight） | `input_gw` | backward gemm `grad_w = g^T @ x` 中 x 的 MX 复量化 |
| 梯度输出（grad weight） | `grad_output_gw` | 同上 gemm 中 g 的 MX 复量化 |
| 权重（grad input） | `weight_gi` | backward gemm `grad_x = g @ w` 中 w 的 MX 复量化 |
| 梯度输出（grad input） | `grad_output_gi` | 同上 gemm 中 g 的 MX 复量化 |

> **QAT 自动启用**：任一 `grad_*` 字段非空时，`cfg.is_training = True`，backward 路径自动量化。否则退化为 STE（inference-only）。

### 2. 量化模型（一键入口）

```python
from src.mapping.quantize_model import quantize_model

model = MyModel()  # 标准 PyTorch 模型

# 全模型统一配置
qmodel = quantize_model(model, cfg=cfg)

# 分层配置（支持 glob 匹配）
qmodel = quantize_model(model, cfg={
    "encoder.*":  encoder_cfg,
    "decoder.*":  decoder_cfg,
    "lm_head":    head_cfg,
})

# 按 op 类型配置（inline ops：matmul / add / mul 等）
qmodel = quantize_model(model, cfg=default_cfg, op_cfgs={
    "matmul": matmul_cfg,
    "add":    add_cfg,
})

# 推理
output = qmodel(x)
```

`quantize_model` 自动处理两类算子：

| 类型 | 量化方式 |
|---|---|
| Module（Linear / Conv / Norm / Activation） | 原地替换为 `Quantized*` 类 |
| Inline（`torch.matmul` / `torch.add` / `torch.exp`） | forward patching 注入 `QuantizeContext` |

### 3. 校准（Calibration）

推荐使用 `QuantSession.calibrate()` 上下文管理器（scales 在退出时自动写入）：

```python
with session.calibrate():
    for batch in calib_loader:
        session(batch)
# scales 已自动写入 model buffer
```

或直接使用底层 `CalibrationSession`：

```python
from src.calibration.pipeline import CalibrationSession
from src.calibration.strategies import MaxScaleStrategy

with CalibrationSession(qmodel, MaxScaleStrategy()) as calib:
    for batch in calib_loader:
        qmodel(batch)
# scales 在 __exit__ 时自动写入
```

> `CalibrationPipeline`（旧 DataLoader 驱动接口）保留为 `CalibrationSession` 的向后兼容子类。

### 4. 端到端精度对比

推荐使用 `QuantSession.compare()` 或 `Comparator`：

```python
# 自动模式：fp32 vs quant
result = session.compare(eval_loader, my_eval_fn)

# 手动模式（更灵活）
cmp = session.comparator()
with cmp:
    for inputs, labels in eval_loader:
        session.use_fp32()
        fp32_out = session(inputs)
        session.use_quant()
        q_out = session(inputs)
        cmp.record(fp32_out, q_out, labels)
result = cmp.evaluate(my_eval_fn, directions={"acc": "higher"})

# 对比多个 session
from src.analysis.e2e import compare_sessions
results = compare_sessions({"int8": s1, "fp4": s2}, eval_loader)
```

用户自定义 eval_fn：`(logits, labels) -> dict[str, float]`。框架只负责执行并返回 `{"fp32": {...}, "quant": {...}, "delta": {...}}`。

### 5. 层级误差分析

推荐使用 `QuantSession.analyze()`：

```python
with session.analyze() as ctx:
    for batch in data:
        session(batch)
report = ctx.report()
```

或直接使用底层 `AnalysisContext`：

```python
from src.analysis.context import AnalysisContext
from src.analysis.observers import QSNRObserver, MSEObserver

with AnalysisContext(qmodel, [QSNRObserver(), MSEObserver()]) as ctx:
    for batch in data:
        qmodel(batch)
report = ctx.report()
```

**可用的 Observer**：

| Observer | 指标 | 用途 |
|---|---|---|
| `QSNRObserver` | 量化信噪比（dB） | 识别精度损失最大的层 |
| `MSEObserver` | 均方误差 | 标量误差量级 |
| `HistogramObserver` | fp32 vs quant 值分布直方图 | 可视化量化偏差模式 |
| `DistributionObserver` | 均值、方差、偏度、峰度、稀疏度 | 分布指纹变化 |

所有 observer 自动按 scheme 的 granularity（per_tensor / per_channel / per_block）切片聚合。

### 6. ONNX 导出

推荐使用 `QuantSession.export_onnx()`：

```python
session(x)                    # 推理时自动记录输入
session.export_onnx("m.onnx")  # 使用记录的输入作为 dummy_input

# 或显式传入
session.export_onnx("m.onnx", dummy_input=torch.randn(1, 768))
```

或使用量化模型的便捷方法：

```python
qmodel.export_onnx(torch.randn(1, 768), "model.onnx")
```

导出规则：

| 量化方案 | ONNX 节点 |
|---|---|
| int8/int4 + per_tensor/per_channel | `QuantizeLinear` / `DequantizeLinear`（标准 QDQ） |
| fp8 + per_tensor/per_channel | QDQ（已知限制：JIT tracer 不支持 FP8 sign-magnitude 路径） |
| 任意格式 + per_block（MX block） | `com.microxscaling::MxQuantize`（自定义 domain） |
| NF4 / fp6 / bf16（非标准格式） | `com.microxscaling::MxQuantize`（自定义 domain） |

导出目标：图结构正确 + `onnx.checker` 通过。

---

## Transform（量化变换）

Transform 在量化前后对张量做可逆变换，降低量化误差：

```python
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform

# Hadamard 正交旋转：分散 outlier 能量
scheme = QuantScheme(
    format=FormatBase.from_str("int4"),
    granularity=GranularitySpec.per_channel(axis=-1),
    transform=HadamardTransform(),  # 正向：WHT，反向：WHT（自逆）
)

# SmoothQuant：平滑 activation outlier → weight
scale = SmoothQuantTransform.from_calibration(
    X_act=calib_activations, W=layer.weight.data, alpha=0.5
)
scheme = QuantScheme(
    format=FormatBase.from_str("int8"),
    granularity=GranularitySpec.per_channel(axis=-1),
    transform=scale,  # forward: x/scale, inverse: x*scale
)
```

---

## 支持的数值格式

| 格式名 | 类型 | ebits | mbits | 注册名 | 备注 |
|---|---|---|---|---|---|
| `int8` / `int4` / `int2` | 对称整数 | 0 | 2-8 | `"int8"` `"int4"` `"int2"` | 标准量化 |
| `fp8_e4m3` / `fp8_e5m2` | 浮点 8 位 | 4/5 | 5/4 | `"fp8_e4m3"` `"fp8_e5m2"` | OCP FP8 |
| `fp6_e3m2` / `fp6_e2m3` | 浮点 6 位 | 3/2 | 4/5 | `"fp6_e3m2"` `"fp6_e2m3"` | MX FP6 |
| `fp4_e2m1` | 浮点 4 位 | 2 | 3 | `"fp4_e2m1"` (alias: `"fp4"`) | MX FP4 |
| **`nf4`** | **查找表（非均匀）** | 0 | 4 | **`"nf4"`** | **QLoRA 正态优化 4-bit LUT** |
| `bfloat16` / `float16` | 16 位浮点 | 8/5 | 9/12 | `"bfloat16"` `"float16"` | 硬件快捷路径 |

**添加新格式**（不改核心代码）：
```python
from src.formats.lookup_formats import LookupFormat
from src.formats.registry import register_format

custom = LookupFormat("my_lut", levels=[-1.0, -0.3, 0.0, 0.3, 1.0])
register_format("my_lut", custom)
scheme = QuantScheme(format=custom, granularity=GranularitySpec.per_tensor())
```

---

## Calibration 策略

| 策略 | 类 | 说明 |
|---|---|---|
| Max (absmax) | `MaxScaleStrategy` | `amax = max(|x|)`，当前行为 |
| Percentile | `PercentileScaleStrategy(q=99.0)` | 取第 q 分位数，排除 outlier |
| MSE | `MSEScaleStrategy(n_bins=256)` | 最小化 MSE 的 scale |
| KL | `KLScaleStrategy(n_bins=256)` | TensorRT 风格，最小化 KL divergence |

策略通过 `CalibrationSession` 运行（退出时自动 assign），或通过 `session.calibrate()`。

---

## 训练感知量化（QAT）

```python
cfg_qat = OpQuantConfig(
    input=(scheme,), weight=(scheme,), output=(scheme,),
    grad_output=(scheme,), grad_input=(scheme,), grad_weight=(scheme,),
)
linear = QuantizedLinear(768, 768, cfg=cfg_qat)
# forward + backward 全程量化，与 mx/ bit-exact 等价
```

---

## 项目结构

```
src/
├── formats/         # FormatBase 及各格式实现（int / fp / nf4 / lookup / bf16）
├── scheme/          # QuantScheme、GranularitySpec、OpQuantConfig
├── quantize/        # 核心量化函数（elemwise / mx_quantize）
├── ops/             # 量化算子（Linear / Conv / Norm / Activation 等）
├── analysis/        # 误差分析框架（AnalysisContext / Observer / e2e comparison）
├── mapping/         # quantize_model 模型级量化入口
├── calibration/     # Calibration 管线（ScaleStrategy / CalibrationSession）
├── transform/       # Hadamard / SmoothQuant 变换
├── onnx/            # ONNX 导出
├── context/         # QuantizeContext（inline op 截获）
└── session.py       # QuantSession 统一 API
mx/                  # 原始 microsoft/microxcaling（只读）
```

---

## 示例

`examples/` 目录下有一个**全功能汇总示例**和 6 个专项示例：

```bash
# 全功能汇总（推荐首先运行）— 涵盖所有 11 种格式、3 种粒度、2 种 Transform、
# 4 种校准策略、4 种 Observer、QAT、自定义格式注册、多 session 对比等
PYTHONPATH=. python examples/00_comprehensive.py

# 专项示例
PYTHONPATH=. python examples/01_quickstart.py    # 四种配置方式 + fp32/int8/nf4 对比
PYTHONPATH=. python examples/02_session_workflow.py  # QuantSession 完整工作流
PYTHONPATH=. python examples/03_calibration_analysis.py  # 四种策略对比 + 四种 Observer
PYTHONPATH=. python examples/04_e2e_comparison.py  # Comparator / compare_models / compare_sessions
PYTHONPATH=. python examples/05_onnx_export.py   # ONNX 导出（int8/channel/block + auto-input）
PYTHONPATH=. python examples/06_transforms.py    # Hadamard + SmoothQuant 变换
```

所有示例独立可运行，输出可验证结果。

---

## 与 mx/ 的等价性

`src/` 中所有算子与 `mx/` **bit-exact 等价**（`torch.equal`）。可通过适配器从旧 `MxSpecs` dict 构造 `OpQuantConfig`：

```python
from src.scheme._compat import op_config_from_mx_specs
cfg = op_config_from_mx_specs(mx_specs)
```

---

## 测试

```bash
pytest src/tests/ -q
# 1247 passed, 0 xfail
```

所有等价性测试严格使用 `torch.equal`（bit-exact），不允许 atol/rtol 宽松匹配。
