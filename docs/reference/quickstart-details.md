# 快速开始详解

本文档是 README 快速开始的详细展开，涵盖所有配置方式、高级功能和完整 API 参考。

**API 层级**：`QuantConfig`（用户配置）→ `Session`（执行单元）→ `Study`（多配置对比） 为高层入口；`OpQuantConfig` + `QuantSession` 为低层精细控制。

---

## 0. Session 工作流（高层入口）

`QuantConfig` 是用户唯一需要关心的配置入口——一个 dataclass，IDE 自动补全所有字段。它内部翻译为 `OpQuantConfig`，用户无需直接接触 `QuantScheme` / `GranularitySpec` / `FormatBase`。

### 0.1 定义配置（QuantConfig）

```python
from src.session import QuantConfig

# 最简配置：全部默认（INT8 per-tensor, no transform）
cfg = QuantConfig()

# 典型配置：INT8 per-channel + percentile 校准
cfg = QuantConfig(
    name="int8-pc",
    w_format="int8",
    w_granularity="per_channel",
    calibrator="percentile",
)

# MX 格式：block-wise 量化（scale 动态计算，无需校准）
cfg = QuantConfig(
    w_format="fp4_e2m1", w_granularity="per_block", w_block_size=32,
    a_format="fp4_e2m1", a_granularity="per_block", a_block_size=32,
)

# SmoothQuant：激活平滑 + 权重量化
cfg = QuantConfig(
    w_format="int8", w_granularity="per_channel",
    transform="smoothquant", sq_alpha=0.5,
)

# PreScale + LSQ：可学习前置 scale
cfg = QuantConfig(
    transform="prescale", prescale_init="amax",
    lsq_steps=100, lsq_lr=1e-3,
)

# Weight-only：仅量化权重
cfg = QuantConfig(weight_only=True, w_format="int4", w_granularity="per_channel")

# Element-wise 存储量化 + 计算量化（两级模型）
cfg = QuantConfig(
    w_format="int8", w_granularity="per_channel",
    storage_bits=16, storage_kind="bfloat",  # 所有张量先过 bfloat16
)

# 只量化 MatMul 算子，非线性（norm/activation/pool）保持 fp32
cfg = QuantConfig(
    w_format="fp4_e2m1", w_granularity="per_block", w_block_size=32,
    a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
    quantize_nonlinear=False,
)
```

| QuantConfig 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `name` | `str` | `""` | 配置名（Study 对比时显示） |
| `w_format` | `str` | `"int8"` | 权重数值格式 |
| `w_granularity` | `str` | `"per_tensor"` | 权重量化粒度：per_tensor / per_channel / per_block |
| `w_block_size` | `int\|None` | `None` | per_block 时的 block 大小 |
| `w_axis` | `int` | `-1` | 权重量化轴 |
| `a_format` | `str\|None` | `None` | 激活格式（None = 同权重） |
| `a_granularity` | `str` | `"per_tensor"` | 激活量化粒度 |
| `a_block_size` | `int\|None` | `None` | 激活 per_block 时的 block 大小 |
| `a_axis` | `int` | `-1` | 激活量化轴 |
| `transform` | `str` | `"none"` | none / hadamard / smoothquant / prescale |
| `sq_alpha` | `float` | `0.5` | SmoothQuant 平滑强度 |
| `prescale_init` | `str` | `"ones"` | PreScale 初始化：ones / amax / pot_amax |
| `prescale_pot` | `bool` | `False` | PreScale 是否投影到 2 的幂 |
| `prescale_granularity` | `str\|None` | `None` | None = 跟随 a_granularity |
| `lsq_steps` | `int` | `0` | LSQ 优化步数（>0 需要 transform="prescale"） |
| `lsq_lr` | `float` | `1e-3` | LSQ 学习率 |
| `scale_storage` | `str` | `"fp32"` | Scale 存储格式：fp32 / pot |
| `calibrator` | `str` | `"mse"` | 校准策略：mse / max / percentile / kl |
| `storage_bits` | `int` | `0` | Element-wise 存储位宽（0 = 禁用） |
| `storage_kind` | `str` | `"bfloat"` | 存储类型：bfloat / fp |
| `weight_only` | `bool` | `False` | 仅量化权重，激活保持 FP32 |
| `quantize_nonlinear` | `bool` | `True` | False = norm / activation / pool 保持 fp32 |

### 0.2 Session — 执行单元

一个 `Session` = 一个 `QuantConfig` → 一个 `SessionResult`。支持两种使用方式：

**全自动 `run()`**：一行完成 quantize → calibrate → analyze → evaluate → cost。

```python
from src.session import Session

result = Session(model, cfg).run(calib_data, eval_data=eval_data, eval_fn=eval_fn)
```

**分步链式 API**：每步可干预，`session.qmodel` 在 `.quantize()` 后即可直接推理。

```python
session = Session(model, cfg)

session.quantize(calib_data=calib_data)   # 构建量化模型
# session.qmodel 现在可用：output = session.qmodel(x)

session.calibrate(calib_data)             # MX per_block 自动跳过
session.analyze(calib_data, outputs="default")
session.evaluate(eval_data, eval_fn)
session.cost()

result = session.result

# 链式一行：
result = (Session(model, cfg)
    .quantize(calib_data=calib_data)
    .calibrate(calib_data)
    .analyze(calib_data)
    .evaluate(eval_data, eval_fn)
    .cost()
    .result)
```

**Session 方法与属性**：

| 方法 / 属性 | 返回 | 说明 |
|---|---|---|
| `.quantize(*, calib_data=None)` | `self` | 构建量化模型。smoothquant/prescale 需传 calib_data |
| `.calibrate(calib_data, *, eval_fn=None)` | `self` | 校准 scale。MX per_block 自动跳过（no-op） |
| `.analyze(calib_data, *, outputs, eval_fn)` | `self` | 误差分析。outputs: "default" / "all" / ["qsnr","mse",...] |
| `.evaluate(eval_data, eval_fn)` | `self` | 精度评估（fp32 vs quant + delta） |
| `.cost()` | `self` | 延迟/显存估算 |
| `.result` | `SessionResult` | 构建返回结果 |
| `.run(calib_data, *, eval_data, eval_fn, outputs)` | `SessionResult` | 全自动快捷方式（向后兼容） |
| `.qmodel` | `nn.Module` | 量化模型（`.quantize()` 后可访问） |
| `.fp32_model` | `nn.Module` | FP32 参考模型 |
| `.use_fp32()` / `.use_quant()` | `self` | 切换推理模式 |
| `.mode` | `str` | 当前模式："fp32" / "quant" |
| `session(x)` | `Tensor` | 委托推理（`__call__`，默认量化模式） |

### 0.3 SessionResult — 结果访问

```python
result = session.result

# 一行摘要
print(result.summary())
# Config: int8-pc | loss: fp32=0.1234 quant=0.1456 | Δloss=+0.0222 | avg QSNR=34.2 dB

# 精度对比表
print(result.accuracy_table())
# Metric    FP32      Quant     Δ
# --------------------------------
# loss      0.1234    0.1456    +0.0222
# acc       0.9500    0.9300    -0.0200

# QSNR 最差的 K 层（定位精度瓶颈）
for name, qsnr in result.top_k_qsnr(5):
    print(f"  {name}: {qsnr:.1f} dB")

# 逐层 DataFrame
df = result.layer_report()
print(df.sort_values("qsnr_db").head())
```

| 访问方法 | 返回 | 说明 |
|---|---|---|
| `.summary()` | `str` | 一行可读摘要（配置名 + 指标 + 平均 QSNR） |
| `.accuracy_table()` | `str` | 格式化精度对比表（FP32 / Quant / Δ） |
| `.top_k_qsnr(k=10)` | `List[Tuple[str, float]]` | QSNR 最差的 k 层（升序，最差优先） |
| `.layer_report()` | `DataFrame | None` | 逐层 DataFrame（pandas 不可用时返回 None） |

### 0.4 Study — 多配置对比

```python
from src.session import Study

study = Study([cfg_int8, cfg_int4, cfg_fp4], model=model)
report = study.run(calib_data, eval_fn=eval_fn, outputs="all")
report.save("results/")
```

---

## 1. 定义量化配置（OpQuantConfig — 低层）

以下为低层 API，直接操作 `OpQuantConfig` + `QuantScheme`。日常使用推荐上面的 `QuantConfig`。

### 方式 A：统一配置

所有算子使用同一 scheme：

```python
fmt = FormatBase.from_str("int8")
scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
```

### 方式 B：分层配置

不同层不同精度，通过 glob 匹配：

```python
cfg_dict = {
    "encoder.*":  OpQuantConfig(input=int8_scheme, weight=int8_scheme, output=int8_scheme),
    "decoder.0":  OpQuantConfig(input=fp8_scheme, weight=fp8_scheme, output=fp8_scheme),
    "decoder.*":  OpQuantConfig(input=int4_scheme, weight=int4_scheme, output=int4_scheme),
}
```

### 方式 C：MX block-wise 量化

```python
mx_fmt = FormatBase.from_str("fp4_e2m1")
mx_scheme = QuantScheme(mx_fmt, GranularitySpec.per_block(32))
mx_cfg = OpQuantConfig(input=mx_scheme, weight=mx_scheme, output=mx_scheme)
```

### 方式 D：NF4 查找表格式

```python
nf4_fmt = FormatBase.from_str("nf4")  # QLoRA 正态优化 4-bit LUT
nf4_scheme = QuantScheme(nf4_fmt, GranularitySpec.per_channel(axis=0))
nf4_cfg = OpQuantConfig(weight=nf4_scheme)
```

### OpQuantConfig 完整字段

两阶段模型（storage + compute），每个字段是 `QuantScheme | None`（None = 不量化）：

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

> QAT 自动启用：任一 `grad_*` 字段非空时，`cfg.is_training = True`，backward 路径自动量化。

---

## 2. 量化模型（一键入口）

```python
from src.session import quantize_model

# 全模型统一配置
qmodel = quantize_model(model, cfg=cfg)

# 分层配置（支持 glob 匹配）
qmodel = quantize_model(model, cfg={"encoder.*": encoder_cfg, "decoder.*": decoder_cfg})

# 按 op 类型配置（inline ops：matmul / add / mul 等）
qmodel = quantize_model(model, cfg=default_cfg, op_cfgs={"matmul": matmul_cfg, "add": add_cfg})
```

`quantize_model` 自动处理两类算子：

| 类型 | 量化方式 |
|---|---|
| Module（Linear / Conv / Norm / Activation） | 原地替换为 `Quantized*` 类 |
| Inline（`torch.matmul` / `torch.add` / `torch.exp`） | forward patching 注入 `QuantizeContext` |

---

## 3. QuantSession API 参考（低层）

`QuantSession` 是低层 API，需要用户直接传入 `OpQuantConfig`。高层推荐使用上面的 `Session`（接受 `QuantConfig`，内部自动翻译）。

| 方法 | 返回 | 说明 |
|---|---|---|
| `session(x)` | Tensor | 推理（默认量化模型，`use_fp32()` 切换） |
| `session.calibrate()` | `CalibrationSession` | 上下文管理器，退出时自动写入 scales |
| `session.analyze()` | `AnalysisContext` | 上下文管理器，退出后 `ctx.report()` 获取层级报告 |
| `session.compare(dl, fn)` | dict | 自动模式：fp32 vs quant + delta |
| `session.comparator()` | `Comparator` | 手动模式：用户控制循环 + 自定义指标 |
| `session.export_onnx(path)` | — | 导出 ONNX（自动使用上次推理输入） |
| `session.clear_scales()` | list | 清除所有 `_output_scale` buffer |
| `session.initialize_pre_scales(data, init, pot)` | int | 初始化 `_pre_scale` buffer |
| `session.optimize_scales(opt, data)` | dict | 逐层 LSQ 梯度优化 pre_scale |

### 手动 Comparator 用法

```python
cmp = session.comparator()
with cmp:
    for inputs, labels in eval_loader:
        session.use_fp32()
        fp32_out = session(inputs)
        session.use_quant()
        q_out = session(inputs)
        cmp.record(fp32_out, q_out, labels)
result = cmp.evaluate(my_eval_fn, directions={"acc": "higher"})
```

用户自定义 eval_fn：`(logits, labels) -> dict[str, float]`。返回 `{"fp32": {...}, "quant": {...}, "delta": {...}}`。

### 多 Session 对比

```python
from src.analysis.e2e import compare_sessions
results = compare_sessions({"int8": s1, "fp4": s2, "nf4": s3}, eval_loader)
```

---

## 4. 校准（Calibration）

### 推荐方式：QuantSession.calibrate()

```python
with session.calibrate():
    for batch in calib_loader:
        session(batch)
# scales 已自动写入 model buffer
```

### 底层方式：CalibrationSession

```python
from src.calibration.pipeline import CalibrationSession
from src.calibration.strategies import MaxScaleStrategy

with CalibrationSession(qmodel, MaxScaleStrategy()) as calib:
    for batch in calib_loader:
        qmodel(batch)
```

### 可用策略

| 策略 | 类 | 说明 |
|---|---|---|
| Max (absmax) | `MaxScaleStrategy` | `amax = max(\|x\|)` |
| Percentile | `PercentileScaleStrategy(q=99.0)` | 取第 q 分位数，排除 outlier |
| MSE | `MSEScaleStrategy(n_bins=256)` | 最小化 MSE 的 scale |
| KL | `KLScaleStrategy(n_bins=256)` | TensorRT 风格，最小化 KL divergence |

---

## 5. 层级误差分析

### 推荐方式：QuantSession.analyze()

```python
with session.analyze() as ctx:
    for batch in data:
        session(batch)
report = ctx.report()
```

### 底层方式：AnalysisContext

```python
from src.analysis.context import AnalysisContext
from src.analysis.observers import QSNRObserver, MSEObserver

with AnalysisContext(qmodel, [QSNRObserver(), MSEObserver()]) as ctx:
    for batch in data:
        qmodel(batch)
report = ctx.report()
```

### 可用 Observer

| Observer | 指标 | 用途 |
|---|---|---|
| `QSNRObserver` | 量化信噪比（dB） | 识别精度损失最大的层 |
| `MSEObserver` | 均方误差 | 标量误差量级 |
| `HistogramObserver` | fp32 vs quant 值分布直方图 | 可视化量化偏差模式 |
| `DistributionObserver` | 均值、方差、偏度、峰度、稀疏度 | 分布指纹变化 |

所有 observer 自动按 scheme 的 granularity 切片聚合。

---

## 6. ONNX 导出

### 推荐方式：QuantSession.export_onnx()

```python
session(x)                    # 推理时自动记录输入
session.export_onnx("m.onnx")  # 使用记录的输入作为 dummy_input
session.export_onnx("m.onnx", dummy_input=torch.randn(1, 768))  # 显式传入
```

### 量化模型便捷方法

```python
qmodel.export_onnx(torch.randn(1, 768), "model.onnx")
```

### 导出规则

| 量化方案 | ONNX 节点 |
|---|---|
| int8/int4 + per_tensor/per_channel | `QuantizeLinear` / `DequantizeLinear`（标准 QDQ） |
| fp8 + per_tensor/per_channel | QDQ（已知限制：JIT tracer 不支持 FP8 sign-magnitude 路径） |
| 任意格式 + per_block（MX block） | `com.microxscaling::MxQuantize`（自定义 domain） |
| NF4 / fp6 / bf16（非标准格式） | `com.microxscaling::MxQuantize`（自定义 domain） |

导出目标：图结构正确 + `onnx.checker` 通过。

---

## 7. Transform（量化变换）

Transform 在量化前后对张量做可逆变换，降低量化误差：

```python
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform
from src.transform.pre_scale import PreScaleTransform

# Hadamard 正交旋转：分散 outlier 能量
scheme = QuantScheme(
    format=FormatBase.from_str("int4"),
    granularity=GranularitySpec.per_channel(axis=-1),
    transform=HadamardTransform(),
)

# SmoothQuant：平滑 activation outlier → weight
scale = SmoothQuantTransform.from_calibration(
    X_act=calib_activations, W=layer.weight.data, alpha=0.5
)
scheme = QuantScheme(
    format=FormatBase.from_str("int8"),
    granularity=GranularitySpec.per_channel(axis=-1),
    transform=scale,
)

# PreScale：可学习前置 scale
scheme = QuantScheme(
    format=FormatBase.from_str("int8"),
    granularity=GranularitySpec.per_tensor(),
    transform=PreScaleTransform(scale=torch.ones(1), pot=True),
)
```

| Transform | 方法 | 说明 |
|---|---|---|
| `PreScaleTransform(scale, pot=False)` | `x * scale` / `x_q / scale` | 可学习前置 scale |
| `HadamardTransform()` | WHT(x) / WHT(x_q) | 正交旋转分散 outlier 能量 |
| `SmoothQuantTransform(scale)` | `x / scale` / `x_q * scale` | 平滑 activation outlier → weight |

---

## 8. LSQ 逐层 Scale 优化

```python
from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer

# 阶段 1：初始化 pre_scale
session.initialize_pre_scales(calib_data, init="ones", pot=True)

# 阶段 2：逐层 LSQ 优化
optimizer = LayerwiseScaleOptimizer(
    num_steps=100, num_batches=8,
    optimizer="adam", lr=1e-3, pot=True,
)
result = session.optimize_scales(optimizer, calib_data)
# → {module_name: optimized_pre_scale, ...}
```

---

## 9. 训练感知量化（QAT）

```python
cfg_qat = OpQuantConfig(
    input=scheme, weight=scheme, output=scheme,
    grad_output=scheme, grad_input=scheme, grad_weight=scheme,
)
linear = QuantizedLinear(768, 768, cfg=cfg_qat)
# forward + backward 全程量化，与 mx/ bit-exact 等价
```

---

## 10. 自定义格式

```python
from src.formats.lookup_formats import LookupFormat
from src.formats.registry import register_format

custom = LookupFormat("my_lut", levels=[-1.0, -0.3, 0.0, 0.3, 1.0])
register_format("my_lut", custom)
scheme = QuantScheme(format=custom, granularity=GranularitySpec.per_tensor())
```

---

## 11. mx/ 等价性适配

测试辅助函数，将 mx 的 specs dict 转为 `OpQuantConfig`：

```python
from src.tests._compat import op_config_from_mx_specs
cfg = op_config_from_mx_specs(mx_specs)
```

---

## 12. 示例

```bash
# 全功能汇总（推荐首先运行）
PYTHONPATH=. python examples/00_comprehensive.py

# 专项示例
PYTHONPATH=. python examples/01_quickstart.py          # 四种配置方式 + 格式对比
PYTHONPATH=. python examples/02_session_workflow.py    # QuantSession 完整工作流
PYTHONPATH=. python examples/03_calibration_analysis.py  # 四种策略 + 四种 Observer
PYTHONPATH=. python examples/04_e2e_comparison.py      # Comparator / compare_sessions
PYTHONPATH=. python examples/05_onnx_export.py         # ONNX 导出
PYTHONPATH=. python examples/06_transforms.py          # Hadamard + SmoothQuant
PYTHONPATH=. python examples/07_pre_scale.py           # PreScale + LSQ + PoT
```

> **Format Study**：`examples/format_study_random.py` 是系统化的量化格式精度研究实验。详见 [format_study_usage.md](format_study_usage.md)。
