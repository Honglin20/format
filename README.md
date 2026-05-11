# microxcaling — 可组合量化框架

基于 [microsoft/microxcaling](https://github.com/microsoft/microxcaling) 的增量式重建。将量化拆为**格式 × 粒度 × 变换**三个正交轴，一个 `QuantConfig` 控制一切。

全算子覆盖（Linear/Conv/Norm/Activation/Softmax/Pool）· MX per-block 位精确等价 · 4 种校准策略 · 5 种 Transform · LSQ · ONNX 导出 · 误差分析 · 性能估算

## 安装

```bash
pip install -r requirements.txt
```

## 30 秒快速体验

```python
import torch, torch.nn as nn
from src.session import Session, QuantConfig

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
calib_data = [torch.randn(32, 128) for _ in range(10)]

def eval_fn(m, data):
    return {"loss": sum(m(batch).sum() for batch in data).item()}

result = Session(model, QuantConfig(name="int8", w_format="int8")).run(
    calib_data, eval_fn=eval_fn)

print(result.summary())            # 一行摘要
print(result.accuracy_table())     # 精度对比表
for name, qsnr in result.top_k_qsnr(3):
    print(f"  {name}: {qsnr:.1f} dB")

# 换个格式只需改一个字符串
cfg = QuantConfig(name="fp4-mx", w_format="fp4_e2m1", w_granularity="per_block",
                  w_block_size=32, a_format="fp4_e2m1", a_granularity="per_block",
                  a_block_size=32)
result2 = Session(model, cfg).run(calib_data, eval_fn=eval_fn)
print(result2.summary())
```

## 按 Role 查看 QSNR

`result.qsnr_per_layer` 默认提取 **output** role 的 QSNR——它能跨层横向对比并追踪累积误差传播。input QSNR 在第一层之后是「已量化数据再量化」（虚高），weight QSNR 是静态的与深度无关。

如需按其他 role 查看：

```python
from src.session._session import _extract_qsnr_mse

# 只看 output（默认）
qsnr_out, _ = _extract_qsnr_mse(result.observers_data, role="output")

# 只看 weight
qsnr_w, _ = _extract_qsnr_mse(result.observers_data, role="weight")

# 只看 input（仅第一层有意义）
qsnr_in, _ = _extract_qsnr_mse(result.observers_data, role="input")
```

## 误差传播分析：累积 vs 本地

量化误差有两套独立的测量路径，结合使用可分辨某层误差是**自己量化引入的（Source）**还是**前层传播过来的（Propagated）**：

| 测量路径 | 数据来源 | 度量含义 | 覆盖范围 |
|---------|---------|---------|---------|
| **累积误差**（Hook） | `true_error=True`，逐层对比 fp32 vs quant output | 从第一层累加到此层的总误差 | `_MODULE_MAPPING` 中的模块（Linear/Conv/Norm/...） |
| **本地误差**（Observer） | QSNRObserver，在量化算子内部 `_emit` 事件 | 仅此层本次量化引入的误差 | 同上 + patched inline ops（`torch.matmul` 等） |

Observer 覆盖范围 > Hook（`hook ⊂ observer`），因此可发现未被 hook 覆盖的自定义模块（如 attention 中的 `torch.matmul`）。

```python
from src.session import Session, QuantConfig

# true_error=True 同时启用 hook（累积）和 observer（本地）两条路径
result = Session(model, cfg).run(
    calib_data, outputs=["qsnr", "mse"], true_error=True,
)

# 累积 QSNR（hook，逐层对比 fp32 参考输出）
print(result.qsnr_per_layer)   # {"0": 33.9, "2": 34.0, "3": 31.5}

# 本地 QSNR（observer，量化算子内部测量）
from src.session._session import _extract_qsnr_mse
local_qsnr, _ = _extract_qsnr_mse(result.observers_data, role="output")
print(local_qsnr)              # {"0": 55.4, "1.matmul": 55.4, "2": 314.0, "3": 57.0}
```

### 诊断表与可视化

```python
from src.report._study_report import StudyReport

report = StudyReport({"my_config": [result]})

# 终端诊断表：每层 Accum QSNR / Local QSNR / Delta / Headroom / Diagnosis
print(report.tables.error_source_analysis(role="output"))

# 三行面板图：分组柱状图 + δ-QSNR + Headroom 诊断
report.plot.error_propagation(role="output")

# 散点图：Accumulated vs Local QSNR，y=x 对角线区分 Source vs Propagated
report.plot.accumulated_vs_local(role="output")

# 一键导出所有图表和 CSV
report.save("output/")
```

**诊断规则**（Headroom = Local − Accumulated）：
- ≤ 3 dB → **Source**：该层是主要误差来源
- 3–10 dB → **Mixed**：部分自产，部分传播
- \> 10 dB → **Propagated**：误差主要来自前层累积

### 演示脚本

```bash
python scripts/test_error_propagation.py          # 4 层简单模型，含 observer-only CustomMatMul
python scripts/test_transformer_error_propagation.py  # 4 层 Transformer Encoder
```

## 文档导航

### 使用指南

| 文档 | 内容 |
|------|------|
| [Session 概览](docs/guides/session/overview.md) | 量化生命周期：quantize → calibrate → analyze → evaluate → cost |
| [SessionResult & 结果查看](docs/guides/session/result.md) | summary / accuracy_table / layer_report / QSNR / MSE |
| [绘图 & 可视化](docs/guides/session/plotting.md) | 12 种内置图表：QSNR/MSE/outlier/block-QSNR/Pareto/correlation/cost/role-distribution/error-propagation |
| [误差分析](docs/guides/session/analysis.md) | 5 种 Observer、分布分析、LayerSensitivity、累积 vs 本地误差传播 |
| [格式选择](docs/guides/formats.md) | int8/fp8/nf4 等格式对比 & 自定义格式注册 |
| [粒度配置](docs/guides/granularity.md) | per_tensor / per_channel / per_block 选择 |
| [Transform](docs/guides/transforms.md) | none / hadamard / smoothquant / prescale / adaptive |
| [校准策略](docs/guides/calibration.md) | mse / max / percentile / kl 对比 |
| [多配置对比 (Study)](docs/guides/session/study.md) | Study → StudyReport → 对比表 + 图表导出 |
| [ONNX 导出](docs/guides/onnx-export.md) | 标准 QDQ + MX 自定义算子 |

### API 参考

| 文档 | 内容 |
|------|------|
| [QuantConfig](docs/reference/quant-config.md) | 完整字段表（含 storage_format） |
| [Session](docs/reference/session.md) | 方法签名 & 链式 API |
| [SessionResult](docs/reference/session-result.md) | 属性/方法速查 |

### 进阶主题

| 文档 | 内容 |
|------|------|
| [自适应 Transform](docs/advanced/adaptive-transform.md) | 逐层自动选择最优变换 |
| [LSQ 可学习量化](docs/advanced/lsq.md) | LayerwiseScaleOptimizer |
| [自定义格式](docs/advanced/custom-formats.md) | register_format / 自动解析 fpN_eXmY |
| [底层 API](docs/advanced/low-level-api.md) | quantize_model / OpQuantConfig |
| [MX 位精确等价](docs/advanced/mx-equivalence.md) | 与 microsoft/microxcaling 的等价性验证 |
| [性能估算](docs/advanced/cost-model.md) | Roofline 延迟 & 内存估算 |
| [架构决策 (ADR)](docs/architecture/INDEX.md) | 设计意图 & 技术决策 |
