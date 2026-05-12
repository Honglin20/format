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

print(result.summary())                  # local QSNR（逐 op）
print(result.summary(qsnr_type="accum"))  # accumulated QSNR（端到端）
print(result.accuracy_table())            # FP32/Quant/Δ 精度对比表
for name, qsnr in result.top_k_qsnr(3, qsnr_type="accum"):
    print(f"  {name}: {qsnr:.1f} dB")

# 换个格式只需改一个字符串
cfg = QuantConfig(name="fp4-mx", w_format="fp4_e2m1", w_granularity="per_block",
                  w_block_size=32, a_format="fp4_e2m1", a_granularity="per_block",
                  a_block_size=32)
result2 = Session(model, cfg).run(calib_data, eval_fn=eval_fn)
print(result2.summary())

# 多 config 对比：Study 统一 DataFrame，含 FP32 基线
from src.session import Study
study = Study([cfg, cfg2], model=model)
report = study.run(calib_data, eval_fn=eval_fn)
report.print_summary()                    # 所有 config 一张表，含 fp32_*/delta_* 列
report.print_summary(qsnr_type="accum")   # 切换为端到端累积 QSNR
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
| [Transform](docs/guides/transforms.md) | none / hadamard / smoothquant / prescale / adaptive · smoothquant_distrib 分布对比 |
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
