# microxcaling — 可组合量化框架

基于 [microsoft/microxcaling](https://github.com/microsoft/microxcaling) 的增量式重建。将量化拆为**格式 × 粒度 × 变换**三个正交轴，一个 `QuantConfig` 控制一切。

全算子覆盖（Linear/Conv/Norm/Activation/Softmax/Pool）· MX per-block 位精确等价 · Sparse outlier 隔离 · 4 种校准策略 · 5 种 Transform · LSQ · ONNX 导出 · 误差分析 · 性能估算

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

## 可视化速览

```python
from src.viz import histogram_overlay, classify_layer_type, filter_layers_by_type

# 单层三 role 直方图叠加（fp32 蓝色填充 / quant 红色虚线 / error 灰色）
fig = result.plot.histogram_overlay(layer="module.0.QuantizedLinear")
# → 1×3 面板：input / weight / output 各自的三通道直方图
fig.savefig("histogram_fc1.png", dpi=150)

# Top-5 最敏感层的 (layer, role) 直方图
fig = result.plot.histogram_overlay(top_k=5)

# 查看所有可可视化的层及其算子类型
layer_names = list(result.qsnr_per_layer.keys())
for name in layer_names:
    print(f"{name:40s} → {classify_layer_type(name)}")

# 只看 Linear / Conv 层的 weight QSNR
linear_layers = filter_layers_by_type(layer_names, ["linear"])
print(f"共 {len(linear_layers)} 个 Linear 层")

# 更多内置图表（20+ 种）
result.plot.qsnr_comparison()             # 逐层 QSNR 柱状图
result.plot.error_propagation(role="output")  # 误差传播三行面板
result.plot.per_role_qsnr_bars()          # 每层三 role 分组柱状图
result.plot.per_layer_role_histogram(k=5) # 最差 k 层 × 三 role 直方图网格
```

## 文档导航

一切从 Session 开始。以下按推荐阅读顺序排列：

### Session 阅读路径

1. [Session 概览](docs/guides/session/overview.md) — 量化生命周期、一键模式 vs 分步模式
2. [QuantConfig 配置](docs/guides/session/config.md) — format × granularity × transform × calibration
3. [精度优化方法](docs/guides/session/optimization.md) — Prescale / LSQ / GPTQ / Hadamard / SmoothQuant / Adaptive / per_layer_optimal
4. [结果查看](docs/guides/session/result.md) — summary / accuracy_table / layer_report / top_k_qsnr
5. [可视化](docs/guides/session/plotting.md) — 20+ 种内置图表，含直方图叠加、三 role 对比、误差传播面板、Pareto 前沿等
6. [误差分析](docs/guides/session/analysis.md) — 5 种 Observer / 累积 vs 本地误差传播
7. [Study 多配置对比](docs/guides/session/study.md) — 批量对比 · DataFrame 导出
8. [ONNX 导出](docs/guides/session/export.md) — QDQ + MX 自定义算子
9. [性能估算](docs/guides/session/cost.md) — Roofline 延迟 & 内存

→ [Session 文档索引](docs/guides/session/INDEX.md)

### 完整示例

→ [系统化误差分析示例 — Transformer + AG News + int4](docs/guides/example/transformer_analysis.md)
  ADR-010 四阶段闭环全流程演示：Diagnose → Characterize → Plan → Intervene → Verify

### 参考手册

| 文档 | 内容 |
|------|------|
| [格式选择](docs/guides/formats.md) | int8/fp8/nf4 等格式对比 & 自定义格式注册 |
| [粒度配置](docs/guides/granularity.md) | per_tensor / per_channel / per_block |
| [Sparse Outlier](docs/guides/sparse-outlier.md) | 离群点隔离量化 — 原理、效果与 ratio 选择 |
| [Transform](docs/guides/transforms.md) | none / hadamard / smoothquant / prescale / adaptive |
| [校准策略](docs/guides/calibration.md) | mse / max / percentile / kl |

### 进阶主题

| 文档 | 内容 |
|------|------|
| [自适应 Transform](docs/advanced/adaptive-transform.md) | 逐层自动选择最优变换 |
| [LSQ 深入](docs/advanced/lsq.md) | LayerwiseScaleOptimizer 原理 |
| [自定义格式](docs/advanced/custom-formats.md) | register_format / FormatBase 子类 |
| [底层 API](docs/advanced/low-level-api.md) | quantize_model / OpQuantConfig |
| [MX 位精确等价](docs/advanced/mx-equivalence.md) | 与 microsoft/microxcaling 的等价性验证 |
| [架构决策 (ADR)](docs/architecture/INDEX.md) | 设计意图 & 技术决策 |

### API 参考

| 文档 | 内容 |
|------|------|
| [QuantConfig](docs/reference/quant-config.md) | 完整字段表（30 个字段） |
| [Session](docs/reference/session.md) | 方法签名 & 链式 API |
| [SessionResult](docs/reference/session-result.md) | 属性/方法速查 |
