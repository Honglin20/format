# Session 文档索引

Session 是量化框架的核心入口。以下按推荐阅读顺序排列，每章标注前提知识和阅读目标。

## 核心路径（1–6）

| # | 章节 | 前提 | 你将学会 |
|---|------|------|---------|
| 1 | [Session 概览](overview.md) | 无 | 一键量化、分步模式、生命周期、eval_fn 合约 |
| 2 | [QuantConfig 配置](config.md) | 第 1 章 | format × granularity × transform × calibration 四大配置维度 |
| 3 | [精度优化方法](optimization.md) | 第 2 章 | Prescale / LSQ / GPTQ / Hadamard / SmoothQuant / Adaptive / per_layer_optimal |
| 4 | [结果查看](result.md) | 第 1 章 | summary / accuracy_table / layer_report / top_k_qsnr / 属性速查 |
| 5 | [可视化](plotting.md) | 第 4 章 | 多种内置图表、单结果与多配置两种模式 |
| 6 | [误差分析](analysis.md) | 第 4 章 | 5 种 Observer、累积 vs 本地误差传播、分布分类 |

## 扩展路径（7–9）

| # | 章节 | 前提 | 你将学会 |
|---|------|------|---------|
| 7 | [Study 多配置对比](study.md) | 第 1–4 章 | 批量运行配置、聚合对比表、DataFrame 导出 |
| 8 | [ONNX 导出](export.md) | 第 1 章 | 标准 QDQ 和 MX 自定义算子导出 |
| 9 | [性能估算](cost.md) | 第 1 章 | Roofline 延迟 & 内存估算 |

## 参考手册

深入某个配置维度时查阅：

| 文档 | 内容 |
|------|------|
| [格式选择](../formats.md) | int8/fp8/nf4 等格式对比 & 自定义格式注册 |
| [粒度配置](../granularity.md) | per_tensor / per_channel / per_block |
| [Transform](../transforms.md) | none / hadamard / smoothquant / prescale / adaptive 详细参数 |
| [校准策略](../calibration.md) | mse / max / percentile / kl 对比 |

## 进阶主题

| 文档 | 内容 |
|------|------|
| [自适应 Transform](../../advanced/adaptive-transform.md) | `transform="adaptive"` 内部机制 |
| [LSQ 深入](../../advanced/lsq.md) | LayerwiseScaleOptimizer 原理 |
| [自定义格式](../../advanced/custom-formats.md) | register_format / FormatBase 子类 |
| [底层 API](../../advanced/low-level-api.md) | quantize_model / OpQuantConfig / QuantScheme |
| [MX 等价性](../../advanced/mx-equivalence.md) | 与 microsoft/microxcaling 位精确等价验证 |

## API 参考

| 文档 | 内容 |
|------|------|
| [QuantConfig](../../reference/quant-config.md) | 完整字段表 |
| [Session](../../reference/session.md) | 方法签名 & 链式 API |
| [SessionResult](../../reference/session-result.md) | 属性/方法速查 |
