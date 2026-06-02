# 测试层级原则 — 最终回归高层 API

## 核心规则

**所有功能开发的最终测试，必须通过 Session / Study 等高层用户接口调用。底层 API 仅限开发调试，不允许留在最终提交的测试代码中。**

## 为什么

1. **用户视角**: 用户通过 `Session(model, config).run()`、`Study(configs, model=model).run()` 使用框架，不会直接调用 `FormatBase.quantize()` 或内部 `_quantize_elemwise_core`。测底层只能验证函数正确性，不能保证用户实际路径可用。
2. **防止 API 断层**: 仅测底层可能导致高层 API 存在未覆盖的路径或接口不一致。例：`OpQuantConfig.from_descriptor()` 仅在测试中使用，与生产路径 `QuantConfig.to_op_config()` 行为不同（2026-05-07 framework review C1），导致测试通过但用户路径缺少 output-side compute 配置。
3. **重构安全**: 内部实现变更时，高层测试不被破坏，真正验证了用户契约。

## 允许的测试入口（高层 API）

| API | 用途 |
|-----|------|
| `Session(model, config).run(...)` | 单配置完整量化流程 |
| `Study(configs, model=model).run(...)` | 多配置对比研究 |
| `QuantConfig(name=..., w_format=..., ...)` | 用户配置入口 |
| `resolve_config(desc)` | 描述符 → OpQuantConfig 解析 |
| `SessionResult` / `StudyReport` | 结果类型（用于断言输出结构） |
| `per_layer_optimal(model, result)` | 逐层最优配置选择 |
| `src.viz` / `src.report` 公共函数 | 可视化与报告（消费上述结果） |

## 禁止用于最终测试的入口（底层/内部 API）

- `_QuantSession` — 内部 session 实现
- `FormatBase.quantize()` / `GranularitySpec.per_tensor()` — 数学层直接调用
- `_quantize_elemwise_core` / `quantize()` — 内部量化函数
- `OpQuantConfig.from_descriptor()` — 仅测试路径，与生产 `QuantConfig.to_op_config()` 行为不一致
- `_Quantizer`、`_Calibrator` 等任何以 `_` 开头的私有模块
- 任何 `from src.formats import ...` 或 `from src.quantize import ...` 的直接调用

## 分阶段规则

| 阶段 | 允许 | 不允许 |
|------|------|--------|
| 开发调试 | 低层 API 调用（理解行为、快速迭代） | — |
| 最终提交（commit） | Session / Study / QuantConfig / resolve_config | `FormatBase.quantize()` 直接调用、`_QuantSession`、任何 `import` 内部模块 |

**流程图**: `写失败测试（高层 API）→ 调试可用底层 → 实现 → 测试回归高层 API → 通过 → commit`

## 边界情况

- **纯数学层验证**（新格式的 bit-exact 量化正确性）：可在专项测试文件中直接测试格式层，但前提是 —— 必须有独立的 Session 级集成测试覆盖同一量化路径。不允许"仅有底层测试、无高层集成测试"。
- **ONNX 导出 / Cost 模型 / Observer** 等横切关注点：遵循各自包的公共 API 约定，禁止直接调用内部实现。
- **v1 遗留测试**（如 `test_format_quantize.py` 中的格式直接调用）：逐步迁移到 Session 集成测试。新增功能不允许添加此类测试。
