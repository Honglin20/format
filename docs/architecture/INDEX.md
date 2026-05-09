# 架构决策文档（ADR）

| 编号 | 文件 | 内容 |
|------|------|------|
| 001 | [001-three-axis-quant-scheme.md](001-three-axis-quant-scheme.md) | QuantScheme 三轴设计、Format/Granularity/Transform 接口规范 |
| 002 | [002-observer-analysis.md](002-observer-analysis.md) | Observer 模式 + SliceAwareObserver + iter_slices |
| 003 | [003-onnx-export.md](003-onnx-export.md) | ONNX export 策略（混合 QDQ + 自定义 domain） |
| 004 | [004-mxspecs-migration.md](004-mxspecs-migration.md) | MxSpecs → QuantScheme 渐进式迁移计划 |
| 005 | [005-op-quant-config.md](005-op-quant-config.md) | OpQuantConfig：算子级 scheme pipeline 容器 |
| 006 | [006-p5-learnable-pre-scale.md](006-p5-learnable-pre-scale.md) | LSQ：PreScaleTransform + LayerwiseScaleOptimizer |
| 007 | [007-p6-cost-model.md](007-p6-cost-model.md) | Coarse Model 架构设计（包结构、Session 集成） |
| 008 | [008-session-refactor.md](008-session-refactor.md) | Session 统一入口 + Output-Driven 架构（QuantConfig / Session / Study） ✅ 已实施 |
| 009 | [009-quantize-nonlinear.md](009-quantize-nonlinear.md) | `quantize_nonlinear`：非线性算子统一量化策略（entry operand 对齐 matmul, 内部不变） |
