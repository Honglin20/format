# Phase 计划总览

## 已完成 Phase

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 2 | 三轴扶正：GranularitySpec、TransformBase、FormatBase、消除 MxSpecs | ✅ |
| Phase 3 | 全算子族：Linear/Conv/Norm/Activation/Softmax/Pool/Elemwise/SIMD | ✅ |
| Phase 4 | 层级误差分析：AnalysisContext + QSNR/MSE/Histogram/Distribution Observer | ✅ |
| Phase 5 | ONNX Export：全算子 symbolic() + QDQ + MxQuantize | ✅ |
| Phase 6 | QuantizeContext：torch/F 命名空间 patch + module-stack hooks | ✅ |
| Phase 7 | Unified quantize_model：Module 替换 + forward patching | ✅ |

## 当前 Phase — Phase 8（研究能力扩展）

| 子任务 | 内容 | 状态 |
|--------|------|------|
| P1 | Transform 体系（SmoothQuant/Hadamard/PreScale） | ✅ |
| P2 | Calibration 管线（4 种策略 + pipeline） | ✅ |
| P3 | NF4/LookupFormat | ✅ |
| P4 | 参数化格式注册 | ✅ |
| P5 | LSQ 可学习量化（ADR-006） | ✅ |
| P6 | Coarse Model — 延迟/显存估算 | ✅ |
| P7 | 自动格式搜索 | 未开始 |
| P8 | 融合 Kernel | 未开始 |
| P9 | ONNX custom op ORT 推理 | 未开始 |

## 测试门

| Phase | 测试命令 | 门槛 |
|-------|---------|------|
| Phase 2~3 | `pytest src/tests/ -x` | bit-exact，`torch.equal` |
| Phase 4 | `pytest src/tests/ -x` | 数值稳定（atol 文档明示） |
| Phase 5 | `pytest src/tests/ -x` | 图结构正确 + `onnx.checker` 通过 |
| Phase 6+ | `pytest src/tests/ -q` | 0 xfail，无 regression |

当前全量：`pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q` → 1,416 passed

## 不在当前范围

- ORT / TensorRT runtime 推理适配
- 模型压缩（剪枝/蒸馏）
- RNN 家族算子
- FlashAttention 量化
