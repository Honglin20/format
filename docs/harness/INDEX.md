# docs/harness/ — AgentHarness 工作流集成文档

> **说明**：本文档中 "harness" 均指 [AgentHarness](../../AgentHarness/) 项目（Agent 编排框架），
> 不是 bitx 自身的模块。bitx 通过 adapter 模式与 AgentHarness 协作。

AgentHarness 提供 agent 编排 + 前端可视化，
bitx 提供量化分析能力，两者通过 adapter 模式协作。

---

## 文件索引

| 文件 | 内容 |
|------|------|
| [adapter-guide.md](adapter-guide.md) | adapter 合约定义 + 生成规范 |
| [workflow-patterns.md](workflow-patterns.md) | 工作流设计模式 + 结构化输出 |
| [result-types.md](result-types.md) | 自定义 Pydantic result_type 定义 |
| [charts.md](charts.md) | render_chart 集成 + 图表清单 |
| [api-block-error-analysis.md](api-block-error-analysis.md) | Agent 6: block error analysis API |
| [api-cross-config-ranking.md](api-cross-config-ranking.md) | Agent 4: cross-config layer ranking API |
| [api-transform-effect.md](api-transform-effect.md) | Agent 3/7: transform effect analysis API |
| [api-intervention-eval.md](api-intervention-eval.md) | Agent 7: intervention evaluation API |
| [example19-spec.md](example19-spec.md) | Example 19 完整设计文档 |

---

## 快速开始

1. bitx 分析脚本：`python -m src.api.mxint_error_analysis --adapter /path/to/_adapter.py`
2. harness 工作流 (Example 18)：`examples/18_bitx_mxint_analysis.py`
3. 工作流定义 (Example 18)：AgentHarness `workflows/mxint-analysis/`
4. 全链路诊断 (Example 19)：`examples/19_mxint_diagnostic.py`
5. 诊断工作流定义：AgentHarness `workflows/mxint-diagnostic/`
