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
| [analysis-methodology.md](analysis-methodology.md) | bitx 分析原子能力清单 + Harness 集成状态 + 方法论 |
| [charts.md](charts.md) | render_chart 集成 + 图表清单 |
| [api-block-error-analysis.md](api-block-error-analysis.md) | Agent 6: block error analysis API |
| [api-cross-config-ranking.md](api-cross-config-ranking.md) | Agent 4: cross-config layer ranking API |
| [api-transform-effect.md](api-transform-effect.md) | Agent 3/7: transform effect analysis API |
| [api-intervention-eval.md](api-intervention-eval.md) | Agent 7: intervention evaluation API |
| [example19-spec.md](example19-spec.md) | Example 19 完整设计文档 |

---

## 工作流清单（Example 18–20）

| Example | 工作流名称 | Agent 数 | 用途 | 入口文件 |
|---------|-----------|---------|------|---------|
| 18 | `mxint-analysis` | 3 | 基础 MXInt 量化分析 | `AgentHarness/examples/18_bitx_mxint_analysis.py` |
| 19 | `mxint-diagnostic` | 8 | MXInt 全链路精度诊断 | `AgentHarness/examples/19_mxint_diagnostic.py` |
| 20 | `precision-diagnostic` | 6 | 通用格式精度诊断 + 内联图表 | `AgentHarness/examples/20_precision_diagnostic.py` |

### Example 18: mxint-analysis
3-agent + 1-judge 流水线：`analyzer → configurator → _judge_configurator → runner`
- 适配任意 PyTorch 项目，运行 MXInt 量化，输出精度结果
- configurator 带 eval 校验：确保 adapter 与原项目逻辑完全等价
- 工作流定义：`AgentHarness/workflows/mxint-analysis/`

### Example 19: mxint-diagnostic
8-agent 流水线：`adapter → study_runner → gap_analyzer + layer_attribution → [distribution_profiler + block_analyst + intervention_evaluator] → synthesis`
- 从粗到细全链路 MXInt 精度诊断
- 工作流定义：`AgentHarness/workflows/mxint-diagnostic/`

### Example 20: precision-diagnostic
6-agent 流水线：`adapter → quant_study → coarse_analyzer → [deep_dive_analyst + intervention_explorer] → summary_painter`
- 格式无关的精度诊断，用户指定目标格式
- 合并了 example 19 的 gap+layer attribution、distribution+block 分析
- summary_painter 通过 `render_chart()` 渲染 10 张内联图表
- adapter/quant_study/intervention_explorer 带 `ask_user` 支持交互
- 工作流定义：`AgentHarness/workflows/precision-diagnostic/`

---

## Agent 工具集

Harness 通过 MCP filesystem server 自动注册文件操作工具。Agent md 的 `tools` 字段是白名单。

### 内置工具

| 工具 | 说明 |
|------|------|
| `bash` | 执行 shell 命令 |
| `grep` | ripgrep 内容搜索 |
| `glob` | 文件模式匹配 |
| `ask_user` | 向用户提问（需要 EventBus） |
| `render_chart` | 内联图表渲染 |
| `sub_agent` | 委托子任务给临时 agent |

### MCP filesystem 工具（自动注册）

| 工具 | 说明 |
|------|------|
| `read_file` | 读取文件内容（**已废弃，用 `read_text_file` 替代**） |
| `read_multiple_files` | 批量读取文件 |
| `read_text_file` | 读取文本文件 |
| `directory_tree` | 目录树 |
| `list_directory` | 列出目录内容 |
| `search_files` | 递归搜索文件 |
| `write_file` | 写文件 |
| `edit_file` | 编辑文件 |
| `get_file_info` | 获取文件信息 |
| `create_directory` | 创建目录 |
| `move_file` | 移动文件 |

**注意**：需要在 agent md 的 `tools` 白名单中显式声明才会生效。推荐使用 `read_text_file` 替代已废弃的 `read_file`。重构类 agent（如 configurator）应包含完整的文件工具集（`read_text_file`, `write_file`, `edit_file`, `grep`, `glob`）。

---

## 快速开始

1. bitx 分析脚本：`python -m src.api.mxint_error_analysis --adapter /path/to/_adapter.py`
2. harness 工作流 (Example 18)：`python examples/18_bitx_mxint_analysis.py /path/to/project`
3. 全链路诊断 (Example 19)：`python examples/19_mxint_diagnostic.py /path/to/project`
4. 通用精度诊断 (Example 20)：`python examples/20_precision_diagnostic.py /path/to/project`
5. UI 启动：`bash examples/launch_ui.sh` → http://localhost:8000
