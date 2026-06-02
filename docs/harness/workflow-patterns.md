# 工作流设计模式

> **说明**：本文档中 "harness" 指 AgentHarness 项目（`AgentHarness/`），提供 agent 编排能力。

## 核心机制：结构化输出自动注入

harness 的结构化输出 **完全由 Python API 的 `result_type` 参数控制**，md 文件无需、也不应包含任何输出格式指令。

工作原理（`harness/engine/macro_graph.py:479-492`）：

1. Agent 定义时传入 `result_type=YourPydanticModel`
2. harness 调用 `model_json_schema()` 提取 JSON Schema
3. 自动在 system prompt 末尾追加 `## Output Format` + schema
4. Pydantic AI 强制 LLM 输出匹配 schema 的结构化 JSON

```
md 文件内容（你写的）        harness 自动追加的
┌──────────────────┐      ┌──────────────────────┐
│ 任务描述          │  +   │ ## Output Format      │
│ 策略              │      │ { "model_class": str, │
│ 注意事项          │      │   "dataset": str, ...}│
└──────────────────┘      └──────────────────────┘
```

## 两种启动方式

### 方式一：workflow.json（UI 启动）

```
workflows/mxint-analysis/
  workflow.json           # agent 定义 + DAG（无 result_type）
  agents/*.md             # 纯任务描述（无输出格式要求）
```

无 `result_type` → 走默认 `AgentResult(summary, details)`。

### 方式二：Python API（程序化启动）

```python
class ProjectAnalysis(BaseModel):
    model_class: str
    dataset: str
    ...

wf = Workflow("mxint-analysis", agents=[
    Agent("analyzer", after=[], result_type=ProjectAnalysis, ...),
])
wf.save()  # 保存 workflow.json + agents/*.md
```

`result_type` → harness 自动注入 schema，Pydantic AI 强制输出结构化 JSON。

**关键**：`result_type` 不序列化到 `workflow.json`，只在 Python API 中生效。因此：
- md 文件 **只写任务逻辑**，不写输出格式
- Python examples 中 **定义结构体 + `wf.save()`** 即可

## 两条铁律

### 1. md 文件不写输出格式

md 文件是纯任务指令。harness 会根据 `result_type` 自动注入 `## Output Format`。
手动在 md 里要求 summary/details/JSON 格式会导致：
- 和自动注入的 schema 冲突
- LLM 浪费 token 处理重复指令
- 输出格式不稳定

### 2. 结构体只在 Python examples 中定义

在 `examples/` 中定义 Pydantic model，传给 `Agent(result_type=...)`，然后 `wf.save()`。
harness 运行时自动处理 schema 注入和输出解析。

## bitx mxint-analysis 工作流

```
examples/18_bitx_mxint_analysis.py
  ├─ ProjectAnalysis (result_type for analyzer)
  ├─ AdapterConfig   (result_type for configurator)
  └─ AnalysisResult  (result_type for runner)
```

```
workflows/mxint-analysis/
  ├─ workflow.json              # DAG 定义（由 save() 生成）
  └─ agents/
      ├─ analyzer.md            # 纯任务指令
      ├─ configurator.md        # 纯任务指令
      └─ runner.md              # 纯任务指令
```

每个 result_type 的字段定义见 [result-types.md](result-types.md)。
