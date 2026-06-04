# 自定义 Result Type 定义

> **说明**：本文档中 "harness" 指 AgentHarness 项目（`AgentHarness/`），提供 agent 编排 + 结构化输出。

bitx mxint-analysis 工作流使用三个自定义 Pydantic model 作为 agent 输出。

## AnalyzerResult / ProjectAnalysis

analyzer 分析目标项目后输出：

```python
class ProjectAnalysis(BaseModel):
    model_class: str          # nn.Module 类名，如 "ResNet18"
    model_module: str         # dotted import path，如 "models.resnet"
    model_init_args: dict     # 初始化参数，如 {"num_classes": 10}
    dataset: str              # 数据集名，如 "CIFAR-10"
    weights_path: str         # 权重绝对路径或 "NOT_FOUND"
    weights_exist: bool       # 权重文件是否存在
    adapter_exists: bool      # 是否已有 _adapter.py
    adapter_path: str         # 已有 adapter 路径（空字符串=不存在）
    summary: str              # 一句话项目描述
```

## AdapterConfig

configurator 生成配置后输出：

```python
class AdapterConfig(BaseModel):
    adapter_path: str     # adapter 文件绝对路径
    cli_command: str      # 完整 CLI 命令
    w_bits: int = 8       # 权重位宽
    a_bits: int = 8       # 激活位宽
    block_size: int = 16  # per-block block 大小
    device: str = "cpu"   # 设备
    summary: str          # 一句话配置摘要
```

## AnalysisResult

runner 执行分析后输出：

```python
class AnalysisResult(BaseModel):
    status: str                # "success" 或 "error"
    fp32_accuracy: float | None   # FP32 精度
    quant_accuracy: float | None  # 量化后精度
    accuracy_delta: float | None  # delta
    worst_layer: str          # 最差层名
    worst_qsnr_db: float | None   # 最差层 QSNR
    summary: str               # 一句话结果摘要
```

## 使用方式

在 Python API 中通过 `result_type` 参数传入：

```python
from harness.api import Agent, Workflow

wf = Workflow("mxint-analysis", agents=[
    Agent("analyzer", after=[], result_type=ProjectAnalysis),
    Agent("configurator", after=["analyzer"], result_type=AdapterConfig),
    Agent("runner", after=["configurator"], result_type=AnalysisResult),
])

wf.save()  # 生成 workflow.json + agents/*.md

result = wf.run(inputs={"project_path": "/path/to/project"})
# result.outputs["analyzer"] 是 ProjectAnalysis 实例
# result.outputs["configurator"] 是 AdapterConfig 实例
```

**注意**：`result_type` 不序列化到 `workflow.json`。harness 运行时通过 Python API 传入的 `result_type` 自动注入 schema 到 LLM prompt，md 文件中不需要写任何输出格式要求。
