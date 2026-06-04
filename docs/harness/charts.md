# Charts — render_chart 集成

> **说明**：本文档中 "harness" 指 AgentHarness 项目（`AgentHarness/`），提供前端可视化。

bitx 通过 `render_chart` 向 AgentHarness 前端输出图表。有两种使用方式。

## render_chart 双通道用法

### 方式 A：Agent 工具调用（inline in conversation）

Agent 将 `render_chart` 列为 tools，推理过程中直接调用。

```yaml
# workflow agent 配置
Agent("report_painter", tools=["render_chart", "read_text_file", "bash"])
```

Agent 在分析数据时，根据推理需要自主决定渲染什么图表：

```
Agent: "W4A4 精度比 W8A8 低 17%，让我看看各层的 QSNR 分布..."
→ 调用 render_chart(data, "bar", x="layer", y="qsnr_db", title="Per-Layer QSNR")
→ 图表内联出现在对话中
→ Agent 继续分析: "fc2 的 QSNR 仅 18dB，我再看一下 block 级别..."
→ 调用 render_chart(block_data, "box", ...)
```

- **显示位置**：Harness Conversation 内联，实时出现在对话流中
- **特点**：Agent 自主决定渲染什么、何时渲染，适合分析报告

### 方式 B：Python 脚本调用（result tab）

在 Python 脚本中 import 并调用：

```python
from harness.tools.chart import render_chart

render_chart(data, "bar", x="layer", y="qsnr_db", title="Per-Layer QSNR")
```

Agent 通过 `bash` 执行脚本：

```
Agent: 调用 bash → python scripts/render_charts.py --result-dir ./output
```

- **显示位置**：Harness Result 标签页，**不在对话中实时可见**
- **特点**：批量预渲染全部图表，适合一次性生成

### 选择指南

| 场景 | 推荐方式 |
|------|---------|
| reporter agent 学术分析报告 | **A: 工具调用** |
| 批量预渲染全部图表 | B: 脚本调用 |
| 需要根据数据动态决定渲染什么 | **A: 工具调用** |

## 传输机制

`render_chart()` 尝试三个通道（按优先级）：
1. **EventBus**：同一进程内，零延迟
2. **Stdout capture**：bash tool 识别 `__HARNESS_CHART__:` 前缀，自动转发
3. **HTTP POST**：发送到 `/api/charts`（需要 `HARNESS_SERVER_URL` 环境变量）

脚本通过 bash 执行时走通道 2，无需额外配置。

## 当前图表清单

### 基础分析（①–⑦）

| 编号 | 图表 | 类型 | x | y | hue |
|------|------|------|---|---|-----|
| ① | Per-Layer QSNR | bar | layer | qsnr_db | — |
| ② | Per-Layer MSE | bar | layer | mse | — |
| ③ | Error Propagation | line | layer_idx | qsnr_db | type |
| ④ | Per-Role QSNR | bar | layer | qsnr_db | role |
| ⑤ | FLOPs per Layer | bar | op_name | flops_math | — |
| ⑥ | Top-K Worst Layers | bar | layer | qsnr_db | — |
| ⑦ | Accuracy Summary | table | metric | fp32 | — |

### 误差归因 + 精度恢复（⑧–⑩）

| 编号 | 图表 | 类型 | x | y | hue |
|------|------|------|---|---|-----|
| ⑧ | Error Attribution | bar | layer | error_contribution | source |
| ⑧ | Attribution Table | table | layer | output_qsnr | — |
| ⑨ | Precision Recovery % | bar | layer | recovery_pct | dominant_error |
| ⑩ | Actual Accuracy | bar | layer | accuracy | config |

## 添加新图表

### 基础图表

```python
from harness.tools.chart import render_chart

render_chart(
    data=[{"layer": "fc1", "value": 42.3}, ...],  # list[dict]
    chart_type="bar",                              # line/bar/scatter/table/heatmap/...
    x="layer",                                      # x 轴列名
    y="value",                                      # y 轴列名
    label="MXInt8",                                 # 分组标签
    title="My Chart",                               # 图表标题
    hue="type",                                     # 颜色分组列名（可选）
)
```

### dist_overlay — 双轴分布叠加图

`dist_overlay` 是通用元绘图函数，支持多数据系列在双 Y 轴上以 Area 或 Line 样式渲染。
适用于量化前后分布对比、多信号叠加等场景。

```python
render_chart(
    data=[
        {"bin": -0.52, "fp32": 120, "quant": 118, "error": 2},
        {"bin": -0.48, "fp32": 350, "quant": 340, "error": 10},
        {"bin": -0.44, "fp32": 280, "quant": 275, "error": 5},
        ...
    ],
    chart_type="dist_overlay",
    x="bin",
    series=[
        {"key": "fp32",  "type": "area", "axis": "left",  "color": "#5B8DB8",
         "fillOpacity": 0.25, "step": True, "label": "FP32"},
        {"key": "quant", "type": "line", "axis": "left",  "color": "#D4605A",
         "dash": "6 3", "label": "Quant"},
        {"key": "error", "type": "area", "axis": "right", "color": "#9CA3AF",
         "fillOpacity": 0.3, "step": True, "label": "Error"},
    ],
    title="layer3.linear (weight) Distribution",
)
```

**数据格式**：宽格式，每行一个 x 值，每列一个数据系列。

**series 配置**：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `key` | str | **是** | — | data 中的列名 |
| `type` | str | 否 | `"area"` | `"area"` 或 `"line"` |
| `axis` | str | 否 | `"left"` | `"left"`（主轴）或 `"right"`（副轴） |
| `color` | str | 否 | PALETTE 轮换 | hex 颜色覆盖 |
| `fillOpacity` | float | 否 | 0.2 | Area 填充透明度 |
| `dash` | str | 否 | 实线 | stroke-dasharray，如 `"6 3"` |
| `step` | bool | 否 | False | 是否使用阶梯插值 |
| `label` | str | 否 | key 值 | 图例显示名 |
| `strokeWidth` | float | 否 | 1.5 | 线宽 |

**使用场景**：
- 量化前后分布对比（fp32 vs quant vs error）
- 多信号叠加（预测值 vs 真实值 vs 残差）
- 任意需要双轴区分量级的可视化

## 注意事项

- `label` + `title` 相同会替换已有图表（支持实时更新）
- `data` 格式等同于 `DataFrame.to_dict("records")`
- 不在 harness 环境下运行时，render_chart 静默跳过（import 失败时不报错）

## Harness 修改规则

> **重要**：AgentHarness (`AgentHarness/`) 是独立项目，不属于 bitx。
> 如需修改 harness 的 chart type、前端组件、render_chart 行为，
> **必须先与用户同步确认**，不可直接修改。

具体场景：
- 需要新增 chart type（如 harness 没有 heatmap 的数据格式不符合需求）
- 需要修改现有 chart type 的渲染行为
- 需要修改 render_chart 的参数签名
- 需要修改前端组件

不需要同步的场景：
- 在 bitx `src/api/harness_charts.py` 中新增使用现有 chart type 的函数
- 修改 bitx 自己的数据提取逻辑
- 修改 bitx 的 matplotlib 可视化 (`src/viz/`)

## bitx 双路可视化架构

```
src/api/harness_charts.py  ← 新增：render_chart 适配层
         ↓ 调用
   render_chart()           ← harness 前端 (可选)
         ↓ 可选
   src/viz/*.py             ← matplotlib 保存 PNG (可选, output_dir)
```

- `harness_charts.py` 不修改 `src/viz/` 任何代码
- 每个函数同时支持两条路径
- `output_dir=None` 时只走 render_chart
- harness 不可用时 render_chart 是 no-op
