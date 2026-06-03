# Charts — render_chart 集成

> **说明**：本文档中 "harness" 指 AgentHarness 项目（`AgentHarness/`），提供前端可视化。

bitx 分析脚本通过 `render_chart()` 向 AgentHarness 前端输出图表。

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
