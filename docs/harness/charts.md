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

## 注意事项

- `label` + `title` 相同会替换已有图表（支持实时更新）
- `data` 格式等同于 `DataFrame.to_dict("records")`
- 不在 harness 环境下运行时，render_chart 静默跳过（import 失败时不报错）
