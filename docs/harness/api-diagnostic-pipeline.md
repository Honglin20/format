# Diagnostic Pipeline API — Design Report

> 生成日期: 2026-06-04
> 状态: 步骤 1 已完成（diagnostic_api.py），步骤 2 设计中

---

## 一、三示例功能对比

| 分析阶段 | bitx 已有 API | Ex18 | Ex19 | Ex20 |
|---------|-------------|------|------|------|
| 项目适配 | `load_adapter()` | ✅ | ✅ | ✅ |
| 多配置 Study | `Session.run()` + `Study` | ✅(单配置) | ✅ | ✅ |
| 精度 Gap 分析 | `TransformEffectReport` | - | ✅ | ✅ |
| 跨配置层排名 | `CrossConfigLayerRanking` | - | ✅ | ✅ |
| 分布诊断 | `DistributionDiagnosis` | - | ✅ | ✅ |
| Block 误差 | `BlockErrorReport` | - | ✅ | ✅ |
| 干预评估 | `InterventionPlanner` | - | ✅ | ✅ |
| **深度衰减** | `ErrorProvenance.depth_decay_data()` | - | **缺失** | **缺失** |
| **误差溯源** | `ErrorProvenance.error_source_analysis()` | - | **缺失** | **缺失** |
| **分布分类学** | `DistributionTaxonomy` / `DistributionFitTaxonomy` | - | **缺失** | **缺失** |
| **误差-分布相关** | `ErrorByDistribution.group_by_range()` | - | **缺失** | **缺失** |
| **层敏感度** | `LayerSensitivity.topk()` / `by_layer_type()` | - | **缺失** | **缺失** |
| **多格式比较** | `compare_formats()` / `rank_formats()` | - | **缺失** | **缺失** |
| Chart 渲染 | `harness_charts.*` | **无** | **无** | ✅ 10张 |

---

## 二、render_chart 双通道用法

render_chart 有两种使用方式，产出的图表出现在不同位置：

### 方式 A：Agent 工具调用（推荐用于 reporter agent）

Agent 直接将 `render_chart` 列为 tools，在推理过程中按需调用。

```
Agent tools: [render_chart, read_text_file, bash]
  ↓
Agent 分析数据 → 提出"为什么 W4A4 损失 17%？" →
  调 render_chart 渲染证据图表 → 继续推理
```

- **显示位置**：Harness Conversation 内联，实时出现在对话流中
- **调用方式**：Agent 的 tool call，与 `bash`/`read_text_file` 平级
- **适用场景**：reporter agent 的分析报告生成

### 方式 B：Python 脚本调用

在 Python 脚本中 import 并调用 `render_chart()`。

```python
from harness.tools.chart import render_chart

render_chart(data, "bar", x="layer", y="qsnr_db", title="Per-Layer QSNR")
```

- **显示位置**：Harness Result 标签页，不在对话中实时可见
- **调用方式**：Agent 通过 `bash` 执行 Python 脚本，脚本内部调用
- **适用场景**：批量预渲染图表（如 harness_charts.py 的 all_harness_charts）

### 选择指南

| 场景 | 用哪种 | 原因 |
|------|--------|------|
| reporter agent 分析报告 | **A: 工具调用** | 图表内联在对话中，形成图文并茂的报告 |
| study_runner 预渲染全部图表 | B: 脚本调用 | 批量生成，不需要推理 |
| 一次性快速查看 | B: 脚本调用 | 简单直接 |
| 需要根据数据动态决定渲染什么 | **A: 工具调用** | Agent 自主选择图表 |

---

## 三、通用 bitx 分析 API（已实现）

### `src/api/diagnostic_api.py` — 三阶段分析函数

| 函数 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `coarse_pass()` | `Dict[str, SessionResult]` | `CoarseReport` | 多配置精度概览 |
| `deep_dive()` | `SessionResult` | `DeepDiveReport` | 单配置层深度分析 |
| `prescribe()` | `SessionResult` | `PrescriptionReport` | 干预推荐 |

每个函数返回的 Report 均支持 `.summary()` + `.to_dict()` + JSON 序列化。

### 可插拔扩展点

- `coarse_pass(bottleneck_fn=...)` — 自定义瓶颈检测器（默认 `detect_wxa_bottleneck`）
- `deep_dive(layers=[...])` — 指定分析层（默认自动选 worst-k）
- `prescribe(strategy="conservative"|"aggressive")` — 干预策略

---

## 四、Reporter Agent 设计

### 数据流

```
上游 agents
  ├── study_runner → 保存 SessionResult 到 output_dir/
  ├── coarse_analyzer → 调 coarse_pass() → 保存 coarse_report.json
  ├── deep_dive_analyst → 调 deep_dive() → 保存 deep_dive_report.json
  └── intervention_explorer → 调 prescribe() → 保存 prescription_report.json
       ↓
  report_painter agent
    tools: [render_chart, read_text_file, bash]
    context: {output_dir, config_names, fp32_accuracy}
    ↓
    ① 读 coarse_report.json → 发现关键问题
    ② 按需读 deep_dive_report.json → 回答问题
    ③ 调 render_chart → 渲染证据图表
    ④ 读 prescription_report.json → 补充建议
    ⑤ 输出学术分析报告
```

### 数据目录（Data Catalog）

Reporter agent 的 MD 中嵌入数据目录，告知 agent 每个文件的 key 结构。
Agent 按需 `read_text_file` 读取，不全量加载。

```
output_dir/
├── coarse_report.json              — 全局精度概览
│   ├── fp32_accuracy               — FP32 基线精度 (float)
│   ├── gaps[]                      — 每个配置的精度差距
│   │     .config                   — 配置名 (str)
│   │     .accuracy                 — 量化精度 (float|null)
│   │     .delta_from_fp32          — 与 FP32 的差值 (float|null)
│   │     .avg_qsnr_db              — 平均 QSNR (float|null)
│   ├── bottleneck                  — 瓶颈判断
│   │     .primary                  — "weight"|"activation"|"both"|"unknown"
│   │     .weight_degradation       — 权重量化导致的精度下降
│   │     .activation_degradation   — 激活量化导致的精度下降
│   ├── consistent_worst[]          — 跨配置一致性最差层
│   │     .layer, .avg_qsnr_db, .worst_config, .dominant_role
│   ├── transform_effects[]         — Transform 恢复效果
│   │     .config, .transform, .accuracy_gain, .recovery_pct
│   ├── distribution_taxonomy[]     — 分布分类学汇总
│   │     .name, .count, .percentage, .avg_metrics
│   └── error_by_range[]            — 按 dynamic range 分桶的误差统计
│         .range_label, .avg_qsnr, .count, .verdict
│
├── deep_dive_report.json           — 细粒度层分析
│   ├── layer_diagnoses[]           — 每层分布诊断
│   │     .layer, .role, .qsnr_db, .classification, .suggestion, .features
│   ├── block_analyses[]            — block/channel 误差
│   │     .layer, .role, .unit_type, .stats, .worst_units, .error_pattern
│   ├── depth_decay[]               — 深度衰减曲线数据点
│   │     .depth, .layer, .qsnr_db
│   ├── error_sources[]             — 误差溯源
│   │     .layer, .output_qsnr, .accum_qsnr, .dominant_role, .error_source
│   ├── sensitivity_topk[]          — 层敏感度排名
│   │     .layer, .role, .value, .layer_type
│   └── layer_type_aggregation[]    — 按层类型聚合
│         .layer_type, .count, .avg_qsnr_db, .avg_mse
│
└── prescription_report.json        — 干预推荐
    ├── boost_targets[]             — 需要提升精度的层
    │     .layer, .current_qsnr, .dominant_role, .action, .reason
    ├── strategies[]                — 恢复策略
    │     .strategy_type, .description, .target_layers, .expected_recovery_pct, .priority
    └── best_strategy               — 最佳策略描述 (str)
```

### Agent 分析流程

```
Step 1: 读 coarse_report.json
  → 了解全局：哪些配置、精度差距多大、瓶颈在哪
  → 提出 2-3 个关键问题：
    "为什么 W4A4 精度比 W8A8 低 17%？"
    "为什么 fc2 和 fc3 在所有配置中都是最差层？"

Step 2: 针对每个问题，读 deep_dive_report.json 的相关 key
  → 查看 layer_diagnoses 找到 fc2 的分类和 suggestion
  → 查看 error_sources 找到误差是 Source 还是 Propagated
  → 查看 depth_decay 看是否有系统性深度衰减
  → 调 render_chart 渲染证据图表（inline in conversation）

Step 3: 读 prescription_report.json
  → 结合发现给出具体建议
  → 渲染推荐图表

Step 4: 综合输出学术报告
  → 每个发现：文字解释 + inline chart
  → 因果推理：数据 → 原因 → 建议
```

---

## 五、实施路径

1. ~~`src/api/diagnostic_api.py` — 三阶段分析函数~~ ✅ 已完成
2. **report_painter agent MD** — 分析指令 + 数据目录 + render_chart 调用
3. **上游 agent 改造** — 保存 JSON 到 output_dir，传递数据目录信息
4. **更新三个示例** — Ex18/19/20 集成 reporter agent
5. **Layer B E2E 测试**
