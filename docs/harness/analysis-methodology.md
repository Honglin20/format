# 量化分析方法论 — bitx 原子能力 + Harness 集成状态

> **定位**：本文档是 bitx 库全部量化分析能力的完整清单，标注每项能力的集成状态，
> 并提供分析方法论指导 harness 工作流搭建。随 bitx 能力迭代持续更新。

---

## 一、QSNR 异常诊断（已知问题）

### 1.1 QSNR 300+ dB 根因

**现象**：某些 input 层的 QSNR 达到 280-300 dB。

**根因**：int8 per_block 量化中，当 tensor 在某个 block 内的值恒定（如常数、全零、
高度稀疏），量化后值完全不变 → error = 0 → QSNR 被 `clamp_min(1e-30)` 钳位到 ~300 dB。

**正确做法**：QSNR 高低本身是诊断信号，不做 cap。当 QSNR 极高时，通过 `layer_deep_dive`
深入分析该层的分布特征，回答 "为什么量化对这个层是 trivial 的"。
当 QSNR 极低时，分析分布诊断因果链，回答 "为什么量化对这个层有挑战"。

### 1.2 mxint8 精度与 FP32 一致

**现象**：某些模型 mxint8 准确率与 FP32 完全相同（delta = 0.0000）。

**判断**：这可能是正常的。对于简单模型，int8 per_block 量化精度足够保留预测不变。
验证方法：检查 `qsnr_per_layer`（全 > 35 dB）、`delta`（非零）、per-role QSNR（合理范围）。

**需警惕**：`_is_passthrough=True`、`weight_only=True`、storage_bits=0 + cfg 为空。

---

## 二、bitx 分析能力清单 + 集成状态

### 2.1 Observer 层（数据采集）

| # | Observer | 采集数据 | 集成状态 | 备注 |
|---|----------|----------|:--------:|------|
| O1 | `QSNRObserver` | 每层每角色 QSNR (dB) | ✅ | 基础 |
| O2 | `MSEObserver` | 每层每角色 MSE | ✅ | 基础 |
| O3 | `DistributionObserver` | 分布指纹（crest, kurtosis, outlier_ratio, entropy, skewness, bimodality, sparsity, dynamic_range） | ✅ | `_measure_batch` 已修复，PER_BLOCK 下不再丢失指纹 |
| O4 | `DistributionFitObserver` | 参数化分布拟合 + KS 检验 | ❌ | 需 scipy |
| O5 | `HistogramObserver` | fp32/quant/error 三通道直方图 | ✅ | 三通道 area 叠加 |
| O6 | `PerBlockQSNRObserver` | 每个 block/channel/bank 的独立 QSNR + MSE | ✅ | block 级细粒度 |

### 2.2 5 个原子函数集成状态

| # | 函数 | 功能 | chart_type | 状态 |
|---|------|------|:----------:|:----:|
| F1 | `layer_deep_dive` | 单层三角色完整诊断 | area + table + bar | ✅ |
| F2 | `compare_extreme_layers` | Top-K 极端层对比 + 跨层 block std | table + bar | ✅ |
| F3 | `block_heatmap` | 单层 per-block QSNR 分布 | bar + table | ✅ |
| F4 | `distribution_table` | 全层分布指纹汇总 | table | ✅ |
| F5 | `diagnosis_report` | 因果分析 + scatter 关联 | table + scatter | ✅ |

### 2.3 可视化输出清单

**核心改动**：accum QSNR 为主指标 + linear-only 层过滤 + dist_overlay 双轴分布图。

#### 全局概览层（charts ①–④）

| # | 内容 | chart_type | 数据来源 | 说明 |
|---|------|:----------:|----------|------|
| ① | **Accum QSNR bar** | bar | accum_qsnr_per_layer | 主指标，linear-only，替代原 ①② |
| ② | Accuracy Summary | table | fp32/quant/delta | 不变 |
| ③ | **Accum vs Local** | line + hue=type | accum + qsnr_per_layer | 唯一保留 local 的图 |
| ④ | **Per-Role Local QSNR** | bar + hue=role | qsnr_by_role | linear-only，标注 Local |

#### 误差归因层（charts ⑤–⑥）

| # | 内容 | chart_type | 数据来源 | 说明 |
|---|------|:----------:|----------|------|
| ⑤ | **Error Attribution** | bar + table | accum_qsnr + qsnr_by_role | accum-based，linear-only |
| ⑥ | Cost decomposition | bar | result.cost | 不变 |

#### 极端层分析层（charts ⑦–⑨，`compare_extreme_layers`）

| # | 内容 | chart_type | 数据来源 | 说明 |
|---|------|:----------:|----------|------|
| ⑦ | **Extreme Layer Table** | table | accum_qsnr_per_layer | accum QSNR 排序，worst-3 + best-3 |
| ⑧ | Cross-layer block std | bar + hue=role | O6 | linear-only 过滤 |
| ⑨ | Per-layer block std | bar + hue=role | O6 | 不变 |

#### 逐层 deep dive 层（`layer_deep_dive`，每个 extreme layer × 3 roles）

| # | 内容 | chart_type | 数据来源 | 说明 |
|---|------|:----------:|----------|------|
| ⑩ | 分布指纹表 | table | O3 | 不变 |
| ⑪ | **dist_overlay 分布** | **dist_overlay** | O5 | fp32 蓝填充 + quant 红虚线 + error 灰右轴 |
| ⑫ | Per-Block QSNR 统计 | table | O6 | 不变 |
| ⑬ | Top-5 Worst Blocks | bar | O6 | 不变 |
| ⑭ | 角色归因表 | table | qsnr_by_role | 不变 |

> ⑩–⑭ 对每个极端层的每个 role 输出。3 层 × 3 角色 = ~15 charts/层。

#### 全局诊断层（不变，`distribution_table` + `diagnosis_report`）

| # | 内容 | chart_type | 数据来源 | 说明 |
|---|------|:----------:|----------|------|
| ⑮ | 全层分布指纹表 | table | O3 | 不变 |
| ⑯ | 因果分析表 | table | O3 + O1 | 不变 |
| ⑰ | Crest vs QSNR | scatter | O3 + O1 | 不变 |
| ⑱ | Outlier vs QSNR | scatter | O3 + O1 | 不变 |

#### 精度恢复层（可选，不变）

| # | 内容 | chart_type | 数据来源 |
|---|------|:----------:|----------|
| ⑲ | Precision Recovery | bar | session.run overrides |
| ⑳ | Recovery Accuracy | bar | session.run overrides |

### 2.4 Harness chart_type 使用统计

| chart_type | 数量 | 用途 |
|:----------:|:----:|------|
| table | ~18 | 分布指纹、block 统计、角色归因、诊断、精度对比 |
| bar | ~10 | accum QSNR、error attribution、block std、worst blocks |
| **dist_overlay** | **~9** | **三通道双轴分布（fp32/quant/error）** |
| line | 1 | accum vs local comparison |
| scatter | 2 | crest vs QSNR、outlier vs QSNR |
| **box** | **~1** | **跨层 per-block QSNR 分布对比 (U6)** |
| **合计** | **~41** | |

### 2.5 已集成：U1–U6 + 补充可视化 (harness_charts.py)

新增 `src/api/harness_charts.py` 作为双路适配层：
- **Path 1 (harness)**: 通过 `render_chart()` 发送到 AgentHarness 前端
- **Path 2 (matplotlib)**: 通过 `output_dir` 参数可选保存 PNG

| # | 能力 | 函数 | chart_type | 来源 |
|---|------|------|:----------:|------|
| U1 | 参数化分布拟合 | `distribution_fit_chart()` | table + bar | O4 DistributionFitObserver |
| U2 | 干预规划 | `intervention_chart()` | table + bar | InterventionPlanner |
| U3 | 通道异质性 | `channel_heterogeneity_chart()` | table + bar | PerBlockQSNRObserver (channel) |
| U4 | 深度衰减 | `depth_decay_chart()` | line + table | ErrorProvenance.depth_decay_data |
| U5 | 误差传播分类 | `error_propagation_chart()` | bar + table | qsnr_by_role + accum_qsnr |
| U6 | 箱线图对比 | `block_qsnr_box_chart()` | **box** + table | PerBlockQSNRObserver (block) |
| — | Block error | `block_error_chart()` | bar + table | PerBlockQSNRObserver |
| — | Channel error | `channel_error_chart()` | bar | PerBlockQSNRObserver |
| — | Error provenance | `error_provenance_chart()` | bar + table | ErrorProvenance |
| — | 跨配置排名 | `cross_config_ranking_chart()` | bar + table | CrossConfigLayerRanking |
| — | 变换效果 | `transform_effect_chart()` | bar | TransformEffectReport |
| — | 多配置块对比 | `multi_config_block_comparison()` | bar | StudyReport |

**一键调用**: `all_harness_charts(result, label=..., output_dir=...)`

---

## 三、Bug 修复记录

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| B1 | PerBlockQSNRObserver 导致聚合 QSNR 被拉低 | `_extract_all_roles_qsnr_mse` 遍历 per-block 条目，取了 min | 跳过 `("block", i)` / `("channel", i)` / `("bank", i)` 条目 |
| B2 | DistributionObserver 在 PER_BLOCK 下不产生分布指纹 | `_measure_batch` 只返回 MSE，覆盖了默认的 `_measure` 调用 | `_measure_batch` 合并 `_measure` 的分布指纹到 aggregate entry |
| B3 | render_table ⑦ 缺少相对精度 | 只有绝对 delta | 增加 `relative_delta_pct` + `quant_pct_of_fp32` 列 |

---

## 四、分析方法论

### 4.1 从粗到细的分析层次

```
Level 0: 全局精度概览
  → fp32_metrics, quant_metrics, delta, relative_delta_pct
  → 判断是否有精度问题

Level 1: 层级误差定位
  → qsnr_per_layer → 最差/最好层
  → qsnr_by_role → 哪个角色最差
  → 跨层 block std → 哪些层 block 间最不均匀

Level 2: 分布诊断（为什么）
  → 三通道分布叠加（area: fp32/quant/error）
  → 分布指纹（crest, outlier_ratio, ...）
  → 分类（outlier_dominated? heavy_tailed? bimodal?）
  → scatter 关联（crest vs QSNR, outlier_ratio vs QSNR）

Level 3: Block 级定位（差在哪里）
  → per-block QSNR 分布（bar: 全 blocks 排序）
  → worst blocks 排名（bar: top-5）
  → block 统计（table: mean/std/min/max）

Level 4: 干预方案（后续）
  → InterventionPlanner → 哪些层提高精度
  → Precision Recovery → 逐层恢复验证
```

### 4.2 Harness 工作流（当前已集成）

```
mxint_error_analysis.py 调用流程:

1. Session 挂载 6 个 Observer
   [QSNRObserver, MSEObserver, DistributionObserver,
    HistogramObserver, PerBlockQSNRObserver]

2. charts_from_result (charts ①–⑧)
   全局概览：accum QSNR bar + accuracy table + accum vs local + per-role
   误差归因：waterfall + cost

3. compare_extreme_layers(top_k=3, linear_only=True)
   → 极端层总览表（accum QSNR）
   → 跨层 block std 对比 bar（linear-only）
   → 对每个极端层调用 layer_deep_dive:
      → 分布指纹表
      → dist_overlay 三通道双轴分布
      → Per-block QSNR 统计 + top-5 worst blocks
      → 角色归因表

4. distribution_table()
   → 全层分布指纹表

5. diagnosis_report()
   → 因果分析表 + scatter × 2

6. charts_precision_recovery()（可选）
   → 精度恢复 + accuracy comparison

输出: ~40 charts, 6 chart_type
核心: accum QSNR + linear-only + dist_overlay
```

---

## 五、集成优先级

### P0: 已完成 ✅

1. ✅ 增强 render_table ⑦（relative_delta_pct + quant_pct_of_fp32）
2. ✅ 还原 QSNR cap（不做钳位，改为逐层深度诊断）
3. ✅ Bug B1: `_extract_all_roles_qsnr_mse` 跳过 per-block 条目
4. ✅ Bug B2: DistributionObserver._measure_batch 合并分布指纹
5. ✅ 6 个 Observer 全部挂载到 Session

### P1: 已完成 ✅

6. ✅ `layer_deep_dive` — 三通道 area 叠加 + block stats + 分类
7. ✅ `compare_extreme_layers` — 极端层对比 + 跨层 block std
8. ✅ `block_heatmap` — per-block QSNR 分布（bar 排序）
9. ✅ `distribution_table` — 全层指纹表
10. ✅ `diagnosis_report` — 因果表 + crest/outlier scatter

### P2: 已完成 ✅

11. ✅ U1–U6 全部集成到 `src/api/harness_charts.py`
12. ✅ 双路适配：render_chart + 可选 matplotlib
13. ✅ box chart_type: 跨层 per-block QSNR 分布对比
14. ✅ 15 个新单元测试通过

---

## 六、变更记录

| 日期 | 变更 |
|------|------|
| 2026-06-03 | 初版：能力审计 + QSNR 异常根因 + 方法论 |
| 2026-06-03 | 确认方案：K=3, 5 原子函数，不做 QSNR cap |
| 2026-06-03 | 实现 5 原子函数 + 修复 3 个 bug + 集成到 mxint_error_analysis |
| 2026-06-03 | 修复分布可视化：bar→area 三通道叠加(fp32/quant/error) + block_heatmap 格式修正 + scatter 关联图 + 跨层 block std |
| 2026-06-03 | 新增 `harness_charts.py` 双路适配：U1–U6 + block/channel/provenance 全部接入 render_chart，box chart_type 使用，15 测试通过 |
