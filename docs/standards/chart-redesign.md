# Chart Redesign Spec — mxint_error_analysis

> 状态：已确认
> 分支：main

---

## 一、MXInt8 主标签图表

### P1: Accum QSNR bar → 按模型层序排列

**当前**：按 QSNR 值排序（worst first），丢失模型层序信息。
**改为**：横轴 = 层名（按模型 forward 顺序），纵轴 = accum QSNR (dB)。
**实现**：`accum_qsnr_bar()` 中用 `observers_data` 的 key 顺序替代 `_sorted_layers()` 的 QSNR 排序。

---

### P2: Accum vs Local line — 不动

---

### P3: Per-Role Local QSNR grouped bar → 过滤高 QSNR

**当前**：QSNR > 100 dB 的层干扰视觉。
**改为**：过滤 QSNR > 100 dB 的数据点，标题注明 "(QSNR ≤ 100 dB only)"。

---

### P4: Error Attribution waterfall → 去重 + 过滤

**当前问题**：
1. `charts_error_attribution()` 与 P4 完全重复
2. error_contribution < 0 无意义（数值噪音）

**改为**：
- 删除 `charts_error_attribution()` 调用，只保留 `error_attribution_waterfall()`
- 过滤 error_contribution < 0
- 标题注明 "(positive error only)"
- 同步 P3 的 QSNR > 100 过滤

---

### Precision Recovery ⑨ → 合并前后精度

**当前**：⑨ recovery_pct + ⑩ actual accuracy，两张图信息重叠，未注明单层/累积。
**改为**：
- 合并为一张 grouped bar：每个 layer 显示 baseline_quant 精度 + restored 精度 + FP32 baseline
- 标题注明 "(single-layer restore: each bar restores exactly 1 layer to FP32, others stay quantized)"
- 删除 ⑩ Actual Accuracy chart

---

### Cost bar (⑥) → 删除

非误差分析。

---

### Extreme Layer Table (P5) → 删除

与 `compare_extreme_layers()` 内的 table 重复。

---

### U2: Intervention — 只保留 table

- U2a table（建议表）：保留
- U2b bar（target layers QSNR）：删除，与 P1 重复

---

### U6: Per-Block QSNR Box Plot → 修正轴标签

**当前**：横轴标签 "value"，方向反了。
**改为**：横轴 = layer (group)，纵轴 = QSNR (dB)。

---

### 删除项

| 删除 | 理由 |
|------|------|
| `charts_error_attribution()` | 与 P4 重复 |
| `main()` 第二次 `compare_extreme_layers()` | 重复调用 |
| Cost bar (⑥) | 非误差分析 |
| Extreme Layer Table (P5) | 与 compare_extreme_layers table 重复 |
| ⑩ Actual Accuracy | 与 ⑨ 合并 |
| U2b Intervention Target bar | 与 P1 重复 |
| U1 Distribution Fit | 与 distribution_table + diagnosis 重叠 |
| U4 Depth Decay | 与 P2 重叠 |
| U5 Error Source Classification | 与 P4 重叠 |
| Error Provenance | 与 P4 重叠 |
| Per-layer channel/block bar charts | 被 heatmap 替代 |

---

## 二、Block QSNR Heatmap（新增）

### 通用规则

MXInt 量化沿最后一轴进行。对任意 tensor shape `(..., N)`：
1. Flatten leading dims → `(D, N)`，其中 `D = product of all dims except last`
2. Block 沿 `N` 维度 → `num_blocks = N // block_size`
3. Heatmap 形状 = `(D, num_blocks)`

**物理含义**：每个像素 = 该 block 的 quantization QSNR (dB)，即量化前后该 block 的误差比。颜色越深 = 量化误差越大。排布与实际量化过程一致。

### Weight heatmap

- Linear: `(out_features, in_features)` → `(out_features, in_features // block_size)`
- Conv2d: `(out_ch, in_ch, kH, kW)` → `(out_ch × in_ch × kH, kW // block_size)`
- 行 = 输出通道（或 flatten 后的位置），列 = block

### Input / Output heatmap

- 对所有 leading dims（含 batch）取平均，只保留最后两维
- Linear input `(batch, features)` → avg batch → `(features,)` → `(1, features // block_size)`
- Conv2d input `(batch, C, H, W)` → avg batch,C → `(H, W)` → `(H, W // block_size)`
- 行 = 倒数第二维（如 spatial H），列 = block

### Harness 渲染

已确认支持 `heatmap` chart type：
- 数据格式：`[{x: col_idx, y: row_idx, value: qsnr_db}, ...]`
- 每轴 max 50 unique values

### 实现要点

**不需要改 observer。** `_store_per_block` 将 tensor reshape 为 `(-1, block_size)`，
block index 是 row-major flat index。只要知道原始 tensor shape 即可 reshape 回 2D。

Shape 获取方式：
- **Weight**：从 model parameters 直接拿 `layer.weight.shape`
- **Input/Output**：从 `total_blocks / (last_dim // block_size)` 推断 leading dim 乘积
  - Linear: `last_dim = weight.shape[1]`（input）或 `weight.shape[0]`（output）
  - leading_dim_product = total_blocks / num_blocks_per_row

对于 Linear 层：total_blocks = batch × (features // block_size)，
所以 heatmap 形状 = (batch, features // block_size)，无需额外信息。

---

## 三、Best/Worst Deep Dive（top_k=1）

每个 extreme layer（1 worst + 1 best = 2 layers）生成：

```
layer_deep_dive(layer):
  # 1. Distribution Overlay × 3 roles（先画，直觉优先）
  dist_overlay(layer, "input")
  dist_overlay(layer, "weight")
  dist_overlay(layer, "output")

  # 2. Block QSNR Heatmap × 3 roles（空间排布）
  block_qsnr_heatmap(layer, "input")    # (batch, num_blocks)
  block_qsnr_heatmap(layer, "weight")   # (out_ch, num_blocks)
  block_qsnr_heatmap(layer, "output")   # (batch, num_blocks)
```

**移除的子图**（原 `layer_deep_dive` 生成）：
- Distribution Fingerprint table → 被 dist_overlay 覆盖
- Per-Block QSNR Stats table → 被 heatmap 覆盖
- Top-5 Worst Blocks bar → 被 heatmap 替代
- Role Attribution table → 被 P4 Error Attribution 覆盖

**Block QSNR Heatmap 只出现在 deep dive**，主标签不画 heatmap。

---

## 四、改后图表总览

```
MXInt8 主标签:
  P1  Accum QSNR bar（按模型层序）
  P2  Accuracy Summary table
  P3  Accum vs Local line
  P4  Per-Role Local QSNR bar（QSNR ≤ 100）
  P5  Error Attribution waterfall（正误差 only）
  —   Extreme Layers table + Block Std bar（compare_extreme_layers, top_k=1）
  ⑨   Precision Recovery grouped bar（baseline + restored + FP32）
  —   Distribution Fingerprint table
  —   Causal Analysis table + 2 scatters
  U2  Intervention Plan table
  U6  Per-Block QSNR Box Plot（轴标签修正）

Deep Dive (×2 layers: top-1 worst + top-1 best):
  dist_overlay × 3 roles
  block_qsnr_heatmap × 3 roles
```

总计约 15（主标签）+ 12（deep dive）= ~27 图。

---

## 五、涉及文件

| 文件 | 改动 |
|------|------|
| `src/api/mxint_error_analysis.py` | 删除重复调用、合并 precision recovery、删 cost bar、P1 改层序 |
| `src/api/layer_diagnostic.py` | 重写 `layer_deep_dive`：只画 dist_overlay + heatmap |
| `src/api/harness_charts.py` | `all_harness_charts` 精简为 U2a + U6，删 U2b/U6b |
| `src/api/_chart_helpers.py` | 新增 `block_qsnr_heatmap()` helper，从 model params 推断 shape |
