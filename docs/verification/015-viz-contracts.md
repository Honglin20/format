# 015: Viz Module — Visualization Function Contracts

**对应测试函数**: `test_viz_theme_constants()`, `test_viz_save_figure()`, `test_viz_tables()`, `test_viz_figures()`
**验证层级**: Layer 4 — Pipeline Refactor

## 验证内容

Refactored `src/viz/` — extract visualization functions from `examples/experiment_format_study.py` into a reusable, parameterized module. Each function is PURE: receives data, returns chart/table, no side effects except file I/O via `save_figure`. All hardcoded titles and colors become parameters.

### 模块结构

```
src/viz/
  __init__.py    # re-export public API
  theme.py       # color palette constants
  save.py        # file I/O utility
  tables.py      # 4 table generators (+ 2 shared helpers)
  figures.py     # 9 figure generators (+ 2 shared helpers)
```

---

## 合约

### Module 1: `src/viz/theme.py` — Color Constants

**所有值均为模块级常量**（不可变、作为纯数据字典使用）。

| 符号 | 类型 | 条目数 | 说明 |
|------|------|--------|------|
| `FORMAT_COLORS` | `Dict[str, str]` | 7 | 格式名 → hex color 映射 (Wong 2011 colourblind-friendly) |
| `TRANSFORM_COLORS` | `Dict[str, str]` | 3 | transform 名 → hex color 映射 |
| `HIST_COLORS` | `Dict[str, str]` | 3 | 直方图通道名 → hex color 映射 |
| `FALLBACK_CYCLE` | `List[str]` | 10 | 颜色回退列表（当指定键不在 FORMAT_COLORS 中时使用） |

#### 颜色映射细节

**FORMAT_COLORS** (7 entries):

| 键 | 颜色 | 十六进制 |
|----|------|----------|
| `"MXINT-8"` | 蓝 (blue) | `#0072B2` |
| `"MXFP-8"` | 朱红 (vermillion) | `#D55E00` |
| `"INT8-PC"` | 蓝绿 (bluish green) | `#009E73` |
| `"MXINT-4"` | 天蓝 (sky blue) | `#56B4E9` |
| `"MXFP-4"` | 橙 (orange) | `#E69F00` |
| `"INT4-PC"` | 黄 (yellow) | `#F0E442` |
| `"NF4-PC"` | 紫 (reddish purple) | `#CC79A7` |

**TRANSFORM_COLORS** (3 entries):

| 键 | 颜色 | 十六进制 |
|----|------|----------|
| `"None"` | 蓝 (blue) | `#0072B2` |
| `"SmoothQuant"` | 朱红 (vermillion) | `#D55E00` |
| `"Hadamard"` | 蓝绿 (bluish green) | `#009E73` |

**HIST_COLORS** (3 entries):

| 键 | 颜色 | 十六进制 |
|----|------|----------|
| `"fp32_hist"` | 蓝 (blue) | `#0072B2` |
| `"quant_hist"` | 朱红 (vermillion) | `#D55E00` |
| `"err_hist"` | 灰 (grey) | `#999999` |

**FALLBACK_CYCLE** (10 entries): Wong 2011 colourblind-friendly palette ordered by contrast:
`["#0072B2", "#D55E00", "#009E73", "#F0E442", "#CC79A7", "#56B4E9", "#E69F00", "#999999", "#000000", "#E5C494"]`

---

### Module 2: `src/viz/save.py` — Save Utility

#### `save_figure(fig, output_dir, name) -> str`

Saves a matplotlib `Figure` to `<output_dir>/figures/<name>.png` and `<output_dir>/figures/<name>.pdf`. Creates the directory if it does not exist.

```python
def save_figure(fig: matplotlib.figure.Figure, output_dir: str, name: str) -> str:
    ...
```

**输入参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `fig` | `matplotlib.figure.Figure` | 是 | 待保存的 matplotlib 图片对象 |
| `output_dir` | `str` | 是 | 根输出目录（`figures/` 子目录自动创建） |
| `name` | `str` | 是 | 不带后缀的文件名（如 `"fig1_qsnr_8bit"`） |

**返回值**: `str` — 保存的 PNG 文件的绝对路径（形如 `<output_dir>/figures/<name>.png`）。

**副作用**: 在 `<output_dir>/figures/` 下创建两个文件：`<name>.png` (300 dpi, `bbox_inches="tight"`) 和 `<name>.pdf` (300 dpi, `bbox_inches="tight"`)。随后调用 `plt.close(fig)` 关闭图片释放内存。

**异常与守卫**:

| 场景 | 期望行为 |
|------|---------|
| `output_dir` 目录不存在 | 自动创建（`os.makedirs` 递归） |
| `name` 含路径分隔符（如 `"sub/fig1"`） | 在 `figures/sub/` 下创建文件（用户负责确保合法性） |
| 文件已存在 | 静默覆盖（不产生错误） |
| `fig` 为 `None` | `AttributeError` 向上传播 |
| `output_dir` 为 `None` | `TypeError` 向上传播（`os.path.join` 抛出） |

---

### Module 3: `src/viz/tables.py` — Table Functions

#### 公共签名约定

每个 table 函数：
1. 接收 results dict + 关键字参数
2. 打印格式化的文本表格到 stdout（直接 `print`，属于 side effect，但这是 CLI 工具约定行为）
3. 写入 CSV 文件到 `<output_dir>/tables/`
4. 返回格式化的字符串（可用于进一步写入文件或日志）

#### `accuracy_table(results, *, title, output_dir, filename) -> str`

通用 accuracy + QSNR/MSE 摘要表格。将每个配置的 accuracy、平均 QSNR 和平均 MSE 排成行。

```python
def accuracy_table(
    results: Dict[str, Dict[str, Any]],
    *,
    title: str,
    output_dir: str,
    filename: str,
) -> str:
    ...
```

**输入参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `results` | `Dict[str, dict]` | 是 | 配置名 → 结果 dict。（结果 dict 结构见 "输入数据结构" 一节） |
| `title` | `str` (keyword-only) | 是 | 表格标题（分隔线包裹显示） |
| `output_dir` | `str` (keyword-only) | 是 | CSV 输出根目录 |
| `filename` | `str` (keyword-only) | 是 | CSV 文件名（如 `"table1_8bit.csv"`） |

**返回值**: `str` — 格式化文本表格字符串（含分隔线、标题、列头、数据行）。

**输出文件**: `<output_dir>/tables/<filename>` (CSV 格式: `Config,Accuracy,Avg_QSNR_dB,Avg_MSE`)

#### `format_comparison_table(results, *, title, output_dir, filename) -> str`

`accuracy_table` 的别名。完全相同的行为。提供此别名以在调用点更准确地表达意图。

```python
def format_comparison_table(results, *, title, output_dir, filename) -> str:
    return accuracy_table(results, title=title, output_dir=output_dir, filename=filename)
```

#### `pot_scaling_table(results, *, output_dir) -> str`

FP32 vs PoT scaling 对比表 (Table 3)。从 results 中提取 baseline 精度，对每个非 baseline 配置计算 delta（量化精度 − baseline），展示 accuracy、delta、avg QSNR、avg MSE。

```python
def pot_scaling_table(
    results: Dict[str, Dict[str, Any]],
    *,
    output_dir: str,
) -> str:
    ...
```

**输入参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `results` | `Dict[str, dict]` | 是 | Part C results。含一个键含 `"baseline"` 的 entry（如 `"FP32 (baseline)"`） |
| `output_dir` | `str` (keyword-only) | 是 | CSV 输出根目录 |

**输出文件**: `<output_dir>/tables/table3_pot.csv` (CSV 格式: `Config,Accuracy,Delta,Avg_QSNR_dB,Avg_MSE`)

**Baseline 查找规则**: 从 `results` 中寻找键含 `"baseline"`（大小写不敏感）的 entry，取其 accuracy 值作为基准。无 baseline 时 delta 为 `0.0`。

#### `transform_matrix_table(results, *, output_dir, filename) -> str`

Format x Transform accuracy 矩阵表 (Table 4)。行 = format 名，列 = transform 变体，单元格 = accuracy 值。

```python
def transform_matrix_table(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    output_dir: str,
    filename: str,
) -> str:
    ...
```

**输入参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `results` | `Dict[str, dict]` | 是 | 嵌套 dict：format 名 → transform 名 → 结果 dict |
| `output_dir` | `str` (keyword-only) | 是 | CSV 输出根目录 |
| `filename` | `str` (keyword-only) | 是 | CSV 文件名（如 `"table4_format_x_transform.csv"`） |

**输出文件**: `<output_dir>/tables/<filename>` (CSV 格式: `Format,<sorted transform names>`)

**缺失值处理**: 当某个 format+transform 组合在 results 中不存在时，显示为 `"N/A"`（文本表格中）和 `"N/A"`（CSV 中）。`_get_acc_val` 的 `nan` 返回值在此处被识别。

#### `transform_distribution_table(results, *, output_dir) -> str`

逐层最优 transform 选择分布表 (Table 5)。对每个 format，统计各 transform 被选为逐层最优的层数。

```python
def transform_distribution_table(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    output_dir: str,
) -> str:
    ...
```

**输入参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `results` | `Dict[str, dict]` | 是 | Part D results（format → transform → result） |
| `output_dir` | `str` (keyword-only) | 是 | CSV 输出根目录 |

**算法**: 对每个 format，通过 `_compute_best_transform_per_layer` 确定每层的最佳 transform（基于 `qsnr_per_layer`），然后统计各 transform 被选中的层数。

**输出文件**: `<output_dir>/tables/table5_transform_distribution.csv` (`Format,<sorted tx names>,Total`)

#### `layer_sensitivity_table(results, *, output_dir) -> str`

全实验中最敏感的 Top-10 层排名 (Table 6)。聚合所有 part 中的全部 QSNR/MSE 数据，按平均 MSE 降序排列取 Top 10。

```python
def layer_sensitivity_table(
    all_results: Dict[str, Dict[str, Any]],
    *,
    output_dir: str,
) -> str:
    ...
```

**输入参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `all_results` | `Dict[str, dict]` | 是 | 完整 results dict（含 `part_a`, `part_b`, 等顶层键） |
| `output_dir` | `str` (keyword-only) | 是 | CSV 输出根目录 |

**数据聚合规则**: 仅处理键以 `"part_"` 开头的顶层 entry。对每个 part 内的每个 config entry，提取 `mse_per_layer` 和 `qsnr_per_layer`，按层名聚合均值。非 `part_*` 的 entry（如 `"FP32 (baseline)"`, `"block_sweep"`）为兼容性考虑不应引发错误，但不会进入聚合。

**输出文件**: `<output_dir>/tables/table6_sensitivity.csv` (CSV 格式: `Rank,Layer,Avg_MSE,Avg_QSNR_dB`)

---

### Module 4: `src/viz/figures.py` — Figure Functions

#### 公共签名约定

每个 figure 函数：
1. 接收 results dict + 可选关键字参数（`colors`, `output_dir`）
2. 不返回 figure 对象（内部 save + close）—— 区别于纯库函数，此处的"无返回值"等价于"图片已写入文件系统"
3. 通过 `save_figure`（`save.py`）保存到 `<output_dir>/figures/`
4. 遇到缺失/空数据时绘制占位文本图（错误容忍模式），而非抛出异常

#### 输入数据结构

所有 figure 函数接受的 results dict 遵循以下结构（来自 `run_experiment` 的返回值）：

```python
{
    "MXINT-8": {
        "accuracy": {"accuracy": 0.82},
        "fp32_accuracy": {"accuracy": 0.85},
        "delta": {"accuracy": -0.03},
        "qsnr_per_layer": {"0.linear": 42.99, "1.linear": 38.52, ...},
        "mse_per_layer": {"0.linear": 0.0012, "1.linear": 0.0034, ...},
        "report": Report,                # AnalysisContext.report()
    },
    "INT8-PC": { ... },
    "FP32 (baseline)": {
        "accuracy": {"accuracy": 0.85},
    },
}
```

`_get_acc_val(data)` 从单个结果 dict 中提取 scalar accuracy 值（作为 `float`）：

```python
def _get_acc_val(data: Dict[str, Any]) -> float:
    # data["accuracy"] → {"accuracy": val} → extract val
    # 缺失或非 dict → float("nan")
```

`_compute_best_transform_per_layer(variant_qsnr)` 决定每层最佳 transform：

```python
def _compute_best_transform_per_layer(
    variant_qsnr: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    # 返回 {layer_name: best_transform_name}
    # ties: 第一个 transform（dict insertion order）
```

#### 图函数列表

##### `qsnr_bar_chart(results, *, title, colors, output_dir) -> Figure`

Per-layer QSNR 折线图。对每个非 baseline 配置绘制一条线：x = 层索引（按层名排序），y = QSNR (dB)。

```python
def qsnr_bar_chart(
    results: Dict[str, Dict[str, Any]],
    *,
    title: str,
    colors: Dict[str, str],
    output_dir: str,
) -> matplotlib.figure.Figure:
    ...
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `results` | `Dict[str, dict]` | 配置名 → 结果 dict（如 `part_a`） |
| `title` | `str` | 图片标题（set_title） |
| `colors` | `Dict[str, str]` | 配置名 → hex color（使用 FORMAT_COLORS） |
| `output_dir` | `str` | 输出根目录 |

**绘制细节**:
- `figsize=(12, 6)`, `marker="o"`, `linewidth=2`
- baseline entry（键含 `"baseline"` 大小写不敏感）被跳过
- 配置名不在 `colors` 中时使用 `FALLBACK_CYCLE[0]`
- 层名缺失时跳过该层（`None` 值逻辑不产生异常）
- x 轴标签为 "Layer Index"，y 轴为 "QSNR (dB)"

**输出文件**: `figures/fig<N>_qsnr_<detail>.png` + `.pdf`（调用时决定 name）

##### `mse_box_plot(results, *, title, colors, output_dir) -> Figure`

Per-layer MSE 箱线图。对每个非 baseline 配置绘制一个箱线图分布。

```python
def mse_box_plot(
    results: Dict[str, Dict[str, Any]],
    *,
    title: str,
    colors: Dict[str, str],
    output_dir: str,
) -> matplotlib.figure.Figure:
    ...
```

**绘制细节**:
- `figsize=(10, 6)`, `patch_artist=True`, `alpha=0.6`
- y 轴为对数尺度（`set_yscale("log")`）
- baseline entry 被跳过
- 缺失/空的 `mse_per_layer` → 不对该配置绘制箱线图

##### `pot_delta_bar(part_c, *, output_dir) -> Figure`

PoT vs FP32 QSNR delta 柱状图 (Fig 5)。对每个 format（INT8-PC, INT4-PC）绘制一个子图，每层一个柱：`delta = QSNR_PoT - QSNR_FP32`。

```python
def pot_delta_bar(
    part_c: Dict[str, Dict[str, Any]],
    *,
    output_dir: str,
) -> matplotlib.figure.Figure:
    ...
```

**绘制细节**:
- 按 format base 分组（键 `rsplit("-", 1)[0]`，如 `"INT8-PC-FP32"` → `"INT8-PC"`）
- 每组绘制独立子图：`figsize=(7 * n_groups, 5)`
- 柱颜色：正 delta `#2ecc71` (绿色)，负 delta `#e74c3c` (红色)，`alpha=0.7`
- x 轴标签为简化后的层名（去除 `"module."` 和 `"Quantized"` 前缀）
- y 轴为 "QSNR Delta (PoT – FP32) [dB]"

##### `histogram_overlay(all_results, *, output_dir) -> Figure`

三通道直方图叠加 (Fig 6)。提取 `HistogramObserver` 的数据（`fp32_hist`, `quant_hist`, `err_hist`），选择数据最丰富的 5 层绘制叠加图。

```python
def histogram_overlay(
    all_results: Dict[str, Dict[str, Any]],
    *,
    output_dir: str,
) -> matplotlib.figure.Figure:
    ...
```

**数据提取**: 遍历 `all_results` 中所有 `part_*` 条目下的所有 config，从 `report._raw` 中查找含 `fp32_hist` 和 `quant_hist` 的 metrics。

**层选择**: 按 `fp32_hist` 的总和降序排列，取前 5 层。

**绘制细节**:
- `n_bins = 128` (直方图的 bin 数由 `HistogramObserver` 决定，此处仅用于可视化)
- 三个通道：fp32 (`#3498db`), quant (`#e74c3c`), error (`#95a5a6`)
- `fill_between(..., alpha=0.35, step="mid")` + `plot(linewidth=0.8)`
- 五层用 5 个子图并排：`figsize=(5 * n, 4)`
- 无数据时显示占位文本（不抛出异常）

##### `transform_heatmap(part_d, *, colors, output_dir) -> Figure`

Format x Transform accuracy 热力图 (Fig 7)。使用 `_get_acc_val` 提取每个单元格的值。

```python
def transform_heatmap(
    part_d: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    colors: Dict[str, str],    # 此处 colors 参数预留但未被热力图使用（用 colormap）
    output_dir: str,
) -> matplotlib.figure.Figure:
    ...
```

**绘制细节**:
- `figsize=(10, 6)`, 色图 `RdYlGn`
- 缺失值设为 `nan`，用灰色（`#d3d3d3`）显示（`cmap.set_bad`)
- 单元格内显示数值文本，`fontsize=9`
- 文本颜色自动取反：值在色图中位数以下用白色，以上用黑色
- 列名旋转 45 度

##### `transform_pie(part_d, *, colors, output_dir) -> Figure`

逐层最优 transform 选择分布饼图 (Fig 8)。每个 format 一个子图。

```python
def transform_pie(
    part_d: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    colors: Dict[str, str],   # TRANSFORM_COLORS
    output_dir: str,
) -> matplotlib.figure.Figure:
    ...
```

**绘制细节**:
- 每个 format 一个饼图：`figsize=(5 * n_fmts, 5)`，`subplot_kw={"aspect": "equal"}`
- 颜色来自 `TRANSFORM_COLORS`（未知 transform 用 `#95a5a6`）
- 显示百分比（`autopct="%1.0f%%"`），标题含层数 `(n=total)`
- 当 `PerLayerOpt` 数据缺失时显示 "No PerLayerOpt data" 占位文本

##### `transform_delta(part_d, *, colors, output_dir) -> Figure`

Transform QSNR delta vs baseline 柱状图 (Fig 9)。每个 format 一个子图，每个 transform 一组柱。

```python
def transform_delta(
    part_d: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    colors: Dict[str, str],   # TRANSFORM_COLORS
    output_dir: str,
) -> matplotlib.figure.Figure:
    ...
```

**绘制细节**:
- 每个 format 占一行子图：`figsize=(14, 4 * n_fmts)`
- 对 "SmoothQuant" 和 "Hadamard" 分别计算 `delta = qsnr_tx - qsnr_baseline`
- 每个 transform 一组柱，组间间距 2
- `alpha=0.6`，柱顶标注层名（`fontsize=4, rotation=90`）
- 当层数 ≤ 20 时标注层名，否则不标注

##### `error_vs_distribution(all_results, *, output_dir) -> Figure`

4-panel 散点图矩阵 (Fig 10)。QSNR vs 分布特征。

```python
def error_vs_distribution(
    all_results: Dict[str, Dict[str, Any]],
    *,
    output_dir: str,
) -> matplotlib.figure.Figure:
    ...
```

**数据提取**: 从 `all_results` 中提取含 `qsnr_db` 和 `dynamic_range_bits` 的 metrics。

**四面板**:
1. QSNR vs Dynamic Range (颜色 = sparse_ratio, cmap="viridis")
2. QSNR vs Skewness (颜色 = kurtosis, cmap="plasma")
3. MSE (dB) vs Dynamic Range (颜色 = `#e74c3c`)
4. Sparsity 直方图 (20 bins)

##### `layer_type_qsnr(all_results, *, output_dir) -> Figure`

Layer-type 分组 QSNR 对比 (Fig 11)。使用 `LayerSensitivity.by_layer_type()` 聚合。

```python
def layer_type_qsnr(
    all_results: Dict[str, Dict[str, Any]],
    *,
    output_dir: str,
) -> matplotlib.figure.Figure:
    ...
```

**绘制细节**:
- 两面板并排：QSNR 箱线图 + MSE 箱线图
- `figsize=(14, 5)`
- MSE 面板 y 轴对数尺度
- 颜色来自 `FALLBACK_CYCLE`
- 无数据时显示占位文本

---

### 输入数据结构（完整参考）

所有 table 和 figure 函数共用的 `results` dict 结构：

```python
# Part A/B 返回: Dict[str, Dict[str, Any]]
{
    "MXINT-8": {                    # 配置名（str）
        "accuracy": {"accuracy": 0.82, ...},   # 量化模型精度（Dict[str, float]）
        "fp32_accuracy": {"accuracy": 0.85},   # FP32 参照精度（Dict[str, float]）
        "delta": {"accuracy": -0.03},          # 精度差值（quant - fp32）
        "report": Report,                      # AnalysisContext.report()
        "qsnr_per_layer": {"layer0": 42.5},    # 逐层平均 QSNR (dB)
        "mse_per_layer": {"layer0": 0.001},    # 逐层平均 MSE
    },
    "FP32 (baseline)": {            # baseline 仅含 accuracy
        "accuracy": {"accuracy": 0.85},
    },
}

# Part D 返回: Dict[str, Dict[str, Dict[str, Any]]]
{
    "MXINT-4": {                    # format 名
        "None": { ... },            # transform 名 → 结果 dict（同上）
        "SmoothQuant": { ... },
        "Hadamard": { ... },
        "PerLayerOpt": { ... },
    },
}

# All results: Dict[str, Any]
{
    "part_a": { ... },              # 键以 "part_" 开头以供 layer_sensitivity 识别
    "part_b": { ... },
    "part_c": { ... },
    "part_d": { ... },
    "FP32 (baseline)": { ... },     # 非 part_* 兼容性
}
```

---

## 期望行为

### 场景 1: Part A results → 8-bit QSNR chart + accuracy table

```python
part_a = {
    "MXINT-8": {"accuracy": {"accuracy": 0.82}, "qsnr_per_layer": {"l0": 42.5, "l1": 38.2}, ...},
    "MXFP-8":  {"accuracy": {"accuracy": 0.79}, "qsnr_per_layer": {"l0": 40.1, "l1": 36.8}, ...},
    "INT8-PC": {"accuracy": {"accuracy": 0.84}, "qsnr_per_layer": {"l0": 45.2, "l1": 41.3}, ...},
    "FP32 (baseline)": {"accuracy": {"accuracy": 0.85}},
}

# qsnr_bar_chart: 3 条折线 (baseline 跳过)，颜色来自 FORMAT_COLORS
qsnr_bar_chart(part_a, title="Fig 1: Per-Layer QSNR — 8-bit Formats",
               colors=FORMAT_COLORS, output_dir="/tmp/viz_test")

# accuracy_table: 4 rows (含 baseline), CSV + 文本
accuracy_table(part_a, title="Table 1: 8-bit Format Comparison",
               output_dir="/tmp/viz_test", filename="table1_8bit.csv")
```

期望：
- Fig 1: 3 条线，x = [0, 1]（层数），图例显示 MXINT-8/MXFP-8/INT8-PC
- Table 1: 4 行（含 FP32 (baseline)），avg QSNR 为各层平均，avg MSE 同样
- CSV 文件 `<output_dir>/tables/table1_8bit.csv` 存在且格式正确

### 场景 2: Part D results → transform 热力图 + 分布表 + delta

```python
part_d = {
    "MXINT-4": {
        "None": {"accuracy": {"accuracy": 0.75}, "qsnr_per_layer": {"l0": 35.0, ...}},
        "SmoothQuant": {"accuracy": {"accuracy": 0.78}, "qsnr_per_layer": {"l0": 37.2, ...}},
        "Hadamard": {"accuracy": {"accuracy": 0.76}, "qsnr_per_layer": {"l0": 36.1, ...}},
        "PerLayerOpt": {"accuracy": {"accuracy": 0.79}, "qsnr_per_layer": {"l0": 38.0, ...}},
    },
}

# _compute_best_transform_per_layer: 每层最优 transform
# transform_heatmap: 4 rows × 4 columns (含 PerLayerOpt)
# transform_distribution_table: 各 transform 被选中层数
# transform_pie: 1 pie chart per format
# transform_delta: 1 subplot per format
```

期望：
- Heatmap: 单元格 = accuracy 值，PerLayerOpt 应为最高（或接近最高）
- Distribution: Σ(counts) = 总层数
- Pie: 百分比总和为 100%
- Delta: SmoothQuant/Hadamard 柱的正负反映 QSNR 提升或下降

### 场景 3: 全 results → histogram overlay + error vs distribution + layer type

```python
all_results = {"part_a": part_a, "part_b": part_b, "part_c": part_c, "part_d": part_d}
histogram_overlay(all_results, output_dir="/tmp/viz_test")
error_vs_distribution(all_results, output_dir="/tmp/viz_test")
layer_type_qsnr(all_results, output_dir="/tmp/viz_test")
```

期望：
- Histogram: 5 层、每层 3 通道叠加
- Error vs distribution: 4 面板（或占位文本），数据点从 `report._raw` 提取
- Layer type: 2 面板（QSNR + MSE boxplot）

### 场景 4: 空/缺失数据 → 占位文本（不抛异常）

```python
histogram_overlay({"part_a": {}}, output_dir="/tmp/viz_test")
```

期望：
- 显示 "Histogram data not available" 占位文本图
- 无 `KeyError` 或 `AttributeError`
- 图片文件正常保存

### 场景 5: Part C results → pot_scaling_table + pot_delta_bar

```python
part_c = {
    "INT8-PC-FP32": {"accuracy": {"accuracy": 0.84}, "qsnr_per_layer": {"l0": 44.0, ...}},
    "INT8-PC-PoT":  {"accuracy": {"accuracy": 0.83}, "qsnr_per_layer": {"l0": 43.5, ...}},
    "INT4-PC-FP32": {"accuracy": {"accuracy": 0.72}, "qsnr_per_layer": {"l0": 32.0, ...}},
    "INT4-PC-PoT":  {"accuracy": {"accuracy": 0.71}, "qsnr_per_layer": {"l0": 31.5, ...}},
    "FP32 (baseline)": {"accuracy": {"accuracy": 0.85}},
}
```

期望：
- Table 3: 4 行（不含 baseline），delta 列 = accuracy - 0.85
- CSV 文件存在，Delta 精度不低于小数点后 6 位
- delta bar: per-layer delta 柱，INT8-PC 和 INT4-PC 各一组子图

---

## 异常与守卫

| 场景 | 期望行为 |
|------|---------|
| results dict 空 | 文本表格显示空/无数据行；figure 显示占位文本 |
| results dict 含非标准键（如 `"metadata"`） | 静默忽略非标准键（仅 `part_*` 被聚合函数处理，非 `part_*` 不被纳入但不抛异常） |
| baseline 不存在（table 函数） | delta 为 `0.0`，正常打印 |
| `qsnr_per_layer` / `mse_per_layer` 为空 dict | avg 为 `0.0`（除以 1 的保护处理） |
| `report._raw` 结构不符合预期 | `histogram_overlay` 和 `error_vs_distribution` 显示占位文本 |
| `fig.savefig` 目标目录不存在 | `save_figure` 自动创建目录 |
| `output_dir` 为 `None` | `save_figure` 中 `os.path.join` 抛出 `TypeError` |
| `colors` dict 缺失某些配置名 | 使用 `FALLBACK_CYCLE[0]` 作为默认颜色 |
| `results` 中某个 entry 不是 dict | `_get_acc_val` 返回 `float("nan")`，table 显示处理 |
| `all_results` 中无 `part_*` 键 | `layer_sensitivity_table` 返回空排名（Top-0），无错误 |
| `transform_pie` 中某个 format 无 `PerLayerOpt` | 显示 "No PerLayerOpt data" 占位文本 |
| `histogram_overlay` 中所有层均无直方图数据 | 显示 "Histogram data not available" 占位文本 |
| `error_vs_distribution` 中无分布数据 | 显示 "Distribution data not available" 占位文本 |

---

## 共享辅助函数合约

#### `_compute_best_transform_per_layer(variant_qsnr) -> Dict[str, str]`

```python
def _compute_best_transform_per_layer(
    variant_qsnr: Dict[str, Dict[str, float]],   # {transform_name: {layer_name: qsnr}}
) -> Dict[str, str]:                               # {layer_name: best_transform_name}
```

- **输入**: 每个 transform 的逐层 QSNR 值。
- **输出**: 每层名称 → 该层 QSNR 最高的 transform 名称。
- **平局规则**: 平局时选第一个 transform（dict insertion order）。
- **缺失层**: 如果一个 layer 只出现在部分 transform 中，只在出现的 transform 中比较（不要求所有 transform 都有该层）。
- **全缺失**: 如果某个 transform 的 dict 为空，不参与 max 比较（`max` 空序列不会发生，因为 `tx_names` 中至少有一个非空）。

#### `_get_acc_val(data) -> float`

```python
def _get_acc_val(data: Dict[str, Any]) -> float:
```

- **输入**: 单个配置的结果 dict（`data["accuracy"]` 应为 dict 内含 scalar accuracy 或直接为 scalar）。
- **输出**: scalar accuracy 值（`float`）。
- **缺失处理**: `data` 为空/非 dict 时返回 `float("nan")`；`accuracy` 缺失时返回 `float("nan")`；`accuracy` 为 dict 但找不到 `"accuracy"` 键时同样返回 `float("nan")`。

---

## 验证结果

- [ ] 运行日期: YYYY-MM-DD
- [ ] 结果: PASS / FAIL
- [ ] 说明
