# Format Study 缺陷修复计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 format study 实验系统的 18 个缺陷（3 Critical + 6 Major + 9 Minor），覆盖实验层面、可视化、表格、测试覆盖四个维度。

**Architecture:** 按文件耦合度分 9 个 task，每个 task 自包含（修改 1-3 个文件 + 对应测试）。优先修复会静默数据丢失或运行时崩溃的 Critical/Major 缺陷，再处理 Minor 改善项。V6（layer_type_qsnr MLP退化）作为遗留问题记录，不纳入本次修复。

**Tech Stack:** Python 3.10+, PyTorch, matplotlib, pytest

---

## 遗留问题

- **V6 — `layer_type_qsnr` 对 MLP-only 模型退化**：当所有层都是 Linear 时，`LayerSensitivity.by_layer_type()` 返回单类别，图表变成两个单箱图。应在修复完成后与用户讨论降级策略（按层深度分组 / 按参数形状分组 / 自动检测并跳过）。

---

### Task 1: 恢复 pipeline refactor 中丢失的 3 个辅助函数（D0 Critical）

**问题**: `_make_smoothquant_transforms`、`_fuse_smoothquant_weights`、`_build_per_layer_optimal_cfg` 三个函数在 pipeline refactor（commit `03d6ff7`）时从 `examples/experiment_format_study.py` 移到了 `src/pipeline/format_study.py`，但实际上**只移了调用方，函数定义本身丢失了**。`_run_transform_part` 会在运行时抛出 `NameError`。

**函数来源**: git commit `dc72075` 中的 `examples/experiment_format_study.py`（已删除文件）。

**Files:**
- Modify: `src/pipeline/format_study.py`（插入 3 个函数 + 必要的 import 补充）
- Create: `src/tests/test_format_study_helpers.py`（3 个函数的单元测试）

**Step 1: 从 git 历史恢复函数定义**

三个函数完整源码见 git commit `dc72075`:
- `_make_smoothquant_transforms` — 约 70 行，hook 注册 + SmoothQuantTransform.from_calibration
- `_fuse_smoothquant_weights` — 约 40 行，W = W * s 权重融合
- `_build_per_layer_optimal_cfg` — 约 45 行，per-layer 最优变换组合 cfg

插入位置：在 `_run_transform_part` 函数之前（`format_study.py` line 318 之前），与现有私有函数放在一起。

需要的 import 补充：无需新增，`SmoothQuantTransform` 已在 line 38 导入，`HadamardTransform` 在 line 37，`_compute_best_transform_per_layer` 在 line 56。

**Step 2: 适配现有代码路径**

检查恢复的函数与当前 `_run_transform_part` 的调用方式一致：
- `_make_smoothquant_transforms(fp32_model, calib_data)` → `Dict[str, TransformBase]`
- `_fuse_smoothquant_weights(fp32_model, sq_transforms, layer_names=...)` → `nn.Module`
- `_build_per_layer_optimal_cfg(variant_results, sq_transforms, fmt_str, gran, builder, weight_only)` → `Dict[str, OpQuantConfig]`

当前调用点：`format_study.py:331-332`, `390`, `393`

**Step 3: 为 3 个函数编写测试**

```bash
# 创建 src/tests/test_format_study_helpers.py
```

测试内容：
- `test_make_smoothquant_transforms_produces_per_layer_dict` — 在 ToyMLP 上生成 transforms，验证返回类型和 key 非空
- `test_make_smoothquant_transforms_none_model_returns_empty` — fp32_model=None → {}
- `test_fuse_smoothquant_weights_does_not_mutate_original` — 原始模型 weight 不变
- `test_fuse_smoothquant_weights_layer_names_filter` — layer_names 参数过滤
- `test_fuse_smoothquant_weights_skip_non_smoothquant` — 遇到 IdentityTransform 跳过
- `test_build_per_layer_optimal_cfg_selects_best_transform` — 验证选择 QSNR 最高的 transform
- `test_build_per_layer_optimal_cfg_weight_only` — weight_only=True 时仅 weight 非 None

**Step 4: 运行测试验证**

```bash
pytest src/tests/test_format_study_helpers.py -v
# Expected: 7 passed

pytest src/tests/ -q --ignore=src/tests/test_golden_equiv.py
# Expected: 1423 passed (was 1416 + 7 new)
```

**Step 5: Commit**

```bash
git add src/pipeline/format_study.py src/tests/test_format_study_helpers.py
git commit -m "fix(pipeline): restore 3 helper functions lost during pipeline refactor

_recover from git history (dc72075): _make_smoothquant_transforms,
_fuse_smoothquant_weights, _build_per_layer_optimal_cfg were
referenced in _run_transform_part but never migrated from
examples/ to src/pipeline/. Add 7 unit tests."
```

---

### Task 2: 修复 JSON 保存过滤 + 增量保存 + plot_from_results 缺失（D1 Critical + D4 Critical）

**问题**: 
- D1: `_save_results_json` 的 `if not part_name.startswith("part_")` 过滤掉了 `block_sweep`
- D4: 整个 study 只在最后保存一次，中间崩溃全部丢失
- `plot_from_results` 缺少 `block_sweep` 和 `part_hierarchical` 的重生成路径

**Files:**
- Modify: `src/pipeline/format_study.py`（`_save_results_json`, `run_format_study`, `plot_from_results`）
- Modify: `src/tests/test_pipeline_integration.py`（增量保存 + plot_from_results 测试）

**Step 1: 修复 `_save_results_json`（line 909-924）**

移除 `part_name.startswith("part_")` 的前缀过滤，改为跳过非 dict 值和 `FP32 (baseline)` 占位符：

```python
def _save_results_json(all_results: dict, output_dir: str):
    serializable: Dict[str, dict] = {}
    for part_name, part_data in all_results.items():
        if not isinstance(part_data, dict):
            continue
        serializable[part_name] = {}
        for cfg_name, cfg_data in part_data.items():
            if isinstance(cfg_data, dict):
                entry: Dict = {}
                for key in ("accuracy", "qsnr_per_layer", "mse_per_layer"):
                    if key in cfg_data:
                        entry[key] = cfg_data[key]
                if entry:
                    serializable[part_name][cfg_name] = entry
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print("  results.json: saved")
```

**Step 2: 添加增量保存**

在 `run_format_study` 的每个 part 完成后立即调用 `_save_results_json`：

```python
# 在 line 815 之后（每个 part runner 返回后）插入：
all_results[part_key] = runner(part_cfg, fp32_model, calib_data, eval_loader, eval_fn=eval_fn)
_save_results_json(all_results, output_dir)  # 增量保存

# 同样在 Part D Conv（line 835 后）插入增量保存
```

**Step 3: 补充 `plot_from_results` 的缺失 part**

在 `plot_from_results`（line 858-881）中添加 `block_sweep` 和 `part_hierarchical` 的重生成路径：

```python
# block_sweep 重生成
if "block_sweep" in all_results:
    print(accuracy_table(all_results["block_sweep"], 
           title="Block Size Sweep Results", output_dir=output_dir, 
           filename="block_sweep.csv"))

# part_hierarchical 重生成
if "part_hierarchical" in all_results:
    print(accuracy_table(all_results["part_hierarchical"],
           title="Hierarchical Pre-Scale Results", output_dir=output_dir,
           filename="hierarchical.csv"))
```

**Step 4: 编写测试**

在 `src/tests/test_pipeline_integration.py` 添加：

```python
def test_save_results_json_includes_block_sweep(self, tmp_path):
    """block_sweep key (not starting with 'part_') must be saved."""
    from src.pipeline.format_study import _save_results_json
    results = {"block_sweep": {"int8-blk32": {"accuracy": 0.85}}}
    _save_results_json(results, str(tmp_path))
    saved = json.load(open(tmp_path / "results.json"))
    assert "block_sweep" in saved

def test_plot_from_results_handles_block_sweep(self, tmp_path):
    """plot_from_results should not crash on block_sweep data."""
    ...

def test_plot_from_results_handles_hierarchical(self, tmp_path):
    """plot_from_results should not crash on part_hierarchical data."""
    ...
```

**Step 5: 运行测试**

```bash
pytest src/tests/test_pipeline_integration.py -v
# Expected: 新增 3 passed
```

**Step 6: Commit**

```bash
git add src/pipeline/format_study.py src/tests/test_pipeline_integration.py
git commit -m "fix(pipeline): include block_sweep in results.json, add incremental save

D1: Remove part_ prefix filter in _save_results_json so block_sweep is saved.
D4: Save results.json incrementally after each part completes.
Also extend plot_from_results to handle block_sweep and part_hierarchical."
```

---

### Task 3: 修复 histogram_overlay 层选择逻辑（V2 Critical）

**问题**: `histogram_overlay`（`src/viz/figures.py:279-282`）用 `fp32_hist.sum()` 排序选层——选的是激活幅度最大的层，而非对量化最敏感的层。应该按敏感度（低 QSNR / 高 MSE）选取。

**Files:**
- Modify: `src/viz/figures.py:278-285`（修改排序逻辑）
- Modify: `src/tests/test_viz_figures.py`（更新 TestHistogramOverlay 测试）

**Step 1: 修改 `histogram_overlay` 的 top_layers 排序**

在收集 `layer_hists` 的同时，收集对应的 QSNR/MSE 数据用于排序：

```python
# 在收集 layer_hists 的循环中同时收集 layer_error
layer_error: Dict[str, float] = {}
# ... 在现有循环中，同时记录：
if "qsnr_db" in metrics:
    layer_error[layer] = metrics["qsnr_db"]  # 或收集 mse

# 然后按 QSNR 升序（最差在前）排序
top_layers = sorted(
    layer_hists.items(),
    key=lambda x: layer_error.get(x[0], float("inf")),  # 低QSNR=高敏感度
)[:5]
```

如果 histogram 数据中不包含 QSNR/MSE（histogram observer 只收集 bin counts），则需要额外遍历 report 的其他 observer 数据获取敏感度指标。优先使用 MSE observer 的数据。

**Step 2: 如果没有 QSNR/MSE 数据，回退到 histogram 计数排序**

保持现有逻辑作为 fallback，但添加 warning 日志：

```python
if not layer_error:
    print("  Warning: No QSNR/MSE data for sensitivity ranking, "
          "falling back to histogram magnitude")
    top_layers = sorted(
        layer_hists.items(),
        key=lambda x: x[1].get("fp32_hist", np.array(0)).sum(),
        reverse=True,
    )[:5]
```

**Step 3: 更新测试**

```python
class TestHistogramOverlay:
    def test_ranks_by_sensitivity_not_magnitude(self):
        """Verify layers with worse QSNR are selected, not larger activations."""
        ...
```

**Step 4: 运行测试**

```bash
pytest src/tests/test_viz_figures.py::TestHistogramOverlay -v
# Expected: PASS
```

**Step 5: Commit**

```bash
git add src/viz/figures.py src/tests/test_viz_figures.py
git commit -m "fix(viz): rank histogram overlay layers by sensitivity, not activation magnitude

V2: histogram_overlay was selecting layers with largest fp32_hist.sum()
(activation magnitude), which picks loud layers, not sensitive ones.
Now uses QSNR/MSE to find layers most affected by quantization."
```

---

### Task 4: 添加 Report 公共 API 消除私有属性访问（V4 Major）

**问题**: `histogram_overlay` 和 `error_vs_distribution` 直接访问 `report._raw`（私有属性），违反模块边界且脆弱。

**Files:**
- Modify: `src/analysis/report.py`（添加 2 个公共方法）
- Modify: `src/viz/figures.py:241-264, 576-602`（替换 `_raw` 访问）
- Modify: `src/tests/test_viz_figures.py`（适配新 API）

**Step 1: 在 Report 类添加 `iter_slices()` 公共方法**

```python
def iter_slices(self):
    """Yield (layer, role, stage, slice_key, metrics) for every slice."""
    for layer, roles in self._raw.items():
        for role, stages in roles.items():
            for stage, slices in stages.items():
                for slice_key, metrics in slices.items():
                    yield layer, role, stage, slice_key, metrics
```

**Step 2: 重构 `histogram_overlay` 的数据收集逻辑**

用 `report.iter_slices()` 替换 `report._raw` 遍历：

```python
for layer, role, stage, slice_key, metrics in report.iter_slices():
    if layer not in layer_hists and "fp32_hist" in metrics and "quant_hist" in metrics:
        layer_hists[layer] = {...}
    if "qsnr_db" in metrics and layer not in layer_error:
        layer_error[layer] = metrics["qsnr_db"]
```

**Step 3: 重构 `error_vs_distribution` 的数据收集逻辑**

同样用 `report.iter_slices()` 替换 `report._raw` 遍历。

**Step 4: 运行测试**

```bash
pytest src/tests/test_viz_figures.py -v
pytest src/tests/ -q --ignore=src/tests/test_golden_equiv.py
# Expected: all pass, no regression
```

**Step 5: Commit**

```bash
git add src/analysis/report.py src/viz/figures.py src/tests/test_viz_figures.py
git commit -m "refactor(analysis): add Report.iter_slices() public API to replace _raw access

V4: histogram_overlay and error_vs_distribution were accessing report._raw
(private attribute). Add iter_slices() generator to Report for stable
public iteration over all analysis slices."
```

---

### Task 5: 内存泄漏 + QSNR 对齐 + transform_delta 标签 + baseline 检测（V3 + V1 + V7 + T1）

**问题**:
- V3: 所有 viz 函数创建 figure 后不 `plt.close()`，内存泄漏
- V1: `qsnr_line_chart` 按 index 对齐不同 config 的 layer
- V7: `transform_delta` 超过 20 层时不显示任何标签
- T1: `generate_table_3` 用 `"baseline" in name.lower()` 检测

**Files:**
- Modify: `src/viz/figures.py`（4 处修复）
- Modify: `src/viz/save.py`（plt.close）
- Modify: `src/pipeline/format_study.py`（T1 修复）
- Modify: `src/tests/test_viz_figures.py`（更新测试）

**Step 1: 修复内存泄漏（V3）**

在 `save_figure` 函数末尾添加 `plt.close(fig)`：

```python
# src/viz/save.py line 21 后
plt.close(fig)
return os.path.join(fig_dir, f"{name}.png")
```

同时从 docstring 移除 "The caller is responsible for closing" 的说明。

**Step 2: 修复 QSNR 折线图对齐（V1）**

```python
# src/viz/figures.py — qsnr_line_chart
# 收集所有 config 共享的 layer 名作为 x 轴
all_layers = sorted(set().union(*[
    data["qsnr_per_layer"].keys()
    for name, data in results.items()
    if "baseline" not in name.lower() and "qsnr_per_layer" in data
]))
if not all_layers:
    # fallback to empty plot
    ...

# 使用 layer 名作为 x 轴类别标签
x = range(len(all_layers))
for name, data in results.items():
    ...
    values = [data["qsnr_per_layer"].get(l, float("nan")) for l in all_layers]
    ax.plot(x, values, ...)

ax.set_xticks(x)
ax.set_xticklabels([l.replace("module.", "").replace("Quantized", "")[:15]
                     for l in all_layers], rotation=45, ha="right", fontsize=8)
```

**Step 3: 修复 transform_delta 标签自适应（V7）**

```python
# 在 transform_delta 中，替换现有的 ≤20 固定阈值逻辑：
num_layers = len(all_layers)
if num_layers <= 10:
    # 显示所有 layer 名
    for i, layer in enumerate(all_layers):
        ax.text(...)
elif num_layers <= 30:
    # 每 3 个显示一个
    for i, layer in enumerate(all_layers):
        if i % 3 == 0:
            ax.text(...)
else:
    # 只显示 top-5 delta 绝对值最大的 layer 标签
    top_indices = sorted(range(len(deltas)), 
                        key=lambda i: abs(deltas[i]), reverse=True)[:5]
    for i in top_indices:
        ax.text(...)
```

**Step 4: 修复 Table 3 baseline 检测（T1）**

```python
# format_study.py line 580
# 改为精确匹配：
if name == "FP32 (baseline)":
```

**Step 5: 更新测试**

- `test_viz_figures.py::TestQSNRBarChart` — 验证 layer 对齐正确
- `test_viz_figures.py::TestTransformDelta` — 验证 >20 层时仍显示标签

**Step 6: 运行测试**

```bash
pytest src/tests/test_viz_figures.py -v
pytest src/tests/ -q --ignore=src/tests/test_golden_equiv.py
# Expected: all pass
```

**Step 7: Commit**

```bash
git add src/viz/save.py src/viz/figures.py src/pipeline/format_study.py src/tests/test_viz_figures.py
git commit -m "fix(viz): close figures, align QSNR by layer name, adaptive transform_delta labels

V3: Add plt.close(fig) in save_figure to prevent memory leak.
V1: Align qsnr_line_chart x-axis by shared layer names, not index.
V7: Adaptive label strategy in transform_delta (show all labels for <=10 layers,
     show every 3rd for <=30, show top-5 delta for >30).
T1: Exact match 'FP32 (baseline)' instead of substring 'baseline'."
```

---

### Task 6: 消除 hierarchical 代码重复 + 修复 Table 6 语义（D5 + D3）

**问题**:
- D5: `_run_hierarchical_part` 重复了 `run_experiment` 的 110 行逻辑
- D3: Table 6 将所有 bit-width 的误差混在一起平均

**Files:**
- Modify: `src/pipeline/format_study.py`（`_run_hierarchical_part`, `generate_table_6`）
- Modify: `src/tests/test_format_study_helpers.py` 或 `test_pipeline_integration.py`

**Step 1: 重构 `_run_hierarchical_part` 复用 `run_experiment`**

`_run_hierarchical_part` 唯一的独特逻辑是在 calibrate 之前调用 `session.initialize_pre_scales()`。将其改为：先创建 session → 初始化 pre-scale → 然后调用 `run_experiment` 的核心逻辑。

但 `run_experiment` 的签名设计不支持"已创建的 session"。有两种方案：

**方案 A（推荐）**：给 `run_experiment` 添加可选参数 `session: Optional[QuantSession] = None`，当提供时跳过 session 创建步骤：

```python
def run_experiment(cfg, fp32_model, calib_data, eval_loader, 
                   observers=None, *, lsq_steps=0, lsq_pot=False, 
                   lsq_lr=1e-3, eval_fn=None, session=None) -> dict:
    if session is None:
        session = QuantSession(
            copy.deepcopy(fp32_model), cfg,
            calibrator=MSEScaleStrategy(),
            keep_fp32=True,
        )
    # ... 其余逻辑不变
```

`_run_hierarchical_part` 简化为：

```python
session = QuantSession(copy.deepcopy(fp32_model), cfg, 
                       calibrator=MSEScaleStrategy(), keep_fp32=True)
session.initialize_pre_scales(calib_data, init=ps_init, pot=..., ...)
results[name] = run_experiment(
    cfg, fp32_model, calib_data, eval_loader, 
    eval_fn=eval_fn, session=session,
)
```

这样 hierarchical part 从 110 行缩减到约 20 行。

**Step 2: 修复 Table 6 敏感度排名（D3）**

方案：改为报告每个 layer 的**最差情况**误差（max MSE across all configs），而不是平均。这更能反映"该层在极端量化下有多脆弱"：

```python
# 在 generate_table_6 中，把 sum/len 改为 max
ranking = sorted(
    (
        (layer,
         max(m["mse"]) if m["mse"] else 0.0,  # 最差 MSE
         min(m["qsnr"]) if m["qsnr"] else 0.0,  # 最低 QSNR
        )
        for layer, m in layer_metrics.items()
    ),
    key=lambda x: x[1], reverse=True,
)[:10]
```

同时更新表头从 "Avg MSE" / "Avg QSNR" 为 "Max MSE" / "Min QSNR"。

**Step 3: 编写测试**

- 测试 `run_experiment` 的 `session=` 参数
- 测试 hierarchical part 产生与手动调用相同的结果
- 测试 Table 6 使用 max/min 而非 avg

**Step 4: 运行测试**

```bash
pytest src/tests/ -q --ignore=src/tests/test_golden_equiv.py
# Expected: all pass
```

**Step 5: Commit**

```bash
git add src/pipeline/format_study.py src/tests/
git commit -m "refactor(pipeline): deduplicate hierarchical logic, fix Table 6 sensitivity

D5: Add session= parameter to run_experiment so _run_hierarchical_part
    reuses the core flow instead of duplicating 110 lines.
D3: Table 6 now reports max MSE / min QSNR per layer (worst-case)
    instead of meaningless cross-bit-width average."
```

---

### Task 7: 添加 block_sweep 和 hierarchical 的可视化图表（V5 Minor）

**问题**: `_generate_figures` 没有 block_sweep 和 hierarchical 的对应图表。

**Files:**
- Modify: `src/viz/figures.py`（添加 2 个新图表函数）
- Modify: `src/pipeline/format_study.py:884-897`（在 `_generate_figures` 中注册）
- Modify: `src/tests/test_viz_figures.py`（新图表测试）

**Step 1: 添加 `block_sweep_line_chart` — 块大小 vs QSNR 折线图**

```python
def block_sweep_line_chart(
    block_sweep: dict,
    *,
    output_dir: str,
) -> plt.Figure:
    """Block size vs per-layer QSNR, one line per layer.
    
    Args:
        block_sweep: Dict mapping "int8-blk{N}" to result dict with qsnr_per_layer.
        output_dir: Output root directory.
    """
    # X 轴：block sizes, Y 轴：per-layer avg QSNR
    # 每条线代表一个 layer
    ...
    save_figure(fig, output_dir, "block_sweep_line")
    return fig
```

**Step 2: 添加 `hierarchical_delta_bar` — pre-scale vs no-pre-scale QSNR delta**

```python
def hierarchical_delta_bar(
    hierarchical: dict,
    *,
    output_dir: str,
    colors: dict | None = None,
) -> plt.Figure:
    """Pre-scale vs plain MX per-layer QSNR delta bar chart.
    
    Compares each HIER variant against its non-HIER counterpart
    (from part_a/part_b) showing the benefit of two-level quantization.
    """
    ...
    save_figure(fig, output_dir, "hierarchical_delta")
    return fig
```

**Step 3: 在 `_generate_figures` 中注册新图表**

```python
(lambda d, od: block_sweep_line_chart(d, output_dir=od), "block_sweep", "fig12"),
(lambda d, od: hierarchical_delta_bar(d, output_dir=od, colors=FORMAT_COLORS), 
 "part_hierarchical", "fig13"),
```

**Step 4: 编写测试**

```python
class TestBlockSweepLineChart:
    def test_renders_without_error(self): ...
    def test_empty(self): ...

class TestHierarchicalDeltaBar:
    def test_renders_without_error(self): ...
    def test_empty(self): ...
```

**Step 5: 运行测试**

```bash
pytest src/tests/test_viz_figures.py -v
# Expected: 新增 4 passed
```

**Step 6: Commit**

```bash
git add src/viz/figures.py src/pipeline/format_study.py src/tests/test_viz_figures.py
git commit -m "feat(viz): add block_sweep and hierarchical figures

V5: Add block_sweep_line_chart (block size vs QSNR) and 
hierarchical_delta_bar (pre-scale vs plain MX delta) to complete
the visualization coverage for all study parts."
```

---

### Task 8: 合并 _resolve_granularity + CLI 补充 + 列宽 + ToyMLP 头（D6 + E4 + T2 + E5）

**问题**:
- D6: 两个 `_resolve_granularity` 函数重复
- E4: CLI 缺少 `--skip-block-sweep` 和 `--skip-hierarchical`
- T2: `accuracy_table` 列宽固定，长名称破坏对齐
- E5: ToyMLP 无分类头，eval_fn 做 10 类 argmax 但模型输出 128 维

**Files:**
- Modify: `src/pipeline/format_study.py:561-569`（删除 `_resolve_granularity`，统一用 config.py 版本）
- Modify: `src/pipeline/config.py`（如有需要增强 `_resolve_granularity` 以覆盖所有调用点）
- Modify: `pipeline/experiment_format_study.py`（添加 CLI 参数）
- Modify: `src/viz/tables.py`（动态列宽）
- Modify: `pipeline/_model.py`（ToyMLP 添加分类头）

**Step 1: 合并 `_resolve_granularity`（D6）**

将 `format_study.py:561-569` 的 `_resolve_granularity` 替换为 import 自 config.py 的版本：

```python
# 在 format_study.py 中，已有的 import：
from src.pipeline.config import resolve_config as _resolve_config

# 再添加：
from src.pipeline.config import _resolve_granularity
```

然后删除 format_study.py 中的 `_resolve_granularity` 函数定义（line 561-569）。

确保 config.py 的版本满足所有调用方的需求（format_study.py 中的调用传递 axis 和 block_size）。

**Step 2: 添加 CLI 跳过参数（E4）**

在 `experiment_format_study.py` 的 argparse 中添加：

```python
parser.add_argument("--skip-block-sweep", action="store_true")
parser.add_argument("--skip-hierarchical", action="store_true")
```

并在 `skip` dict 中注册映射（line 82-86）：

```python
skip = {k: True for k, v in {
    "A": args.skip_part_a, "B": args.skip_part_b,
    "C": args.skip_part_c, "D": args.skip_part_d,
    "D_conv": args.skip_part_d_conv,
    "block_sweep": args.skip_block_sweep,
    "part_hierarchical": args.skip_hierarchical,
}.items() if v}
```

注意：`skip_parts` 的 key 是 study config 中的 key（`"block_sweep"` 和 `"part_hierarchical"`），不是 A/B/C/D。

**Step 3: 修复 accuracy_table 动态列宽（T2）**

```python
# src/viz/tables.py
def accuracy_table(results, *, title, output_dir, filename):
    rows = [...]
    # 计算最大列宽
    max_name = max((len(r[0]) for r in rows), default=20)
    name_w = max(max_name + 2, 20)
    
    lines.append(
        f"{'Config':<{name_w}} {'Accuracy':<20} {'Avg QSNR (dB)':<15} {'Avg MSE':<15}"
    )
    for row in rows:
        lines.append(
            f"{row[0]:<{name_w}} {row[1]:<20} {row[2]:<15.2f} {row[3]:<15.6f}"
        )
```

**Step 4: 修复 ToyMLP 添加分类头（E5）**

在 `_model.py` 中给 ToyMLP 添加 `num_classes` 参数和分类投影层：

```python
class ToyMLP(nn.Module):
    def __init__(self, hidden_size=128, intermediate_size=512, num_classes=10):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)  # 分类头

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x + residual
        x = self.head(x)  # 投影到 num_classes
        return x
```

同时更新 `experiment_format_study.py` 中 `build_model()` 的默认参数（如需要）。

**Step 5: 运行测试**

```bash
pytest src/tests/ -q --ignore=src/tests/test_golden_equiv.py
# Expected: all pass, no regression
```

**Step 6: Commit**

```bash
git add src/pipeline/format_study.py src/pipeline/config.py pipeline/experiment_format_study.py src/viz/tables.py pipeline/_model.py
git commit -m "fix: consolidate _resolve_granularity, add CLI skips, dynamic table widths, ToyMLP head

D6: Remove duplicate _resolve_granularity in format_study.py, use config.py version.
E4: Add --skip-block-sweep and --skip-hierarchical CLI flags.
T2: Dynamic column width in accuracy_table for long config names.
E5: Add num_classes Linear head to ToyMLP so eval_fn argmax is meaningful."
```

---

### Task 9: 补充测试覆盖（G1-G3 Minor）

**问题**:
- G1: viz 测试只验证"不崩溃"，不验证数据正确性
- G2: 无 `run_format_study` 端到端集成测试
- G3: 无 `plot_from_results` 测试

**Files:**
- Modify: `src/tests/test_viz_figures.py`（增强现有测试）
- Modify: `src/tests/test_pipeline_integration.py`（端到端 + plot_from_results 测试）

**Step 1: 增强 viz 测试（G1）**

至少为以下函数增加数据正确性检查：
- `test_qsnr_line_chart_data_points_match` — 验证折线上点数 = layer 数
- `test_qsnr_line_chart_aligns_by_layer_name` — 验证不同 config 在相同 x 位置表示相同 layer
- `test_transform_heatmap_cell_values` — 验证热力图数值与输入一致
- `test_transform_pie_percentages_sum_to_100` — 饼图百分比总和 100%

**Step 2: 添加 `run_format_study` 端到端测试（G2）**

```python
def test_run_format_study_end_to_end(tmp_path):
    """Minimal end-to-end with only part_a and part_b, one variant each."""
    from src.pipeline.format_study import run_format_study
    from pipeline._model import ToyMLP
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Minimal config with just 2 simple parts
    mini_config = {
        "part_a": {
            "type": "simple",
            "description": "mini 8-bit",
            "table": "table1",
            "variants": [{"name": "INT8-PT", "format": "int8", "granularity": "per_tensor"}],
        },
    }
    
    def build_model():
        return ToyMLP(hidden_size=16, intermediate_size=32, num_classes=10)
    
    def make_calib():
        return [torch.randn(4, 16)]
    
    def make_eval():
        x = torch.randn(16, 16)
        y = torch.randint(0, 10, (16,))
        return DataLoader(TensorDataset(x, y), batch_size=4)
    
    def eval_fn(model, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return {"accuracy": correct / total}
    
    results = run_format_study(
        build_model, make_calib, make_eval, eval_fn,
        output_dir=str(tmp_path / "results"),
    )
    
    assert "part_a" in results
    assert os.path.exists(tmp_path / "results" / "results.json")
    assert os.path.exists(tmp_path / "results" / "figures")
    assert os.path.exists(tmp_path / "results" / "tables")
```

**Step 3: 添加 `plot_from_results` 测试（G3）**

```python
def test_plot_from_results_regenerates_outputs(tmp_path):
    """plot_from_results should regenerate tables and figures from a saved JSON."""
    ...
```

**Step 4: 运行测试**

```bash
pytest src/tests/test_viz_figures.py src/tests/test_pipeline_integration.py -v
# Expected: all new tests pass
```

**Step 5: Commit**

```bash
git add src/tests/test_viz_figures.py src/tests/test_pipeline_integration.py
git commit -m "test: strengthen viz assertions, add run_format_study and plot_from_results tests

G1: Add data-correctness checks beyond 'doesn't crash' for key viz functions.
G2: Add minimal end-to-end test for run_format_study with a 1-part config.
G3: Add plot_from_results regeneration test."
```

---

## 执行顺序

Tasks 按依赖关系排序。Task 4 是 Task 3 的推荐前置（添加 Report.iter_slices() 后在 histogram_overlay 中使用更干净的 API），但 Task 3 也可以独立修复。

```
Task 1 (D0: 恢复丢失的函数) ── 最高优先级，transforms part 完全不能用
  │
  ├── Task 2 (D1+D4: JSON 保存 + 增量)
  │
  ├── Task 4 (V4: Report.iter_slices) ── 推荐先于 Task 3
  │     │
  │     └── Task 3 (V2: histogram 选层逻辑)
  │
  ├── Task 5 (V3+V1+V7+T1: 内存+对齐+标签+baseline)
  │
  ├── Task 6 (D5+D3: 消除重复+Table 6)
  │
  ├── Task 7 (V5: block_sweep + hierarchical 图表)
  │
  ├── Task 8 (D6+E4+T2+E5: 合并函数+CLI+列宽+模型头)
  │
  └── Task 9 (G1-G3: 测试覆盖)
```

## 验收标准

1. `_run_transform_part` 不再抛出 `NameError`（Task 1）
2. `results.json` 包含 `block_sweep` 数据（Task 2）
3. 中途崩溃后已运行的 part 结果保留在 `results.json`（Task 2）
4. `plot_from_results` 可重生成所有 part 的图表和表格（Task 2）
5. histogram 图表展示对量化最敏感的层，而非激活幅度最大的层（Task 3）
6. viz 模块不再访问 `report._raw` 私有属性（Task 4）
7. 长时间运行不泄漏 matplotlib figures（Task 5）
8. QSNR 折线图按 layer 名对齐而非 index（Task 5）
9. transform_delta 在任意层数下都至少显示 top-5 delta 标签（Task 5）
10. Table 3 只把 `"FP32 (baseline)"` 当 baseline（Task 5）
11. `_run_hierarchical_part` 不再重复 run_experiment 逻辑（Task 6）
12. Table 6 报告每层的最差情况误差，而非跨 bit-width 平均值（Task 6）
13. 有 block_sweep 和 hierarchical 的可视化图表（Task 7）
14. 只有一个 `_resolve_granularity` 函数（Task 8）
15. CLI 支持跳过 block_sweep 和 hierarchical（Task 8）
16. ToyMLP 有分类头，eval_fn 准确率有意义（Task 8）
17. viz 测试验证数据正确性，不仅是不崩溃（Task 9）
18. `pytest src/tests/ -q --ignore=src/tests/test_golden_equiv.py` 全部通过，无 regression
