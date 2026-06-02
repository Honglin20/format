# Example 19 Spec: MXInt Full Precision Diagnostic Chain

> **Status**: Draft — pending review
> **Branch**: `feature/refactor-src`
> **Depends on**: bitx `src/` (Session, Study, Observers, Intervention), AgentHarness

---

## 1. Goal

对任意 PyTorch 项目，自动运行 FP32 / W8A8 / W4A8 / W4A4 四档量化，
从粗到细做全链路精度诊断，最终输出 **MXInt 格式的缺陷定位 + 改进建议**。

核心交付物：

1. **8-Agent 串行流水线**（适配 → 量化 → 分析 → 综合）
2. **4 个新 bitx API**（PerBlockQSNRObserver、block_error_analysis、CrossConfigLayerRanking、TransformEffectReport）
3. **Block 级误差可视化**（热力图：直观看到 tensor 内哪些 block 误差最大）
4. **Agent 接口文档**（每个 agent 可调用的 API + 参数 + 返回值）

---

## 2. Agent Pipeline DAG

```
[Agent 1: adapter]           ← 适配目标项目，生成 adapter.py
       ↓
[Agent 2: study_runner]      ← 运行 8 configs (4 bit-widths × ±smoothquant)
       ↓
[Agent 3: gap_analyzer]      ← 精度差距分解（weight degradation vs activation degradation）
       ↓
[Agent 4: layer_attribution] ← 跨 config 层排序 + role 归因
       ↓ (worst_layers list)
       ├→ [Agent 5: distribution_profiler]  ──┐
       ├→ [Agent 6: block_analyst]            ──┤ parallel
       └→ [Agent 7: intervention_evaluator]   ──┘
              ↓ (all findings)
       [Agent 8: synthesis]   ← 综合所有发现，输出最终诊断报告
```

**DAG 边**：

```json
{
  "adapter":       {"after": []},
  "study_runner":  {"after": ["adapter"]},
  "gap_analyzer":  {"after": ["study_runner"]},
  "layer_attribution": {"after": ["study_runner"]},
  "distribution_profiler": {"after": ["layer_attribution"]},
  "block_analyst": {"after": ["layer_attribution"]},
  "intervention_evaluator": {"after": ["layer_attribution"]},
  "synthesis": {"after": ["distribution_profiler", "block_analyst", "intervention_evaluator"]}
}
```

---

## 3. New bitx APIs

### 3.1 PerBlockQSNRObserver

**文件**: `src/analysis/observers.py`

```python
class PerBlockQSNRObserver(SliceAwareObserver):
    """Record per-block / per-channel QSNR for fine-grained error localization.

    Unlike QSNRObserver which aggregates across all blocks,
    this observer records QSNR for each block individually.

    Output format (report()):
        {layer: {role: {stage: {("block", i): {"qsnr_db": float, "mse": float}}}}}

    For PER_CHANNEL granularity, uses _measure_per_unit() to produce
    per-channel metrics: {("channel", i): {"qsnr_db": float, "mse": float}}
    """
```

**实现要点**：
- 重写 `_measure_batch(fp32_2d, quant_2d, valid_counts)` → 逐行算 QSNR，返回 per-block dict
- PER_CHANNEL 模式下 `_measure_per_unit(fp32_2d, quant_2d)` → 逐 channel 算 QSNR
- PER_TENSOR 模式退化为单个测量
- `report()` 返回标准 observer 输出格式，与 AnalysisReport 兼容

**消费方**: Agent 6 (block_analyst), Agent 8 (synthesis)

### 3.2 block_error_analysis()

**文件**: `src/api/block_error_analysis.py`

```python
@dataclass
class BlockErrorReport:
    layer: str
    role: str
    granularity: str                    # "per_block" | "per_channel"
    per_unit_qsnr: Dict[int, float]     # unit_idx → qsnr_db
    per_unit_mse: Dict[int, float]      # unit_idx → mse
    worst_units: List[Tuple[int, float]]  # [(unit_idx, qsnr_db)] sorted worst-first
    stats: Dict[str, float]             # mean, std, min, max, p10, p90
    config_name: str

    # 可选：异常 unit 的分布统计
    outlier_unit_stats: Optional[Dict[int, dict]]  # unit_idx → {dynamic_range, outlier_ratio, kurtosis}

def block_error_analysis(
    result: SessionResult,
    layer: str,
    role: str = "weight",
    top_k: int = 10,
) -> BlockErrorReport:
    """For a given layer+role, extract per-block QSNR ranking from observer data.

    Prerequisite: Session must have been run with PerBlockQSNRObserver.
    """
```

**消费方**: Agent 6 (block_analyst)

### 3.3 CrossConfigLayerRanking

**文件**: `src/analysis/cross_config_ranking.py`

```python
class CrossConfigLayerRanking:
    """Compare which layers are consistently worst across multiple configs.

    Usage:
        ranking = CrossConfigLayerRanking.from_study(study_report)
        table = ranking.consistent_worst(k=5)
        delta = ranking.layer_qsnr_delta("fc2", from_config="W4A4", to_config="W8A8")
    """

    @classmethod
    def from_study(cls, study_report: StudyReport) -> CrossConfigLayerRanking:
        """Extract per-config per-layer QSNR from a StudyReport."""

    def consistent_worst(self, k: int = 5) -> List[Tuple[str, float]]:
        """Layers that appear in worst-k across ALL configs. Returns [(layer, avg_qsnr)]."""

    def config_specific_worst(self, config: str, k: int = 5) -> List[Tuple[str, float]]:
        """Layers that are worst only in a specific config."""

    def layer_qsnr_delta(self, layer: str, from_config: str, to_config: str) -> float:
        """QSNR improvement for a specific layer between two configs."""

    def role_dominance_cross_config(self, k: int = 5) -> List[dict]:
        """For worst-k layers, show role dominance per config.
        Returns [{layer, configs: [{config, dominant_role, qsnr}]}]"""
```

**消费方**: Agent 4 (layer_attribution), Agent 8 (synthesis)

### 3.4 TransformEffectReport

**文件**: `src/analysis/transform_effect.py`

```python
class TransformEffectReport:
    """Quantify how much each transform recovers precision, per config.

    Compares configs with/without transform (matched by w_bits, a_bits).
    """

    @classmethod
    def from_study(cls, study_report: StudyReport) -> TransformEffectReport:
        """Auto-detect transform/no-transform config pairs from StudyReport."""

    def summary(self) -> str:
        """Formatted table: config × transform → accuracy + recovery_pct."""

    def per_layer_recovery(self, config: str, transform: str = "smoothquant") -> List[dict]:
        """Per-layer QSNR improvement from transform.
        Returns [{layer, qsnr_no_transform, qsnr_with_transform, delta_db}]"""

    def per_config_recovery(self) -> List[dict]:
        """Accuracy recovery per config.
        Returns [{config, accuracy_no_transform, accuracy_with_transform, recovery_pct}]"""
```

**消费方**: Agent 3 (gap_analyzer), Agent 7 (intervention_evaluator)

---

## 4. Block Error Visualization

### 4.1 Tensor Block Error Heatmap

**文件**: `src/viz/block_error_heatmap.py`

目的：将一个 weight tensor 按 block 拆分后，用热力图展示每个 block 的误差大小，
让用户直观看到 "这块权重矩阵的哪里量化最差"。

```python
def block_error_heatmap(
    result: SessionResult,
    layer: str,
    role: str = "weight",
    *,
    top_k_blocks: int = 0,      # 0 = show all blocks; >0 = highlight top-k worst
    figsize: tuple = (12, 6),
    cmap: str = "RdYlGn",       # Red=bad, Green=good
    title: str | None = None,
) -> plt.Figure:
    """Render a 2D heatmap of per-block QSNR for a weight tensor.

    Layout:
      - X-axis: block index (0, 1, 2, ..., N_blocks)
      - Y-axis: output channel (for Linear: out_features; for Conv: out_channels)
      - Cell color: QSNR (dB) — darker red = worse quantization

    If the observer recorded per-unit (per-channel for weight) QSNR,
    render as a full 2D grid [out_channels × n_blocks].

    If only per-block aggregated QSNR is available, render as a 1D bar
    with block indices on x-axis and QSNR on y-axis.

    Top-k worst blocks are annotated with their index and QSNR value.
    """
```

**可视化示例**（终端文字描述）：

```
┌──────────────────────────────────────────────────────┐
│  Block Error Heatmap: transformer.layers.3.linear2   │
│  Role: weight | Config: W4A4                          │
│                                                       │
│  ch0  [32.1][28.5][15.2][12.8][ 8.3][22.1][30.4]...  │
│  ch1  [29.4][31.2][14.1][ 9.7][ 7.1][20.8][28.9]...  │
│  ch2  [35.2][33.8][16.4][11.2][ 3.8][25.3][32.1]...  │  ← ch2 block4 = 3.8 dB (worst!)
│  ...                                                  │
│                                                       │
│  Color: ■ <10dB  ■ 10-20dB  ■ 20-30dB  ■ >30dB      │
│         (red)    (orange)    (yellow)    (green)       │
└──────────────────────────────────────────────────────┘
```

### 4.2 Activation Channel Error Bar Chart

```python
def channel_error_bar(
    result: SessionResult,
    layer: str,
    role: str = "input",
    *,
    top_k: int = 20,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Bar chart of per-channel QSNR for activations.

    Sorted worst-first. Top-k channels highlighted in red.
    Shows which input features cause the most quantization error.
    """
```

### 4.3 Multi-Config Block Error Comparison

```python
def multi_config_block_comparison(
    study_report: StudyReport,
    layer: str,
    role: str = "weight",
    *,
    configs: List[str] | None = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Side-by-side block error bars for the same layer across configs.

    Shows how block-level error changes from W8A8 → W4A8 → W4A4.
    Identifies blocks that degrade disproportionately at lower bit-widths.
    """
```

### 4.4 render_chart 集成

所有可视化通过 `render_chart()` 发送到 AgentHarness 前端：

```python
# 在 Agent 6 (block_analyst) 的 bash 脚本中
render_chart(
    data=heatmap_data,          # [{block_idx, channel_idx, qsnr_db}]
    chart_type="heatmap",
    x="block_idx",
    y="channel_idx",
    color="qsnr_db",
    title=f"Block Error: {layer} ({role}) — {config}",
)

render_chart(
    data=channel_data,          # [{channel_idx, qsnr_db, is_outlier}]
    chart_type="bar",
    x="channel_idx",
    y="qsnr_db",
    hue="is_outlier",
    title=f"Channel Error: {layer} (input) — {config}",
)

render_chart(
    data=comparison_data,       # [{block_idx, config, qsnr_db}]
    chart_type="bar",
    x="block_idx",
    y="qsnr_db",
    hue="config",
    title=f"Block Error Comparison: {layer} ({role})",
)
```

---

## 5. Agent Definitions

### Agent 1: adapter（项目适配）

| Field      | Value                               |
|------------|-------------------------------------|
| after      | `[]`                                |
| tools      | `["bash", "grep", "glob"]`          |
| result_type| `ProjectAnalysis`                   |

**与 example 18 相同**。分析目标项目，找到 model、data、weights，
确认/生成 adapter.py。

```python
class ProjectAnalysis(BaseModel):
    model_class: str
    model_module: str
    model_init_args: dict = {}
    dataset: str
    weights_path: str
    weights_exist: bool
    adapter_path: str
    summary: str
```

### Agent 2: study_runner（多配置量化）

| Field      | Value                               |
|------------|-------------------------------------|
| after      | `["adapter"]`                       |
| tools      | `["bash"]`                          |
| result_type| `StudyResult`                       |

**职责**：
1. 确认 adapter.py 存在且可运行
2. 编写并执行 Python 脚本，调用 `Study.run()` 运行 8 个 config
3. 保存 StudyReport 到 JSON 供下游 agent 读取

**8 个 config**：

| # | name         | w_bits | a_bits | transform    |
|---|-------------|--------|--------|-------------|
| 1 | W8A8        | 8      | 8      | none        |
| 2 | W4A8        | 4      | 8      | none        |
| 3 | W4A4        | 4      | 4      | none        |
| 4 | W8A8+SQ     | 8      | 8      | smoothquant |
| 5 | W4A8+SQ     | 4      | 8      | smoothquant |
| 6 | W4A4+SQ     | 4      | 4      | smoothquant |
| 7 | W8A8+HD     | 8      | 8      | hadamard    |
| 8 | W4A4+HD     | 4      | 4      | hadamard    |

**Observers**: `QSNRObserver()`, `MSEObserver()`, `PerBlockQSNRObserver()`, `DistributionFitObserver()`

```python
class StudyResult(BaseModel):
    status: str                          # "success" | "error"
    output_dir: str                      # StudyReport.save() 的输出目录
    config_names: List[str]              # 所有成功运行的 config 名
    fp32_accuracy: float | None = None
    configs_summary: List[ConfigSummary] = []
    error: str = ""
    summary: str

class ConfigSummary(BaseModel):
    name: str
    accuracy: float | None
    delta: float | None
    avg_qsnr_db: float | None
    avg_mse: float | None
```

**md 指令要点**：
- 用 bash 写一个临时 Python 脚本，import bitx src
- 调用 `Study(configs, model=...).run(calib_data, eval_data=..., eval_fn=...)`
- `study_report.save(output_dir)` 保存结果
- 打印 accuracy table 到 stdout 供 LLM 解析

### Agent 3: gap_analyzer（精度差距分析）

| Field      | Value                               |
|------------|-------------------------------------|
| after      | `["study_runner"]`                  |
| tools      | `["bash"]`                          |
| result_type| `GapAnalysis`                       |

**职责**：
1. 读取 StudyReport JSON
2. 对比各 config 的精度差距
3. 分解：weight degradation (W8A8→W4A8) vs activation degradation (W4A8→W4A4)
4. 量化 transform 的恢复效果

```python
class GapAnalysis(BaseModel):
    # 基础精度
    fp32_accuracy: float
    config_results: List[ConfigGap]

    # 差距分解
    weight_degradation: float       # W8A8→W4A8 accuracy loss
    activation_degradation: float   # W4A8→W4A4 accuracy loss
    primary_bottleneck: str         # "weight" | "activation" | "both"

    # Transform 效果
    transform_recovery: List[TransformRecovery]

    # 结论
    summary: str

class ConfigGap(BaseModel):
    name: str
    accuracy: float
    delta_from_fp32: float

class TransformRecovery(BaseModel):
    config: str                      # e.g. "W4A4"
    transform: str                   # "smoothquant" | "hadamard"
    accuracy_gain: float             # with - without transform
    recovery_pct: float              # % of gap recovered
```

### Agent 4: layer_attribution（层级归因）

| Field      | Value                               |
|------------|-------------------------------------|
| after      | `["study_runner"]`                  |
| tools      | `["bash"]`                          |
| result_type| `LayerAttribution`                  |

**职责**：
1. 读取 StudyReport JSON
2. 用 `CrossConfigLayerRanking` 找跨 config 一致最差层
3. 每个最差层的 role 归因 (activation vs weight dominant)
4. 标记 "config-specific worst" vs "consistently worst"

```python
class LayerAttribution(BaseModel):
    consistent_worst: List[LayerInfo]    # 所有 config 都最差的层
    config_specific_worst: List[ConfigSpecificLayer]  # 只在某 config 最差
    cross_config_delta: List[LayerDelta] # 层级 QSNR 跨 config 变化
    summary: str

class LayerInfo(BaseModel):
    layer: str
    avg_qsnr_db: float                  # 跨 config 平均
    worst_config: str
    worst_qsnr_db: float
    dominant_role: str                   # "input" | "weight" | "output"
    role_qsnr: Dict[str, float]         # {"input": ..., "weight": ..., "output": ...}

class ConfigSpecificLayer(BaseModel):
    layer: str
    config: str
    qsnr_db: float
    dominant_role: str

class LayerDelta(BaseModel):
    layer: str
    w8a8_qsnr: float | None
    w4a8_qsnr: float | None
    w4a4_qsnr: float | None
    w4a8_delta: float | None            # W4A8 - W8A8
    w4a4_delta: float | None            # W4A4 - W4A8
```

### Agent 5: distribution_profiler（分布特征分析）

| Field      | Value                               |
|------------|-------------------------------------|
| after      | `["layer_attribution"]`             |
| tools      | `["bash"]`                          |
| result_type| `DistributionProfile`               |

**职责**：
1. 从 LayerAttribution 获取最差层列表
2. 对每个最差层调用 `result.characterize.profile(layer, role)`
3. 调用 `result.characterize.causal_analysis()` 获取因果矩阵
4. 识别分布异常（outlier-heavy, bimodal, high dynamic range）

```python
class DistributionProfile(BaseModel):
    layer_profiles: List[LayerDistributionProfile]
    causal_summary: str                  # causal_analysis() 的文字摘要
    format_weaknesses: List[FormatWeakness]
    summary: str

class LayerDistributionProfile(BaseModel):
    layer: str
    role: str                            # 分析的 role
    config: str                          # 来自哪个 config
    qsnr_db: float
    distribution_type: str               # "zero-centered-gaussian" | "bimodal" | "outlier-heavy" | ...
    key_features: Dict[str, float]       # {outlier_ratio, dynamic_range_bits, kurtosis, ...}
    diagnosis: str                       # 人可读诊断
    suggestion: str                      # 建议动作

class FormatWeakness(BaseModel):
    format: str                          # "int4" | "int8"
    issue: str                           # e.g. "outlier_ratio 31% exceeds int4 representable range"
    affected_layers: List[str]
    evidence: str                        # 支撑证据
```

### Agent 6: block_analyst（Block 级误差分析 + 可视化）

| Field      | Value                               |
|------------|-------------------------------------|
| after      | `["layer_attribution"]`             |
| tools      | `["bash"]`                          |
| result_type| `BlockAnalysis`                     |

**职责**：
1. 从 LayerAttribution 获取最差层列表
2. 对每个最差层的 weight 和 input 分别做 block/channel 级误差分析
3. 调用 `block_error_analysis()` 获取 BlockErrorReport
4. 渲染 block error heatmap + channel error bar + multi-config comparison
5. 识别：哪些 block 最差、哪些 channel 是 outlier、误差是集中还是均匀

```python
class BlockAnalysis(BaseModel):
    layer_analyses: List[LayerBlockAnalysis]
    summary: str

class LayerBlockAnalysis(BaseModel):
    layer: str
    config: str

    # Weight 分析
    weight_block_qsnr: Dict[str, float] | None   # {block_idx_str: qsnr_db}
    worst_weight_blocks: List[BlockDetail]        # top-k 最差 block
    weight_error_pattern: str                     # "concentrated" | "uniform" | "channel-boundary"

    # Activation 分析
    activation_channel_qsnr: Dict[str, float] | None  # {channel_idx_str: qsnr_db}
    worst_activation_channels: List[ChannelDetail]
    activation_error_pattern: str                      # "outlier_channel" | "uniform" | "feature-correlated"

    # 可视化
    heatmap_rendered: bool
    bar_chart_rendered: bool
    comparison_rendered: bool

    # 关键发现
    finding: str  # 人可读的一句话发现

class BlockDetail(BaseModel):
    block_idx: int
    qsnr_db: float
    stats: Dict[str, float]  # {dynamic_range, outlier_ratio, ...}

class ChannelDetail(BaseModel):
    channel_idx: int
    qsnr_db: float
    stats: Dict[str, float]  # {outlier_ratio, crest_factor, ...}
```

### Agent 7: intervention_evaluator（精度恢复评估）

| Field      | Value                               |
|------------|-------------------------------------|
| after      | `["layer_attribution"]`             |
| tools      | `["bash"]`                          |
| result_type| `InterventionEvaluation`            |

**职责**：
1. 对最差层逐一恢复 FP32，测量 gap recovery %
2. 对最差层做 bit-width boost (int4→int8)，测量 recovery
3. 测试 transform (smoothquant / hadamard) 对最差层的效果
4. 测试组合策略 (top-3 layers boost + smoothquant)

```python
class InterventionEvaluation(BaseModel):
    single_layer_recovery: List[LayerRecovery]     # 单层恢复 FP32
    bit_boost_recovery: List[LayerRecovery]         # 单层 bit boost
    transform_recovery: List[LayerRecovery]         # 单层 transform
    combined_recovery: List[CombinedRecovery]       # 组合策略
    best_strategy: str                              # 最佳策略描述
    summary: str

class LayerRecovery(BaseModel):
    layer: str
    intervention: str                  # "fp32_restore" | "int4_to_int8" | "smoothquant" | "hadamard"
    accuracy_before: float
    accuracy_after: float
    gap_recovered_pct: float           # 恢复了多少比例的 gap
    dominant_role: str

class CombinedRecovery(BaseModel):
    description: str                   # e.g. "top-3 layers int8 + smoothquant"
    layers_modified: List[str]
    accuracy: float
    gap_recovered_pct: float
```

### Agent 8: synthesis（综合诊断报告）

| Field      | Value                               |
|------------|-------------------------------------|
| after      | `["distribution_profiler", "block_analyst", "intervention_evaluator"]` |
| tools      | `["bash"]`                          |
| result_type| `MXIntDiagnosticReport`            |

**职责**：
1. 综合 Agent 3-7 的所有发现
2. 形成最终诊断结论
3. 指出 MXInt 格式的具体缺陷
4. 给出可操作的改进建议

```python
class MXIntDiagnosticReport(BaseModel):
    # 精度概览
    fp32_accuracy: float
    configs: List[ConfigAccuracy]

    # 差距分解
    weight_degradation: float
    activation_degradation: float
    primary_bottleneck: str

    # 层级发现
    consistent_worst_layers: List[str]
    layer_findings: List[LayerFinding]

    # 格式缺陷
    format_weaknesses: List[FormatWeakness]

    # 改进建议
    recommendations: List[Recommendation]

    # 一句话结论
    conclusion: str
    summary: str

class ConfigAccuracy(BaseModel):
    name: str
    accuracy: float
    delta: float
    with_smooth: float | None = None
    with_hadamard: float | None = None

class LayerFinding(BaseModel):
    layer: str
    config: str
    output_qsnr: float
    dominant_role: str
    diagnosis: str
    worst_block_idx: int | None = None
    worst_channel_idx: int | None = None
    recovery_pct: float | None = None

class FormatWeakness(BaseModel):
    format: str
    issue: str
    affected_layers: List[str]
    evidence: str

class Recommendation(BaseModel):
    type: str               # "mixed_precision" | "transform" | "format_change" | "granularity"
    priority: str           # "high" | "medium" | "low"
    target_layers: List[str]
    action: str
    expected_recovery: float  # 预期 gap recovery %
    rationale: str
```

---

## 6. Agent Interface Docs

每个 agent 可读的 API 文档，agent 通过 bash 调用 Python 脚本使用这些 API。

### 6.1 adapter 阶段可用 API

**adapter 合约** (与 example 18 相同)：
```python
# _adapter.py 必须定义:
def get_model() -> nn.Module: ...
def get_eval_fn() -> callable: ...     # eval_fn(model, data) -> {"accuracy": float}
def get_data() -> (list, iterable): ... # (calib_data, eval_data)
```

### 6.2 study_runner 阶段可用 API

```python
from src.session import Session, QuantConfig
from src.session._study import Study
from src.analysis.observers import QSNRObserver, MSEObserver, PerBlockQSNRObserver, DistributionFitObserver

# 构建 8 个 config
configs = []
for w, a in [(8,8), (4,8), (4,4)]:
    configs.append(QuantConfig(name=f"W{w}A{a}", w_format=f"int{w}", a_format=f"int{a}",
                               w_granularity="per_block", a_granularity="per_block", w_block_size=16, a_block_size=16))
    configs.append(QuantConfig(name=f"W{w}A{a}+SQ", ..., transform="smoothquant"))
# ... hadamard variants

study = Study(configs, model=model)
report = study.run(calib_data, eval_data=eval_data, eval_fn=eval_fn, outputs="all")
report.save(output_dir)
```

### 6.3 gap_analyzer / layer_attribution 阶段可用 API

```python
from src.report._study_report import StudyReport
from src.analysis.cross_config_ranking import CrossConfigLayerRanking
from src.analysis.transform_effect import TransformEffectReport

study_report = StudyReport.from_file(output_dir)

# 精度对比
df = study_report.summary_dataframe()

# 跨 config 层排序
ranking = CrossConfigLayerRanking.from_study(study_report)
worst = ranking.consistent_worst(k=5)

# Transform 效果
transform_report = TransformEffectReport.from_study(study_report)
print(transform_report.summary())
```

### 6.4 distribution_profiler 阶段可用 API

```python
# 需要从 StudyReport 中取出单个 SessionResult
result = study_report.get_result(config_name="W4A4")

# 分布诊断
print(result.characterize.profile("transformer.layers.3.linear2", role="weight"))
print(result.characterize.causal_analysis())

# 分类
label = result.characterize.classify("transformer.layers.3.linear2", role="input")
```

### 6.5 block_analyst 阶段可用 API

```python
from src.api.block_error_analysis import block_error_analysis
from src.viz.block_error_heatmap import block_error_heatmap, channel_error_bar, multi_config_block_comparison

# Block 级误差
report = block_error_analysis(result, layer="transformer.layers.3.linear2", role="weight", top_k=10)
print(report.worst_units)

# 可视化
fig = block_error_heatmap(result, "transformer.layers.3.linear2", role="weight")
fig.savefig("block_heatmap.png")

fig = channel_error_bar(result, "transformer.layers.3.linear2", role="input")
fig.savefig("channel_error.png")

fig = multi_config_block_comparison(study_report, "transformer.layers.3.linear2", role="weight")
fig.savefig("block_comparison.png")
```

### 6.6 intervention_evaluator 阶段可用 API

```python
from src.scheme.op_config import OpQuantConfig
from src.analysis._intervention import InterventionPlan

# 单层 FP32 恢复
override_cfg = OpQuantConfig()  # all None = FP32
overrides = {"transformer.layers.3.linear2": override_cfg}
sess = Session(copy.deepcopy(model), config, observers=[QSNRObserver()], keep_fp32=True)
boosted = sess.run(calib_data, eval_data=eval_data, eval_fn=eval_fn, overrides=overrides)

# Bit boost (int4 → int8)
plan = result.plan.top_k_boost(k=5, role="auto", target_bits=8)
comparison = result.intervention.compare(model, calib_data, plan, eval_data=eval_data, eval_fn=eval_fn)
print(comparison.summary())
```

---

## 7. Implementation Order

### Phase A: New bitx APIs (不涉及 agent)

| Step | Task                                             | Files                                       |
|------|--------------------------------------------------|---------------------------------------------|
| A1   | `PerBlockQSNRObserver`                           | `src/analysis/observers.py`                 |
| A2   | `block_error_analysis()` + `BlockErrorReport`    | `src/api/block_error_analysis.py` (new)     |
| A3   | `CrossConfigLayerRanking`                        | `src/analysis/cross_config_ranking.py` (new)|
| A4   | `TransformEffectReport`                          | `src/analysis/transform_effect.py` (new)    |
| A5   | Tests for all new APIs                           | `src/tests/test_per_block_*.py` etc.        |

### Phase B: Visualization

| Step | Task                                             | Files                                       |
|------|--------------------------------------------------|---------------------------------------------|
| B1   | `block_error_heatmap()`                          | `src/viz/block_error_heatmap.py` (new)      |
| B2   | `channel_error_bar()`                            | same file                                   |
| B3   | `multi_config_block_comparison()`                | same file                                   |
| B4   | Tests for visualization functions               | `src/tests/test_block_viz.py`               |

### Phase C: Agent Pipeline

| Step | Task                                             | Files                                       |
|------|--------------------------------------------------|---------------------------------------------|
| C1   | Define result types (Pydantic models)            | `examples/19_mxint_diagnostic.py`           |
| C2   | Define Workflow + Agents                         | same file                                   |
| C3   | Write agent md files (8 agents)                  | `workflows/mxint-diagnostic/agents/*.md`    |
| C4   | `wf.save()` → generate workflow.json             | same file                                   |
| C5   | Write agent interface docs                       | `docs/harness/api-*.md`                     |

### Phase D: Integration Test

| Step | Task                                             |
|------|--------------------------------------------------|
| D1   | 用 bitx 自带的 MLP (MNIST) 做 end-to-end test   |
| D2   | 用 Transformer (AG News) 做 end-to-end test     |
| D3   | 验证所有 charts 正确渲染                        |

---

## 8. File Layout

```
microxcaling/
  src/
    analysis/
      observers.py                    # + PerBlockQSNRObserver
      cross_config_ranking.py         # NEW
      transform_effect.py             # NEW
    api/
      block_error_analysis.py         # NEW
    viz/
      block_error_heatmap.py          # NEW
    tests/
      test_per_block_observer.py      # NEW
      test_block_error_analysis.py    # NEW
      test_cross_config_ranking.py    # NEW
      test_transform_effect.py        # NEW
      test_block_viz.py               # NEW

AgentHarness/
  examples/
    19_mxint_diagnostic.py            # NEW (Python API entry point)
  workflows/
    mxint-diagnostic/
      workflow.json                   # generated by wf.save()
      agents/
        adapter.md                    # Agent 1
        study_runner.md               # Agent 2
        gap_analyzer.md               # Agent 3
        layer_attribution.md          # Agent 4
        distribution_profiler.md      # Agent 5
        block_analyst.md              # Agent 6
        intervention_evaluator.md     # Agent 7
        synthesis.md                  # Agent 8

  docs/harness/
    api-per-block-analysis.md         # Agent 6 接口文档
    api-cross-config-ranking.md       # Agent 4 接口文档
    api-transform-effect.md           # Agent 3,7 接口文档
    api-intervention-eval.md          # Agent 7 接口文档
    example19-spec.md                 # 本文档
```

---

## 9. Analysis Walkthrough (Concrete Example)

以 bitx 自带 Transformer (AG News) 为例，展示完整的分析流程：

### Step 1: Study Runner 输出

```
Config      Accuracy  Delta     Avg QSNR
FP32        0.9120    —         —
W8A8        0.9085    -0.0035   32.4 dB
W4A8        0.8950    -0.0170   21.8 dB
W4A4        0.8580    -0.0540   12.3 dB
W8A8+SQ     0.9100    -0.0020   35.1 dB
W4A8+SQ     0.9010    -0.0110   24.6 dB
W4A4+SQ     0.8790    -0.0330   16.8 dB
W4A4+HD     0.8710    -0.0410   15.1 dB
```

### Step 2: Gap Analyzer 输出

```
Weight degradation (W8A8→W4A8):  -1.35%
Activation degradation (W4A8→W4A4): -3.70%
Primary bottleneck: ACTIVATION

SmoothQuant recovery for W4A4: +2.10% (38.9% of gap recovered)
Hadamard recovery for W4A4:    +1.30% (24.1% of gap recovered)
→ SmoothQuant is more effective for this model
```

### Step 3: Layer Attribution 输出

```
Consistent Worst Layers (across W8A8, W4A8, W4A4):
  transformer.layers.3.linear2   avg QSNR=14.2dB  dominant=input (worst in all configs)
  transformer.layers.1.linear1   avg QSNR=16.8dB  dominant=weight (worst in W4A4)

Config-specific worst:
  transformer.layers.0.self_attn  only bad in W4A4 (QSNR=11.2dB), fine in W8A8 (28.4dB)
```

### Step 4: Distribution Profiler 输出

```
transformer.layers.3.linear2 (input, W4A4):
  QSNR: 8.2 dB
  Distribution: outlier-heavy (outlier_ratio=4.2%, dynamic_range=14.2 bits)
  Diagnosis: 1-2 outlier channels dominate error — int4 (4 bits) cannot represent 14-bit range
  Suggestion: SmoothQuant to redistribute outliers, or per-channel scaling

transformer.layers.1.linear1 (weight, W4A4):
  QSNR: 6.3 dB
  Distribution: bimodal (BC=0.71), dynamic_range=9.1 bits
  Diagnosis: Two weight clusters — single scale clips one cluster
  Suggestion: Per-channel quantization or split block scheme
```

### Step 5: Block Analyst 输出 (+ 可视化)

```
transformer.layers.3.linear2 (weight, W4A4):
  Block error pattern: CONCENTRATED
  Worst 5 blocks (of 150):
    Block 47:  QSNR=2.1 dB  (dynamic_range=11.3, outlier_ratio=12%)
    Block 156: QSNR=3.8 dB  (dynamic_range=9.8,  outlier_ratio=8%)
    Block 23:  QSNR=4.2 dB  (...)
    Block 89:  QSNR=5.1 dB  (...)
    Block 112: QSNR=5.8 dB  (...)
  Median block QSNR: 25.1 dB
  → 3% of blocks (5/150) contain 80% of the total error

  Activation channel error:
    Channel 47: QSNR=1.8 dB  (outlier_ratio=31%, crest_factor=12.3)  ← ROOT CAUSE
    Channel 183: QSNR=4.5 dB (outlier_ratio=15%)
    Median channel QSNR: 28.3 dB
    → Single outlier channel (47) at 31% outlier ratio destroys entire layer's QSNR

  [HEATMAP: 红色热点集中在 block 47, channel 47]
  [BAR: channel 47 远低于其他 channel]
  [COMPARISON: W8A8 时 channel 47 QSNR=18.2dB, W4A4 时骤降至 1.8dB]
```

### Step 6: Intervention Evaluator 输出

```
Single-layer FP32 restore:
  transformer.layers.3.linear2 → FP32: 85.8% → 88.9% (recovered 5.7% of gap)
  transformer.layers.1.linear1 → FP32: 85.8% → 87.5% (recovered 3.1%)

Bit boost (int4→int8) for worst layers:
  transformer.layers.3.linear2 input→int8: 85.8% → 88.2% (recovered 4.4%)
  transformer.layers.1.linear1 weight→int8: 85.8% → 87.1% (recovered 2.4%)

Transform effect on worst layers:
  transformer.layers.3.linear2 + SQ: channel 47 QSNR 1.8→12.5 dB (huge improvement)

Combined strategy (top-3 int8 boost + smoothquant):
  Accuracy: 85.8% → 89.5% (recovered 68.5% of gap)
```

### Step 7: Synthesis Final Report

```
CONCLUSION:
  1. MXInt W8A8 is safe (Δ=0.35%); W4A8 is moderate (Δ=1.7%); W4A4 is risky (Δ=5.4%)
  2. Primary bottleneck is ACTIVATION quantization (3.7% of the 5.4% loss)
  3. Root cause: outlier channels (esp. ch47 in layer3.linear2) exceed int4 range
  4. SmoothQuant recovers 39% of W4A4 gap; Hadamard recovers 24%

FORMAT WEAKNESSES:
  - int4 cannot represent outlier-heavy activations (dynamic_range > 8 bits)
  - per_block(16) too coarse for channels with 30%+ outlier ratio
  - No outlier-aware scaling in base MXInt format

RECOMMENDATIONS:
  1. [HIGH] Mixed precision: keep transformer.layers.3.linear2 at int8 input
     Expected: +3.7% accuracy recovery
  2. [HIGH] Apply SmoothQuant to transformer.layers.3 and layers.1
     Expected: +2.1% accuracy recovery
  3. [MEDIUM] Combine #1+#2: recover ~68% of total gap
  4. [LOW] Consider per-channel activation scaling for outlier channels
```

---

## 10. Key Design Decisions

### Q: 为什么 Agent 2 用 Study 而不是单独 Session？
A: Study 一次运行多个 config，内部做 deepcopy，保证结果一致性。结果保存在一个 StudyReport JSON 里，
   下游 agent 从同一个文件读取，避免状态不一致。

### Q: 为什么 Agent 5/6/7 是并行而不是串行？
A: 它们只依赖 Agent 4 的输出（worst layers list），互相独立。
   并行可以节省时间（每个 agent 都要跑 bash 脚本）。
   Agent 8 等待全部完成后综合。

### Q: PerBlockQSNRObserver 与 QSNRObserver 的区别？
A: QSNRObserver 在 PER_BLOCK 模式下会把所有 block 打平成一个大 tensor 算一个 QSNR。
   PerBlockQSNRObserver 则是逐 block 单独算 QSNR，保留空间分布信息。
   两者可以同时挂在一个 Session 上，互不干扰。

### Q: 可视化为什么放在 bitx src/viz 而不是 agent 脚本里？
A: 可视化是可复用的分析能力，不绑定到特定 agent。
   未来 bitx 的其他分析场景（比如纯 Python 脚本）也能用。
   Agent 通过 bash 调用 Python 一行脚本即可渲染。

### Q: md 文件里要不要写输出格式？
A: **不要**。harness 的 `result_type` 会自动注入 `## Output Format` schema。
   md 文件只写任务逻辑和分析策略。这是 harness 的铁律。
