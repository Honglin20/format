# ADR-010 实施计划

## Phase 1: 数据基础 — 多 role QSNR 提取 (1 file)

**目标**: 一次遍历提取 input/weight/output 所有 role 的 QSNR/MSE，替代当前只取 output 的行为

### 1.1 重构 `_extract_qsnr_mse()` (`src/session/_session.py`)
- 当前签名: `_extract_qsnr_mse(data, role="output")` → 一次只取一个 role
- 新签名: `_extract_all_roles_qsnr_mse(data)` → 返回 `Dict[role, Dict[layer, float]]`
- 逻辑不变，只是把 role 循环从外部移入内部，一次遍历完成
- 同时在 `Session.analyze()` 中调用，存储 `_qsnr_by_role` 和 `_mse_by_role`
- SessionResult 新增字段: `qsnr_by_role: Dict[str, Dict[str, float]]` (role → layer → value)

### 1.2 扩展 `qsnr_per_role()` (`src/session/_result.py`)
- 已有方法，不变。但从新字段读取而非每次重新遍历 observers_data

### 1.3 测试
- `src/tests/test_error_analysis.py`: 验证三 role 提取的正确性，验证 input/weight/output 都在 dict 中

---

## Phase 2: Diagnose + Plot (7 new files, 2 modified)

**目标**: 用户可以 `result.diagnose.*` 看到误差溯源，`result.plot.*` 看到诊断图

### 2.1 ErrorProvenance (`src/analysis/_error_provenance.py`)

```python
class ErrorProvenance:
    def __init__(self, result: SessionResult): ...
    def summary(self) -> str              # 按 role × layer_type 分层统计表
    def per_role_table(self, max_layers) -> str  # 每层 input/weight/output 并排
    def error_source_analysis(self) -> str        # 委托给 result.correlate_hook_observer
    def top_k(self, k, role) -> List[Tuple]       # role="auto": 每层取最差 role
```

依赖: `result.qsnr_by_role`, `result.accum_qsnr_per_layer`, `result.observers_data`

### 2.2 Propagation 图 (`src/viz/_propagation.py`)

```python
def plot_propagation_dag(result: SessionResult) -> plt.Figure
    # 有向图: 节点=层名, 颜色=local QSNR colormap, 边上标 accum QSNR 衰减量
def plot_error_waterfall(result: SessionResult) -> plt.Figure
    # X轴=层(按深度), Y轴=accum QSNR, 每层下降量着色
def plot_local_vs_accum_scatter(result: SessionResult) -> plt.Figure
    # 散点: X=local QSNR, Y=accum QSNR, 对角线参考线, 点标层名
```

### 2.3 Per-role 图 (`src/viz/_per_role.py`)

```python
def plot_per_role_qsnr_bars(result: SessionResult) -> plt.Figure
    # 堆叠 grouped bar: 每层三根 bar (input/weight/output)
def plot_depth_decay(result: SessionResult, role: str) -> plt.Figure
    # 折线图: X=深度, Y=QSNR, 一条 role 或三条线
```

### 2.4 SessionPlotAccessor (`src/report/_session_plot.py`)

```python
class SessionPlotAccessor:
    def __init__(self, result: SessionResult): ...
    # 传播
    def propagation_dag(self) -> plt.Figure
    def error_waterfall(self) -> plt.Figure
    def local_vs_accum_scatter(self) -> plt.Figure
    # Per-role
    def per_role_qsnr_bars(self) -> plt.Figure
    def depth_decay(self, role="output") -> plt.Figure
    # 分布 (Phase 3 会用到)
    def layer_histogram(self, layer, role) -> plt.Figure
    def channel_heterogeneity(self, layer, role) -> plt.Figure
```

每个方法调用 `src/viz/_propagation.py` 或 `src/viz/_per_role.py` 中的对应函数。

### 2.5 现有文件扩展 (`src/report/_session_tables.py`)

```python
class SessionTablesAccessor:
    # 新增方法
    def per_role_qsnr(self, max_layers=60) -> str
        # 每层 input/weight/output QSNR 三列表格
```

### 2.6 挂载 property (`src/session/_result.py`)

```python
class SessionResult:
    # 新字段
    qsnr_by_role: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mse_by_role: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 新 property
    @property
    def diagnose(self) -> "ErrorProvenance": ...
    
    @property
    def plot(self) -> "SessionPlotAccessor": ...
```

### 2.7 测试
- `src/tests/test_error_analysis.py`: ErrorProvenance 四个方法的输出格式验证
- 图不测试渲染，只测试不抛异常

---

## Phase 3: Characterize — 规则引擎 (2 new files, 1 modified)

**目标**: 用户可以 `result.characterize.*` 知道"为什么这层量化差"

### 3.1 规则引擎 (`src/analysis/_distribution_diagnosis.py`)

```python
# 退化分类枚举 + 判定阈值
_DEGRADATION_RULES = [
    ("outlier_dominated",    lambda m: m["outlier_ratio"] > 0.02 and m["crest_factor"] > 10),
    ("high_dynamic_range",  lambda m: m["dynamic_range_bits"] > 8 and m["outlier_ratio"] <= 0.02),
    ("heavy_tailed",        lambda m: m["excess_kurtosis"] > 3 and m["outlier_ratio"] > 0.01),
    ("bimodal",             lambda m: m["bimodality_coefficient"] > 0.7),
    ("low_entropy",         lambda m: m["norm_entropy"] < 0.3),
    ("benign",              lambda m: True),  # fallback
]

def classify_distribution(metrics: dict) -> str: ...
    # 按优先级匹配，返回第一个命中的分类名

class DistributionDiagnosis:
    def __init__(self, result: SessionResult): ...
    def profile(self, layer: str, role: str) -> str
        # 单层深度诊断: 分布族 + 关键统计量 + 分类 + 建议
    def classify(self, layer: str, role: str) -> str
        # 退化分类名
    def causal_analysis(self) -> str
        # 全局矩阵: 每层每 role 一行，含 QSNR + 分布统计 + 分类
```

依赖: `result.observers_data` (需要 DistributionObserver 已采集数据)。若数据缺失，方法返回带 HOW-TO 的提示文本（优雅降级）。

### 3.2 部分依赖 Phase 2 的 plot 方法

`SessionPlotAccessor.layer_histogram()` 和 `.channel_heterogeneity()` 已在 Phase 2 中定义了接口，Phase 3 实现：

- `layer_histogram`: 从 `observers_data` 取 HistogramObserver 数据，overlay fp32/quant hist
- `channel_heterogeneity`: 从 observers_data 取 per-channel QSNR，画 violin plot

### 3.3 挂载 property (`src/session/_result.py`)

```python
@property
def characterize(self) -> "DistributionDiagnosis": ...
```

### 3.4 测试
- `src/tests/test_error_analysis.py`: 规则引擎每个分类的边界条件
- 手工构造 mock metrics dict，验证 classify 返回正确分类名

---

## Phase 4: Intervene — per-layer override + 对比 (3 new files, 1 modified)

**目标**: 用户可以 `result.plan.*` 生成方案，`result.intervention.compare()` 验证

### 4.1 InterventionPlanner (`src/analysis/_intervention.py`)

```python
class InterventionPlan:
    overrides: Dict[str, OpQuantConfig]
    metadata: Dict[str, Any]  # strategy, k, role, etc.
    
    def explain(self) -> str     # 文本表: 每层改了什么、为什么、预期提升
    def to_dict(self) -> dict     # 序列化

class InterventionPlanner:
    def __init__(self, result: SessionResult): ...
    def top_k_boost(self, k: int, role: str, target_bits: int) -> InterventionPlan
        # role="auto": 每层取 QSNR 最低的 role
        # 生成新 OpQuantConfig: 该 role 的 format 替换为 target_bits 宽度
    def transform_ranking(self, k: int) -> str
        # 复用 _estimate_layer_qsnr 对每层评估 none/hadamard/smoothquant
        # 返回排序表含 margin
    def recommend(self, strategy="conservative") -> InterventionPlan
        # 组合 top_k_boost + transform_ranking
        # conservative: 只改 QSNR < 15dB 且 transform margin > 3dB 的
        # aggressive: QSNR < 25dB 的全改
```

依赖: `result.qsnr_by_role`, `_estimate_layer_qsnr` (已有，在 `_session.py`)，`DistributionDiagnosis.classify` (可选)

### 4.2 Session overrides (`src/session/_session.py`)

```python
class Session:
    def __init__(self, ..., overrides: Optional[Dict[str, OpQuantConfig]] = None):
        self._overrides = overrides
    
    def quantize(self, ...):
        base_cfg = self._config.to_op_config()
        if self._overrides:
            from src.session._model import _get_quantized_modules
            per_layer = {}
            for name, _ in _get_quantized_modules(model):
                per_layer[name] = base_cfg
            per_layer.update(self._overrides)
            self._quant_session = _QuantSession(model, per_layer, ...)
        else:
            self._quant_session = _QuantSession(model, base_cfg, ...)
```

### 4.3 InterventionAccessor (`src/analysis/_intervention_accessor.py`)

```python
class InterventionComparison:
    baseline: SessionResult
    intervention: SessionResult
    plan: InterventionPlan
    
    def print_summary(self) -> str        # 对比表
    @property
    def plot(self) -> "InterventionPlotAccessor": ...  # before_after, qsnr_improvement

class InterventionAccessor:
    def __init__(self, result: SessionResult): ...
    def compare(self, plan, model, calib_data, eval_fn) -> InterventionComparison
        # 内部: Session(model, cfg, overrides=plan.overrides).run(...)
```

### 4.4 挂载 property (`src/session/_result.py`)

```python
@property
def plan(self) -> "InterventionPlanner": ...
@property
def intervention(self) -> "InterventionAccessor": ...
```

### 4.5 测试
- `src/tests/test_error_analysis.py`: top_k_boost 生成正确 OpQuantConfig, recommend 策略逻辑
- `src/tests/test_session.py`: Session overrides 参数正确合并到 _QuantSession dict
- E2E: `scripts/mnist_hadamard_study.py` 加一个 `--overrides` smoke test

---

## 文件依赖关系

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4
                    │                  │
                    └── SessionPlotAccessor.layer_histogram()
                         channel_heterogeneity() 在 Phase 3 实现
```

- Phase 2 的 plot accessor 定义接口，Phase 3 的 distribution plot 方法填充实现
- Phase 4 依赖 Phase 1 的 `qsnr_by_role` 和 Phase 3 的 `classify`
- 每个 phase 不阻塞后续 phase 的新文件创建，但 property 挂载需顺序进行

---

## 新增文件汇总

| 文件 | Phase | 内容 |
|------|-------|------|
| `src/analysis/_error_provenance.py` | P2 | ErrorProvenance accessor |
| `src/analysis/_distribution_diagnosis.py` | P3 | DistributionDiagnosis + 规则引擎 |
| `src/analysis/_intervention.py` | P4 | InterventionPlanner + InterventionPlan |
| `src/analysis/_intervention_accessor.py` | P4 | InterventionAccessor + InterventionComparison |
| `src/report/_session_plot.py` | P2 | SessionPlotAccessor |
| `src/viz/_propagation.py` | P2 | DAG / waterfall / scatter 图 |
| `src/viz/_per_role.py` | P2 | per-role bar / depth decay 图 |
| `src/tests/test_error_analysis.py` | P1-4 | 全链路测试 |

## 修改文件汇总

| 文件 | Phase | 变更 |
|------|-------|------|
| `src/session/_session.py` | P1, P4 | `_extract_qsnr_mse` 改多 role + Session overrides |
| `src/session/_result.py` | P1-4 | 新字段 + 5 个 property |
| `src/report/_session_tables.py` | P2 | `per_role_qsnr()` 表方法 |
