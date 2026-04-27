# Phase 8: Transform 体系 + Calibration 管线

**Task ID**: P8.1 — Hadamard, SmoothQuant, Calibration
**Branch**: `feature/refactor-src`
**Tests baseline**: 1068 passed, 0 xfail
**范围外**：CLE（Cross-Layer Equalization）、Bias Correction

---

## 设计决策

### D1: SmoothQuantTransform 采用不可变 + 工厂模式

`SmoothQuantTransform` 的 per-channel scale 在构造时传入、不可变。

**Why:**
- `QuantScheme` 是 frozen dataclass，transform 字段的值语义应与 format/granularity 一致——构造完成即完整可用
- 可变 scale 引入"未校准"非法状态，需要运行时 guard；工厂模式消除该状态（make illegal states unrepresentable）
- `__eq__`/`__hash__` 基于不可变 scale，避免 mutable 对象在 dict/set 中的静默 bug
- 校准逻辑（统计收集、scale 计算）与 transform 本身（应用 scale）职责分离：校准是工厂，transform 是纯数据应用

```python
class SmoothQuantTransform(TransformBase):
    invertible = True
    def __init__(self, scale: Tensor): ...
    def forward(self, x): return x / scale
    def inverse(self, x_q): return x_q * scale

    @staticmethod
    def from_calibration(X_act, W_weight, alpha=0.5) -> "SmoothQuantTransform": ...
```

### D2: 文件组织 — `src/transform/` 作为新 package

- `TransformBase` / `IdentityTransform` 保留在 `src/scheme/transform.py`（契约定义）
- 新 transform 实现在 `src/transform/`（与 `src/formats/` 同模式：FormatBase 在 base.py，实现在独立文件）
- `src/transform/__init__.py` 集中导出

### D3: Calibration 分层

- `src/calibration/strategies.py` — 纯函数式 scale 计算（max/percentile/MSE/KL），无状态
- `src/calibration/pipeline.py` — `CalibrationPipeline` 遍历 DataLoader + 聚合统计量 → 产出 scale tensors
- scale 持久化后续集成到 `OpQuantConfig` / 模块 buffer（Phase 8B.3）

---

## 子任务清单

### 8A.1 — Hadamard Transform

**目标**：实现 `HadamardTransform`，验证 Transform 扩展模式（最纯粹的 forward/inverse transform）

**新建文件：**
- `src/transform/__init__.py` — package 初始化，导出 HadamardTransform
- `src/transform/hadamard.py` — `hadamard(x)` 函数 + `HadamardTransform(TransformBase)`

**修改文件：**
- `src/scheme/transform.py` — 无修改（TransformBase 和 IdentityTransform 保留原地）

**实现要点：**
- 利用 Walsh-Hadamard 递归结构实现 O(n log n) FWHT
- 输入长度非 2 的幂时，pad 到 2^n
- `HadamardTransform.forward(x)` = hadamard(x) / √d → `inverse(x_q)` = hadamard(x_q) / √d
- `__eq__` 基于类型匹配，`__hash__` 基于常量字符串

**测试文件：** `src/tests/test_transform_hadamard.py`
- `test_hadamard_orthogonal` — H @ H^T ≈ I
- `test_hadamard_transform_roundtrip` — inverse(forward(x)) ≈ x
- `test_hadamard_transform_quant_scheme` — QuantScheme + HadamardTransform 创建
- `test_hadamard_transform_eq_hash` — value-based equality
- `test_hadamard_non_power_of_two` — pad 行为
- `test_hadamard_quantize_pipeline` — quantize(x, scheme_with_hadamard) 完整性

---

### 8B.1 — Scale Strategy 抽象

**目标**：定义可插拔 scale 计算策略，替换 `_quantize_per_channel` 中硬编码的 `amax = torch.amax(torch.abs(x), dim=axis)`

**新建文件：**
- `src/calibration/__init__.py` — package 初始化
- `src/calibration/strategies.py` — `ScaleStrategy` ABC + 4 种实现

**接口：**
```python
class ScaleStrategy(ABC):
    @abstractmethod
    def compute(self, x: Tensor, axis: int) -> Tensor: ...

class MaxScaleStrategy(ScaleStrategy): ...       # absmax（当前行为）
class PercentileScaleStrategy(ScaleStrategy): ... # N% 分位数
class MSEScaleStrategy(ScaleStrategy): ...        # 网格搜索最小 MSE
class KLScaleStrategy(ScaleStrategy): ...         # KL 散度
```

**测试文件：** `src/tests/test_calibration_strategies.py`
- 各 strategy 的 scale 形状正确性
- `MaxScaleStrategy == 当前 amax 行为`（回归）
- Percentile/MSE/KL 数值合理性（scale 在合理范围内）

---

### 8B.2 — Calibration Pipeline

**目标**：遍历校准数据集 → 收集逐层统计量 → 聚合 → 产出 ScaleStrategy 最终 scale

**新建文件：**
- `src/calibration/pipeline.py` — `CalibrationPipeline`

**接口：**
```python
class CalibrationPipeline:
    def __init__(self, model, strategy: ScaleStrategy, num_batches: int = 64): ...
    def calibrate(self, dataloader) -> dict[str, Tensor]: ...
```

**实现要点：**
- 遍历 dataloader（最多 num_batches），forward pass 收集每层 activation 统计量
- 统计量聚合方式由 strategy 决定：max → running max，percentile → 直方图累积，MSE/KL → 逐 batch 保存
- 最终调用 `strategy.compute(aggregated_stats)` 产出最终 scale
- 暂不修改模型参数（persistence 留到 8B.3）

**测试文件：** `src/tests/test_calibration_pipeline.py`
- `test_pipeline_collects_stats` — 统计量收集完整性
- `test_pipeline_respects_num_batches` — 批次数限制
- `test_pipeline_different_strategies` — 四种 strategy 端到端

---

### 8A.2 — SmoothQuant Transform

**目标**：实现 `SmoothQuantTransform` + `compute_smoothquant_scale()` 工厂

**新建文件：**
- `src/transform/smooth_quant.py`

**实现要点：**
```python
def compute_smoothquant_scale(X_act: Tensor, W: Tensor, alpha: float = 0.5) -> Tensor:
    """s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)"""
    ...

class SmoothQuantTransform(TransformBase):
    invertible = True
    def __init__(self, scale: Tensor):
        # scale 不可变；forward 时除以 scale，inverse 时乘以 scale
    @staticmethod
    def from_calibration(X_act, W_weight, alpha=0.5) -> "SmoothQuantTransform": ...
```

- scale shape: `[C]`，沿 channel dim 广播
- `forward(x)`: `x / scale`（平滑 activation outlier，将量化难度转移到 weight）
- `inverse(x_q)`: `x_q * scale`
- `__eq__`/`__hash__`：基于 scale tensor 值（`torch.equal` + hash of values）

**测试文件：** `src/tests/test_transform_smooth_quant.py`
- `test_compute_scale_alpha_extremes` — alpha=0 只用 weight，alpha=1 只用 activation
- `test_smooth_quant_transform_roundtrip` — inverse(forward(x)) ≈ x
- `test_smooth_quant_from_calibration` — 工厂方法
- `test_smooth_quant_eq_hash` — 相同 scale → 相等；不同 scale → 不等
- `test_smooth_quant_quant_scheme` — QuantScheme 集成
- `test_smooth_quant_quantize_pipeline` — quantize(x, scheme_with_sq) 端到端

---

## 实施顺序

```
8A.1 Hadamard ──┐
                 ├──> 8A.2 SmoothQuant
8B.1 Strategies ─┤
    │             └── 后续：8B.3 Scale 持久化
8B.2 Pipeline ───┘
```

8A.1 和 8B.1 相互独立，可并行。8B.2 依赖 8B.1。8A.2 在 8A.1 和 8B.2 均完成后开始。

---

## 验收标准

| 子任务 | 验收标准 |
|---|---|
| 8A.1 | `pytest src/tests/test_transform_hadamard.py -x` 通过，全量 1068 不 regression |
| 8B.1 | `pytest src/tests/test_calibration_strategies.py -x` 通过，MaxStrategy 与现有 amax 行为 bit-exact |
| 8B.2 | `pytest src/tests/test_calibration_pipeline.py -x` 通过 |
| 8A.2 | `pytest src/tests/test_transform_smooth_quant.py -x` 通过 |
| 总门 | `pytest src/tests/ -x` 全量通过（>1068），0 xfail |

---

## 涉及文件总览

| 新建 | 修改 |
|---|---|
| `src/transform/__init__.py` | `src/calibration/__init__.py`（渐进添加） |
| `src/transform/hadamard.py` | `docs/status/CURRENT.md`（每子任务更新） |
| `src/transform/smooth_quant.py` | |
| `src/calibration/__init__.py` | |
| `src/calibration/strategies.py` | |
| `src/calibration/pipeline.py` | |
| `src/tests/test_transform_hadamard.py` | |
| `src/tests/test_transform_smooth_quant.py` | |
| `src/tests/test_calibration_strategies.py` | |
| `src/tests/test_calibration_pipeline.py` | |
