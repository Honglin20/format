# ADR-007: P6 Coarse Cost Model — 架构设计

## 概述

为量化模型提供**无需真实硬件部署**的延迟和显存估算。基于 modified roofline model，
输入为模型图结构 + 量化配置，输出逐层和总计的延迟/显存。

**公式权威来源**：`docs/refs/p6-cost-model-formulas.md`。实现必须与公式文档完全一致。

**目标精度**：与真实硬件测量误差 < 50%。

---

## 1. 包结构

```
src/cost/                        # 新 package
├── __init__.py                  # 导出 CostReport, analyze_model_cost, DeviceSpec
├── defaults.py                  # 全部可调常数（按需修改）
├── device.py                    # DeviceSpec: peak FLOPS, bandwidth, memory
├── op_cost.py                   # 逐算子 cost 函数
├── model_cost.py                # 模型遍历 + 逐层聚合
└── report.py                    # CostReport 输出（DataFrame + 格式化打印）
```

Cost model 是纯函数式、无状态的：输入图结构和配置，输出数值。不依赖 forward pass。

---

## 2. 数据模型

### 2.1 DeviceSpec

```python
@dataclass
class DeviceSpec:
    peak_flops_fp32: float    # TFLOPS
    memory_bandwidth_gbs: float  # GB/s
    device_memory_gb: float      # GB
    utilization: float = 0.4     # 0-1
    kernel_overhead: float = 1.3 # >1.0
```

### 2.2 OpCost

```python
@dataclass
class OpCost:
    op_name: str
    op_type: str                # "linear" / "conv2d" / "layernorm" / ...
    flops_math: int
    flops_quantize: int
    flops_transform: int
    bytes_read: int
    bytes_write: int
    latency_us: float
    memory_weight_bytes: int
    memory_activation_bytes: int
```

### 2.3 CostReport

```python
class CostReport:
    layers: list[OpCost]
    model_name: str = ""
    
    @property
    def total_latency_us(self) -> float: ...
    @property
    def total_memory_bytes(self) -> int: ...
    
    def summary(self) -> dict: ...
    def to_dataframe(self): ...       # pandas or list of dicts
    def print_summary(self): ...
    def print_per_layer(self): ...
```

---

## 3. Session 集成

`QuantSession` 新增方法，与 `analyze()` / `calibrate()` / `compare()` / `export_onnx()` 并列：

```python
def estimate_cost(self, fp32: bool = False) -> CostReport:
    """返回当前模型的 cost 估算。不需要 forward pass，同步返回。"""
    from src.cost.model_cost import analyze_model_cost
    model = self.fp32_model if fp32 else self.qmodel
    return analyze_model_cost(model)
```

### 3.1 Pipeline 集成

`run_experiment()` 返回 dict 自动附加：

```python
return {
    "accuracy": ...,
    "report": ...,
    "cost": session.estimate_cost(),       # 新增
    "cost_fp32": session.estimate_cost(fp32=True),  # 新增
}
```

### 3.2 独立使用

不走 session 也可直接调用：

```python
from src.cost.model_cost import analyze_model_cost

fp32_cost = analyze_model_cost(fp32_model)
qmodel = quantize_model(fp32_model, cfg)
q_cost = analyze_model_cost(qmodel)
q_cost.print_comparison(fp32_cost)
```

---

## 4. 关键设计决策

### D1: Cost model 不开 observer / context manager

与 `analyze()` 不同，cost model 只看图结构和量化配置，不需要 forward pass。设计为同步函数调用，不引入 context manager 模式。

### D2: OpQuantConfig 推断

model_cost 遍历 `nn.Module` 树时，通过模块属性推断 op 类型和量化配置：
- `isinstance(m, QuantizedLinear)` → op_type = "linear"
- `m.cfg`（OpQuantConfig）→ 各 role 的 scheme
- `m.weight.shape` 等 → tensor shapes

对于非量化模块（FP32 模型），所有 scheme 视为 None。

### D3: 量化步骤数硬编码

各算子类型的量化步骤数量（§3.2 of formulas doc）是代码结构的静态属性，在 `op_cost.py` 中硬编码为 dict，不通过运行时计数。新增算子类型时需同步更新。

### D4: 公式与实现严格对应

`docs/refs/p6-cost-model-formulas.md` 是公式唯一权威来源。`op_cost.py` 中每个函数的 docstring 引用公式文档的具体节号。Review 时逐节对比公式与实现。

---

## 5. 测试策略

| 测试 | 验证内容 |
|---|---|
| `test_cost_defaults.py` | DeviceSpec 构造 + defaults 导入 + 负数/零守卫 |
| `test_cost_op_cost.py` | 每种 op 类型的 `op_cost()` 函数，验证 shape 变化影响 FLOPs 方向正确、None scheme 跳过量化开销、quantize 步骤计数正确 |
| `test_cost_model_cost.py` | 小模型（2-3 layer）端到端，验证逐层聚合 = 手动 sum、激活值取 max 而非 sum |
| `test_cost_report.py` | CostReport.to_dataframe / print_summary / print_comparison 输出完整性 |
| `test_cost_integration.py` | QuantSession.estimate_cost() 端到端、pipeline run_experiment 含 cost key |

数值精确度测试：对小模型手工计算预期 FLOPs，assert 误差 < 1%。
