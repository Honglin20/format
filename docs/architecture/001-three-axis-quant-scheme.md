# ADR-001: 三轴量化方案（Format + Granularity + Transform）

**状态**: 已决策  
**日期**: 2026-04-24

---

## 背景与问题

原 `mx/` 库的量化配置以 39-key `MxSpecs` dict 表达，格式和粒度逻辑全部硬编码在 `elemwise_ops.py` 的 if-elif 链中。这导致：
- 新增格式（nf4, APOT）需要改动量化核心函数
- Granularity 无法独立参数化（block_size 固定为 MX 规范）
- 变换（Hadamard 等）无处可放

`src/` 的 Phase 2 初版引入了 `QuantScheme` 和 `FormatBase`，但 `QuantScheme` 只有 format 字符串（非对象）、granularity 是虚设的 enum、无 transform 字段，量化函数仍依赖 `MxSpecs`。

## 决策

采用**三轴可组合设计**：`QuantScheme = format + granularity + transform`。三个轴独立可插拔，任意组合均合法。量化核心函数只依赖 `QuantScheme`，不感知具体格式。

---

## 接口规范

### QuantScheme

```python
@dataclass(frozen=True)
class QuantScheme:
    format: FormatBase            # 格式对象（Strategy 模式）
    granularity: GranularitySpec  # 粒度规格
    transform: TransformBase = field(default_factory=IdentityTransform)
    round_mode: str = "nearest"   # "nearest" | "floor" | "even" | "dither"
```

**工厂方法（语义化构造）**：
```python
QuantScheme.mxfp(FP8E4M3Format(), block_size=32)
QuantScheme.per_tensor(INT8Format())
QuantScheme.per_channel(FP8E5M2Format(), axis=0)
QuantScheme.mxfp(FP4Format(), block_size=16, transform=HadamardTransform())
```

**量化执行入口**（量化核心函数的统一签名）：
```python
def quantize(x: Tensor, scheme: QuantScheme) -> Tensor:
    x_t = scheme.transform.forward(x)
    x_q = scheme.format.quantize(x_t, scheme.granularity, scheme.round_mode)
    return scheme.transform.inverse(x_q)
```

---

### FormatBase（Strategy 模式）

```python
class FormatBase(ABC):
    name: str     # 注册键，如 "fp8_e4m3"
    ebits: int    # 指数位数（0 = 整数格式）
    mbits: int    # 尾数位数

    @abstractmethod
    def quantize(
        self,
        x: Tensor,
        granularity: "GranularitySpec",
        round_mode: str,
    ) -> Tensor:
        """将 x 量化到本格式。granularity 决定 scale 的共享方式。"""

    @property
    def is_integer(self) -> bool:
        return self.ebits == 0
```

**注册机制**：
```python
from src.formats.registry import register_format, get_format

@register_format("nf4")
class NF4Format(FormatBase):
    def quantize(self, x, granularity, round_mode): ...
```

**当前已注册格式**：

| 名称 | 类 | ebits | mbits | 说明 |
|---|---|---|---|---|
| `int2`…`int8` | `IntFormat` | 0 | 2-8 | 对称整数 |
| `fp4_e2m1` | `FPFormat` | 2 | 2 | |
| `fp6_e2m3`, `fp6_e3m2` | `FPFormat` | 2/3 | 3/2 | |
| `fp8_e4m3`, `fp8_e5m2` | `FPFormat` | 4/5 | 3/2 | |
| `bf16` | `BF16Format` | 8 | 8 | |
| `fp16` | `FP16Format` | 5 | 10 | |

**扩展新格式步骤**（只做这两步，不改其他文件）：
1. 在 `src/formats/` 下新建文件，继承 `FormatBase`，实现 `quantize()`
2. 用 `@register_format("your_name")` 注册

---

### GranularitySpec

```python
@dataclass(frozen=True)
class GranularitySpec:
    mode: GranularityMode          # PER_TENSOR | PER_CHANNEL | PER_BLOCK
    block_size: int = 0            # PER_BLOCK 时必须 > 0
    channel_axis: int = 0          # PER_CHANNEL 时指定 axis（默认 0 = output channel）
```

**channel_axis 负值与越界**：支持 NumPy 风格负索引（如 `axis=-1` 表示最后一维）。由于 `GranularitySpec` 不持有张量形状，越界检查延迟到 `FormatBase.quantize()` 中动态做。

**预定义快捷实例**：
```python
GranularitySpec.per_tensor()           # block_size=0, channel_axis=0
GranularitySpec.per_channel(axis=0)    # output channel axis
GranularitySpec.per_block(32)          # block_size=32
GranularitySpec.per_block(16)          # NVFP4 / smaller block
```

**MXINT / MXFP 映射**：
- `MXINT8` = `INT8Format() + GranularitySpec.per_block(32)`
- `MXFP8` = `FP8E4M3Format() + GranularitySpec.per_block(32)`
- `MXFP4` = `FP4Format() + GranularitySpec.per_block(32)`
- `NVFP4`（未来）= `FP4Format() + GranularitySpec.per_block(16)` + 不同 shared_exp 计算

`FormatBase.quantize()` 接收 `GranularitySpec` 对象并据此计算 scale：
- `PER_BLOCK`：沿最后一维切块，每块计算 shared exponent（mx 风格）
- `PER_CHANNEL`：沿 `channel_axis` 计算每通道 max 作为 scale
- `PER_TENSOR`：全张量单一 scale

---

### TransformBase（可选逆变换）

```python
class TransformBase(ABC):
    invertible: bool = False       # 子类声明是否可逆

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """量化前变换。"""

    def inverse(self, x_q: Tensor) -> Tensor:
        """量化后逆变换。默认 identity（不可逆变换的默认行为）。"""
        return x_q

class IdentityTransform(TransformBase):
    invertible = True
    def forward(self, x): return x
    def inverse(self, x_q): return x_q
```

**量化流程**：
```
输入 x
  → transform.forward(x)          # 可选：Hadamard 旋转、absmax rescale 等
  → format.quantize(x_t, gran, round_mode)   # 格式量化
  → transform.inverse(x_q)        # 可选：逆变换还原
  → 量化输出
```

**Hadamard 示例**（Phase 2 修正后实现）：
```python
class HadamardTransform(TransformBase):
    invertible = True

    def forward(self, x):
        return hadamard(x)   # 正交旋转，降低量化误差

    def inverse(self, x_q):
        return hadamard(x_q) # 转置即逆（正交矩阵）
```

---

## 可扩展性分析

| 新需求 | 需要做什么 | 不需要改什么 |
|---|---|---|
| 新格式（nf4） | 新建 NF4Format，注册 | elemwise.py, mx_quantize.py, QuantScheme |
| 新粒度（NVFP4 块级） | GranularitySpec.per_block(16) 直接可用 | 量化核心函数 |
| 新变换（旋转量化） | 新建 RotationTransform | QuantScheme, 量化核心函数 |
| APOT（幂次量化） | 新建 APOTFormat，实现 quantize() | 其他所有文件 |

---

## 被拒绝的方案

**方案：保留 MxSpecs + QuantScheme 门面（to_specs() 转换）**
拒绝原因：随格式数量增长，`to_specs()` 成为所有格式逻辑的单点，不支持"加格式不改核心代码"的可扩展性目标。且 MxSpecs 39-key schema 不稳定，新 granularity 参数无标准位置。
