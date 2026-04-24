# P2F-2 代码审查发现（2026-04-24）

**审查对象**: `feature/refactor-src` 分支，P2F-1 + P2F-2 实现  
**审查提交**: `1401cfb` (P2F-2) + `6664476` / `c991b43` (P2F-1 + fix)  
**涉及文件**:
- `src/scheme/granularity.py`
- `src/scheme/transform.py`
- `src/scheme/quant_scheme.py`
- `src/tests/test_formats_equiv.py`

---

## 总结

| 级别 | 数量 | 说明 |
|---|---|---|
| **Critical** | 3 | 必须修复后方可进入 P2F-3 |
| **Major** | 4 | 建议本轮修复 |
| **Minor** | 2 | 可在后续 PR 中清理 |
| OK | 4 | 无需动作 |

---

## Critical 问题（阻塞进入 P2F-3）

### C1 — `TransformBase` 未约束子类实现 `__eq__`/`__hash__`

**文件**: `src/scheme/transform.py`

`QuantScheme` 是 `frozen=True` 的 dataclass，Python 自动生成的 `__hash__` 和 `__eq__` 依赖所有字段，包括 `transform`。`IdentityTransform` 正确实现了这两个方法，但 `TransformBase` 基类未将其声明为抽象方法。

**后果**：用户自定义 `TransformBase` 子类时若忘记实现，默认使用 id-based 比较，导致两个语义相同的 `QuantScheme` 实例不相等、在 set/dict 中无法去重，且**完全静默**，无任何报错。

**修复**：
```python
# transform.py — TransformBase 中增加：
from abc import ABC, abstractmethod

class TransformBase(ABC):
    invertible: bool = False

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    def inverse(self, x_q: Tensor) -> Tensor:
        return x_q

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Required for QuantScheme value equality."""

    @abstractmethod
    def __hash__(self) -> int:
        """Required for QuantScheme hashability."""
```

**需补充测试**：
```python
def test_transform_base_requires_eq_and_hash():
    """Concrete TransformBase without __eq__/__hash__ should be rejected."""
    # 可通过 ABC 机制保证，或写一个不实现的子类确认 TypeError
```

---

### C2 — `QuantScheme.__post_init__` 未验证 `transform` 字段类型

**文件**: `src/scheme/quant_scheme.py:43-56`

`__post_init__` 对 `format` 字段做了完整的类型验证（`TypeError`），对 `round_mode` 做了值验证，但对 `transform` **完全没有类型验证**。

**后果**：
```python
QuantScheme(format="fp8_e4m3", transform="not_a_transform")  # 静默通过！
# 直到调用 quantize() 时才在 transform.forward() 处报 AttributeError
# 错误信息完全不指向根因
```

**修复**：在 `__post_init__` 中添加：
```python
if not isinstance(self.transform, TransformBase):
    raise TypeError(
        f"transform must be TransformBase, got {type(self.transform).__name__}"
    )
```

**需补充测试**：
```python
def test_quant_scheme_invalid_transform_type_raises():
    with pytest.raises(TypeError, match="transform must be TransformBase"):
        QuantScheme(format="fp8_e4m3", transform="invalid")

def test_quant_scheme_invalid_transform_none_raises():
    with pytest.raises(TypeError, match="transform must be TransformBase"):
        QuantScheme(format="fp8_e4m3", transform=None)
```

---

### C3 — `per_channel` 签名变更引入静默语义错误

**文件**: `src/scheme/quant_scheme.py:62-65`

旧签名：`per_channel(format, round_mode="nearest")`  
新签名：`per_channel(format, axis=0, round_mode="nearest")`

两者之间插入了 `axis` 位置参数，导致旧式调用 `per_channel("fp8_e4m3", "floor")` 不报错，但语义完全改变：`"floor"` 被当作 `axis`，`round_mode` 默认为 `"nearest"`。

**修复**：在 `per_channel` 开头加类型守卫：
```python
@staticmethod
def per_channel(format: Union[str, FormatBase], axis: int = 0,
                round_mode: str = "nearest") -> "QuantScheme":
    if isinstance(axis, str):
        raise TypeError(
            f"axis must be int, not str. "
            f"Did you mean: per_channel({format!r}, round_mode={axis!r})? "
            f"The API changed: axis was inserted before round_mode."
        )
    ...
```

**需补充测试**：
```python
def test_per_channel_rejects_string_axis():
    with pytest.raises(TypeError, match="axis must be int"):
        QuantScheme.per_channel("fp8_e4m3", "floor")
```

---

## Major 问题（建议本轮修复）

### M1 — 默认 `format="int8"` 无文档说明

**文件**: `src/scheme/quant_scheme.py:34`

```python
format: FormatBase = field(default_factory=lambda: _resolve_format("int8"))
```

`QuantScheme()` 无参构造时静默使用 int8 + per_tensor + identity，但无任何文档说明这是设计的默认配置。

**修复**：在 `QuantScheme` 的 docstring 中明确写出：
```
Default configuration: INT8Format, per_tensor, IdentityTransform, round_mode="nearest".
```

### M2 — 缺少 `IdentityTransform` hash 一致性测试

**文件**: `src/tests/test_formats_equiv.py`

现有测试只验证 `IdentityTransform() == IdentityTransform()`，未验证两个实例在 `QuantScheme` 中的 hash 一致性（两者 hash 相同、在 set 中去重）。

**需补充测试**：
```python
def test_quant_scheme_identity_transform_hash_stable():
    s1 = QuantScheme(format="fp8_e4m3", transform=IdentityTransform())
    s2 = QuantScheme(format="fp8_e4m3", transform=IdentityTransform())
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert len({s1, s2}) == 1
```

### M3 — `channel_axis` 允许负整数，无文档无验证

**文件**: `src/scheme/granularity.py`

`GranularitySpec.per_channel(axis=-1)` 静默通过，但量化运算时行为未定义。

**修复方案**（二选一）：
- **支持负轴**：在 docstring 中明确"支持 NumPy 风格负索引（-1 = 最后一维）"，并确保量化函数正确处理
- **不支持负轴**：在 `__post_init__` 中加 `if self.channel_axis < 0: raise ValueError(...)`

### M4 — 缺少自定义 Transform 在 QuantScheme 中的相等性测试

**文件**: `src/tests/test_formats_equiv.py`

没有测试验证：一个正确实现了 `__eq__`/`__hash__` 的自定义 `TransformBase` 子类，在两个 `QuantScheme` 实例中能产生正确的相等性和去重行为。

**需补充测试**：
```python
def test_quant_scheme_custom_transform_equality():
    class MyTransform(TransformBase):
        def forward(self, x): return x
        def __eq__(self, other): return isinstance(other, MyTransform)
        def __hash__(self): return hash("MyTransform")

    s1 = QuantScheme(format="fp8_e4m3", transform=MyTransform())
    s2 = QuantScheme(format="fp8_e4m3", transform=MyTransform())
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert len({s1, s2}) == 1
```

---

## Minor 问题

### m1 — `test_quant_scheme_immutability` 命名误导

**文件**: `src/tests/test_formats_equiv.py:248`

该测试实际上验证的是 hashability（能用作 dict key），不是 immutability（字段不可修改）。`test_quant_scheme_frozen` 才是真正的 immutability 测试。

**修复**：将 `test_quant_scheme_immutability` 重命名为 `test_quant_scheme_hashable`。

### m2 — `object.__setattr__` 缺注释说明

**文件**: `src/scheme/quant_scheme.py:47-48`

`object.__setattr__(self, "format", ...)` 是 frozen dataclass 的标准模式，但初次阅读容易引起误解。建议加一行注释：
```python
# frozen dataclass 的标准做法：在 __post_init__ 内用 object.__setattr__ 做强制转换
object.__setattr__(self, "format", _resolve_format(self.format))
```

---

## 架构合规性（符合预期，无需动作）

- `src/quantize/` 中 MxSpecs 残留 — 正常，P2F-5 处理 ✓
- `FormatBase` 无 `quantize()` 抽象方法 — 正常，P2F-3 处理 ✓
- P2F-1（GranularitySpec）实现设计干净，验证逻辑完整 ✓
- 测试覆盖密度高，等价性测试策略正确 ✓
