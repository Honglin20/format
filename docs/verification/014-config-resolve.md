# 014: resolve_config() — Descriptor-to-OpQuantConfig 合约

**对应测试函数**: `test_resolve_config()`, `test_resolve_config_errors()`
**验证层级**: Layer 4 — Pipeline Refactor

## 验证内容

`resolve_config(descriptor: dict) -> OpQuantConfig` 的完整合约：将"Config as Data"的 descriptor dict（来自搜索空间数据或用户手动构造）转换为框架内部的 `OpQuantConfig` 对象。这是不可变配置数据（dict）与有类型量化的 `src/` 框架之间的桥梁。

## 合约

### 签名

```python
def resolve_config(descriptor: Dict[str, Any]) -> OpQuantConfig:
    ...
```

接收一个 descriptor dict，返回一个 `OpQuantConfig`，其 `input` / `weight` / `output` 角色按 descriptor 中的 `weight_only` 标志填充为 `QuantScheme` 或 `None`。

### 输入 descriptor 格式

| 键 | 类型 | 必填 | 默认值 | 说明 |
|----|------|------|--------|------|
| `format` | `str` | 是 | — | 格式名称（如 `"int8"`, `"fp4_e2m1"`, `"nf4"`），通过 registry 解析为 `FormatBase` |
| `granularity` | `str` | 是 | — | 粒度名称（`"per_tensor"`, `"per_channel"`, `"per_block"`） |
| `axis` | `int` | 否 | `-1` | PER_CHANNEL 和 PER_BLOCK 的 axis/block_axis 参数；PER_TENSOR 时被忽略 |
| `block_size` | `int` | PER_BLOCK 时必填 | — | PER_BLOCK 的块大小；非 PER_BLOCK 时被忽略 |
| `weight_only` | `bool` | 否 | `False` | `True` 时只量化 weight；`False` 时量化 input + weight + output |
| `transform` | `str` 或 `TransformBase` 或 `None` | 否 | `None` | 变换：`None` → `IdentityTransform`, `"hadamard"` → `HadamardTransform()`, `TransformBase` 实例直接传递 |

### 返回值

`OpQuantConfig`，其填充规则：

- `weight_only=False`（默认）: `OpQuantConfig(input=scheme, weight=scheme, output=scheme)` — 三个 forward compute 角色使用同一个 QuantScheme。
- `weight_only=True`: `OpQuantConfig(weight=scheme)` — 仅量化 weight；input 和 output 保持 `None`。
- `storage`: 始终为 `None`（descriptor 不表达存储量化；这是未来扩展点）。
- `bias`: 始终为 `None`（descriptor 不表达 bias 量化）。
- 所有 backward 角色: 始终为 `None`（descriptor 不表达 QAT 配置）。

> **为什么 input/weight/output 共享同一个 scheme？** 在搜索空间层面，一个"候选配置"是一个统一的 quantization scheme 设定 —— 它定义"用什么格式、粒度、变换来量化该层"。算子的不同角色是否用不同 scheme 是更高阶的配置问题，不在 descriptor 的覆盖范围内。未来可通过 `per_role_schemes` 等扩展键支持。

### Granularity 字符串 → GranularitySpec 映射

| 字符串值 | factory 调用 | 等价构造 |
|----------|-------------|----------|
| `"per_tensor"` | `GranularitySpec.per_tensor()` | `GranularitySpec(mode=PER_TENSOR, block_size=0, channel_axis=0, block_axis=-1)` |
| `"per_channel"` | `GranularitySpec.per_channel(axis)` | `GranularitySpec(mode=PER_CHANNEL, channel_axis=axis, block_size=0, block_axis=-1)` |
| `"per_block"` | `GranularitySpec.per_block(block_size, axis)` | `GranularitySpec(mode=PER_BLOCK, block_size=block_size, block_axis=axis, channel_axis=0)` |

**axis 默认值**: 当 descriptor 不包含 `axis` 键时，`per_channel` 使用 `channel_axis=-1`（last dim），`per_block` 使用 `block_axis=-1`（last dim）。PyTorch 风格的负索引支持在 `GranularitySpec` 层面原生支持，`resolve_config` 不做额外校验。

### Format 字符串 → FormatBase 映射

通过 `src/formats/registry.py::get_format()` 解析。支持的格式名（大小写不敏感）：

| 名称 | 类型 | 说明 |
|------|------|------|
| `"int8"` | IntFormat(bits=8) | 8-bit 整数格式 |
| `"int4"` | IntFormat(bits=4) | 4-bit 整数格式 |
| `"int2"` | IntFormat(bits=2) | 2-bit 整数格式 |
| `"fp8_e5m2"` | FPFormat(ebits=5, mbits=4) | FP8 高动态范围 |
| `"fp8_e4m3"` | FPFormat(ebits=4, mbits=5, max_norm=448.0) | FP8 高精度 |
| `"fp6_e3m2"` | FPFormat(ebits=3, mbits=4) | FP6 |
| `"fp6_e2m3"` | FPFormat(ebits=2, mbits=5) | FP6 |
| `"fp4_e2m1"` | FPFormat(ebits=2, mbits=3) | FP4 / 别名 `"fp4"` |
| `"nf4"` | NF4Format() | NormalFloat4 (QLoRA) |
| `"float16"` | Float16Format() | IEEE FP16 / 别名 `"fp16"` |
| `"bfloat16"` | BFloat16Format() | BFloat16 / 别名 `"bf16"` |

### Transform 解析

| descriptor 中的值 | 解析结果 |
|-------------------|----------|
| `None` (缺失) | `IdentityTransform()` |
| `"hadamard"` | `HadamardTransform()` |
| `TransformBase` 实例 | 保持原实例不变 |

## 期望行为

### 场景 1: 基础 — int8 per_tensor

```python
descriptor = {"format": "int8", "granularity": "per_tensor"}
cfg = resolve_config(descriptor)
```

期望：
- `cfg.weight` 是 `QuantScheme(format=Int8Format(), granularity=GranularitySpec.per_tensor(), transform=IdentityTransform())`
- `cfg.input` 与 `cfg.weight` 值相等（值相等，不是同一对象引用）
- `cfg.output` 与 `cfg.weight` 值相等
- `cfg.storage` 为 `None`
- `cfg.bias` 为 `None`
- 所有 backward 字段为 `None`
- `cfg.is_training` 为 `False`

### 场景 2: per_channel 带 axis

```python
descriptor = {"format": "int8", "granularity": "per_channel", "axis": 0}
cfg = resolve_config(descriptor)
```

期望：
- `cfg.weight.granularity == GranularitySpec(mode=PER_CHANNEL, channel_axis=0, ...)`
- `cfg.weight.granularity.channel_axis == 0`
- 其余行为同场景 1

### 场景 3: per_channel 默认 axis (-1)

```python
descriptor = {"format": "int8", "granularity": "per_channel"}
cfg = resolve_config(descriptor)
```

期望：
- `cfg.weight.granularity.channel_axis == -1`
- 其余行为同场景 1

### 场景 4: per_block MX

```python
descriptor = {"format": "fp4_e2m1", "granularity": "per_block", "block_size": 32}
cfg = resolve_config(descriptor)
```

期望：
- `cfg.weight.granularity == GranularitySpec(mode=PER_BLOCK, block_size=32, block_axis=-1)`
- `cfg.weight.format.name == "fp4_e2m1"`
- 其余行为同场景 1

### 场景 5: per_block 指定 block_axis

```python
descriptor = {"format": "fp4_e2m1", "granularity": "per_block", "block_size": 32, "axis": 0}
cfg = resolve_config(descriptor)
```

期望：
- `cfg.weight.granularity.block_axis == 0`
- `cfg.weight.granularity.block_size == 32`

### 场景 6: weight_only 模式

```python
descriptor = {"format": "nf4", "granularity": "per_channel", "axis": 0, "weight_only": True}
cfg = resolve_config(descriptor)
```

期望：
- `cfg.weight` 是 `QuantScheme(format=NF4Format(), granularity=GranularitySpec.per_channel(0), ...)`
- `cfg.input` 为 `None`
- `cfg.output` 为 `None`
- `cfg.storage` 为 `None`
- 所有 backward 字段为 `None`

### 场景 7: 带 Hadamard transform

```python
descriptor = {"format": "int4", "granularity": "per_tensor", "transform": "hadamard"}
cfg = resolve_config(descriptor)
```

期望：
- `cfg.weight.transform` 是 `HadamardTransform` 实例
- `cfg.weight.transform == HadamardTransform()` 为 `True`

### 场景 8: TransformBase 实例直接传入

```python
from src.transform.hadamard import HadamardTransform
descriptor = {"format": "int8", "granularity": "per_tensor", "transform": HadamardTransform()}
cfg = resolve_config(descriptor)
```

期望：
- `cfg.weight.transform` 是传入的 `HadamardTransform` 实例
- `cfg.weight.transform == HadamardTransform()` 为 `True`

### 场景 9: descriptor 无 transform（默认 Identity）

```python
descriptor = {"format": "int8", "granularity": "per_tensor"}
# 不传递 transform 键
cfg = resolve_config(descriptor)
```

期望：
- `cfg.weight.transform` 是 `IdentityTransform` 实例
- `cfg.weight.transform == IdentityTransform()` 为 `True`

### 场景 10: nf4 weight_only per_channel — QLoRA 风格

```python
descriptor = {"format": "nf4", "granularity": "per_channel", "axis": 0, "weight_only": True}
cfg = resolve_config(descriptor)
```

期望：
- `cfg.weight.format.name == "nf4"`
- `cfg.weight.granularity.mode == PER_CHANNEL`
- `cfg.weight.granularity.channel_axis == 0`
- `cfg.input` 为 `None`
- `cfg.output` 为 `None`

### 异常与守卫

| 场景 | 期望异常 | 说明 |
|------|---------|------|
| 未知 format 字符串（如 `"mxint4"`） | `KeyError` | 字符串在 registry 中找不到对应 FormatBase |
| 未知 granularity 字符串（如 `"per_group"`） | `ValueError` | 非 "per_tensor"/"per_channel"/"per_block" 的值 |
| per_block 时缺失 `block_size` | `KeyError` | 必填键不存在：descriptor 中有 `"granularity": "per_block"` 但没有 `"block_size"` |
| format 字段缺失 | `KeyError` | descriptor 中无 `"format"` 键 |
| granularity 字段缺失 | `KeyError` | descriptor 中无 `"granularity"` 键 |
| `weight_only` 类型错误（如 `"true"` 字符串） | `TypeError` | `weight_only` 必须是 `bool` | 
| `format` 字段非字符串（如 `int`） | `TypeError` | format 必须是 `str` |
| `granularity` 字段非字符串 | `TypeError` | granularity 必须是 `str` |
| `axis` 非 int | `TypeError` | axis 必须是 `int` |
| `block_size` 非 int | `TypeError` | block_size 必须是 `int` |
| `transform` 非法字符串（非 `"hadamard"`） | `ValueError` | 未知的 transform 名称 |
| `transform` 非 str/TransformBase/None | `TypeError` | transform 必须是 str、TransformBase 或 None |

### 不可变性与幂等性

- `resolve_config()` 不修改输入的 descriptor dict
- 相同 descriptor 的多次调用返回值相等的 `OpQuantConfig` 对象（`==` 为 `True`），但非同一对象引用
- 返回值是 frozen dataclass，不可修改

### Descriptor 验证顺序（实现参考）

1. 检查 `"format"` 存在且为 `str`，否则 `KeyError` / `TypeError`
2. 通过 `get_format(fmt_str)` 解析 format，失败时 `KeyError`
3. 检查 `"granularity"` 存在且为 `str`，否则 `KeyError` / `TypeError`
4. 按 granularity 字符串分支：
   - `"per_tensor"`: `GranularitySpec.per_tensor()`，忽略 `axis` 和 `block_size`
   - `"per_channel"`: 读取 `axis`（默认 -1），校验类型；调用 `GranularitySpec.per_channel(axis)`
   - `"per_block"`: 读取 `block_size`（必填，否则 `KeyError`），校验类型；读取 `axis`（默认 `-1`）；调用 `GranularitySpec.per_block(block_size, axis)`
   - 其他: `ValueError`
5. 解析 `"transform"`（可选，默认 `None` → `IdentityTransform`）
   - `None`: `IdentityTransform()`
   - `"hadamard"`: `HadamardTransform()`
   - `TransformBase` 实例: pass through
   - 其他: `TypeError` / `ValueError`
6. 构造 `QuantScheme(format=fmt, granularity=g, transform=t)`
7. 读取 `"weight_only"`（可选，默认 `False`），校验类型
   - `False`: `OpQuantConfig(input=scheme, weight=scheme, output=scheme)`
   - `True`: `OpQuantConfig(weight=scheme)`

## 验证结果

- [ ] 运行日期: YYYY-MM-DD
- [ ] 结果: PASS / FAIL
- [ ] 说明
