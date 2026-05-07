# Framework Review Report — 2026-05-07

**审视范围**: `src/` 全部包的 API 设计、配置流、命名一致性、死代码、边界违规
**审视者**: 软件架构专家视角
**分支**: `feature/refactor-src`
**测试门**: 1,712 passed
**状态**: All 13 issues resolved (P0–P2)

---

## 总体评价

框架的三轴正交设计（format × granularity × transform）和四层依赖架构（Math → Ops → Integration → Tools）是扎实的。Session 2.0 分层 API（`.quantize() → .calibrate() → .analyze() → .evaluate() → .cost()`）设计合理，Output-Driven Observer 选择模式优雅。以下问题按严重程度排列。

---

## 严重问题 (Critical)

### C1 — Descriptor 反序列化路径三份实现，行为不一致

三个不同的函数做同一件事（dict → 配置对象），每个有不同的默认值、不同的字段支持和不同的输出结构：

| 函数 | 位置 | 使用场景 | 设置的 OpQuantConfig 角色 |
|------|------|---------|--------------------------|
| `OpQuantConfig.from_descriptor()` | `scheme/op_config.py:73` | 仅测试 | `input`, `weight`, `output` |
| `QuantConfig.to_op_config()` | `session/_config.py:228` | 生产 | `input`, `weight` (无 `output`) |
| `resolve_config()` | `session/_config.py:390` | 后向兼容 | `input`, `weight` (无 `output`) |

具体不一致：
- `OpQuantConfig.from_descriptor()` 允许 `transform=None` 表示 IdentityTransform，`resolve_config()` 和 `QuantConfig` 强制用 `"none"` 字符串
- `OpQuantConfig.from_descriptor()` 会设置 `output` 角色，`QuantConfig.to_op_config()` 不设置——用户无法通过高层 API 配置 output-side compute 量化
- `OpQuantConfig.from_descriptor()` 没有 `bfloat`/`fp` storage 支持，`QuantConfig` 有

**建议**: 删除 `OpQuantConfig.from_descriptor()`（它仅在测试中使用），统一到 `QuantConfig.from_descriptor()` → `QuantConfig.to_op_config()` 单一路径。

---

### C2 — `study_config.py` 配置字段与 `QuantConfig` 不同步

`STUDY_CONFIG` 是候选的公开 API（从 `src.session` 导出），但包含 QuantConfig 不支持的字段：

1. **`lsq_pot`** (part_c): `QuantConfig` 无此字段，`QuantConfig.from_descriptor()` 也不处理——静默丢弃
2. **`part_hierarchical` 缺少 `transform`**: 使用 `pre_scale_init` / `pre_scale_pot` 但未设 `transform: "prescale"`——prescale 永远不会被激活，这些实验等同于 baseline

```python
# part_hierarchical 中的实际条目——没有 transform 字段
{"name": "MXINT-8-HIER", "format": "int8", "granularity": "per_block",
 "lsq_steps": 0, "pre_scale_init": "pot_amax", "pre_scale_pot": True}
#                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                         这些字段无 transform="prescale" 不会生效
```

**建议**: 要么将 `STUDY_CONFIG` 移入 `tests/` 作为测试固定装置、删除公开导出，要么使其字段与 QuantConfig 保持同步并添加 schema 验证。

---

### C3 — `to_op_config()` 缺少 output / bias compute 配置

`QuantConfig.to_op_config()` 只设置 `input` 和 `weight` 计算方案：

```python
# session/_config.py:276
return OpQuantConfig(input=a_scheme, weight=w_scheme, storage=storage)
```

但 `OpQuantConfig` 有 `output` 和 `bias` 字段。这意味着：
- 用户无法通过 `QuantConfig` 配置 output-side MX 量化
- Linear forward L108 (`if cfg.output: y = quantize(y, cfg.output, scale=output_scale)`) 对 Session API 用户永远不触发
- 这与 ADR-005 的例子直接矛盾（ADR-005 §Linear 表格明确有 `output: fp4 MX block`）

**建议**: `QuantConfig` 添加 `o_format` / `o_granularity` 或让 output 默认跟随 activation scheme。

---

## 重要问题 (Important)

### I1 — `bfloat` / `fp` 字段命名歧义

```python
bfloat: int = 0   # 0=disabled, 16=bfloat16 storage cast
fp: int = 0       # 0=disabled, 8=fp8 storage cast
```

问题是 `bfloat` 和 `fp` 在项目中同时也是 format 名称（`BFloat16Format`、`fp8_e4m3`）。用户看到 `bfloat=16` 容易理解为 "用 bfloat16 做权重量化"，但实际含义是 "对所有张量做 bfloat16 逐元素 storage cast"（storage 层）。

**建议**: 重命名为 `storage_bits` + `storage_kind`:
```python
storage_bits: int = 0          # 0 = disabled
storage_kind: str = "bfloat"   # "bfloat" | "fp"
```

---

### I2 — `_utils/` 单文件包违反项目自身规则

CLAUDE.md 明确规定 "单文件包是错的：一个包只有一个文件 → 包边界画错了 → 合并到所属概念"。`_utils/` 只有 `slicing.py` 一个文件。

**建议**: 将 `slicing.py` 合并入 `observer/` 或 `scheme/`（它的调用者都在这些包中）。

---

### I3 — `scale_storage` vs `scale_format` 命名不一致

| 位置 | 字段名 |
|------|--------|
| `QuantConfig` | `scale_storage: str` |
| `QuantScheme` | `scale_format: str` |
| Legacy descriptor | `scale_format` key |

同一个概念有三个名字。`scale_format` 与 QuantScheme 的 `format` 字段冲突（format 是数值格式，不是 scale 的存储格式）。`scale_storage` 更准确地描述了 "scale 值用什么精度存储"。

**建议**: 统一为 `scale_storage`。`QuantScheme.scale_format` → `QuantScheme.scale_storage`，legacy descriptor 同时接受两个 key。

---

### I4 — `prescale_granularity` 在所有 transform 上都被填充

```python
def __post_init__(self):
    if self.prescale_granularity is None:
        self.prescale_granularity = self.a_granularity  # 无条件执行
```

当 `transform="hadamard"` 或 `transform="none"` 时，`prescale_granularity` 仍被设置为有效值，但永远不会被使用。这浪费了字段语义——用户可能误以为设了 `prescale_granularity` 会生效。

**建议**: 仅在 `transform="prescale"` 时校验和填充，否则保持 `None`。

---

## 中等问题 (Moderate)

### M1 — `QuantSession` / `Session` 命名倒置

- `QuantSession` → 低层类，用户几乎不直接使用
- `Session` → 高层类，用户主要入口

直觉上 `QuantSession` 听起来比 `Session` 更具体，但它反而是更底层的原语。

**建议**: 考虑重命名 `QuantSession` → `_QuantSession`（标记为内部实现）或 `QuantEngine`，让 `Session` 成为唯一公共入口。

### M2 — `DYNAMIC_GROUP` 粒度不可通过 QuantConfig 访问

`GranularityMode.DYNAMIC_GROUP` 在 slicing、observer、calibration pipeline 中有实际实现，但 `_VALID_GRANULARITIES` 不包含它——用户无法通过 `QuantConfig` 使用此粒度模式。它只对低层 API 可用。

**建议**: 如果 `DYNAMIC_GROUP` 是计划中的功能，在 `QuantConfig` 的 `_VALID_GRANULARITIES` 中记录 `NotImplementedError`。如果是内部基础设施，添加注释说明。

### M3 — `QuantConfig` 缺少 `axis` 字段

`OpQuantConfig.from_descriptor()` 和 `resolve_config()` 通过 `axis` 支持非默认轴，但 `QuantConfig` 没有这个字段。`w_granularity="per_channel"` 始终使用 `axis=-1`，用户无法对 weight（通常 axis=0 才是特征维度）指定正确轴。

**建议**: 添加 `w_axis: int = -1` 和 `a_axis: int = -1`。

---

## 低优先级 (Low)

### L1 — `per_layer_optimal` 函数体中有与 Session 重复的逻辑

`per_layer_optimal()` 自己创建 `QuantSession`、运行 calibrate/analyze（`_per_layer_opt.py:132-200`），而不是复用 `Session` 类。这导致 calibrator 解析、observer 创建、SQ transform 修补等逻辑在两个地方重复。

**建议**: `per_layer_optimal` 内部使用 `Session` 实例替代直接操作 `QuantSession`。

### L2 — `_VALID_ROUND_MODES` 重复定义

`scheme/quant_scheme.py:12` 和 `formats/base.py:11` 各定义了一次 `_VALID_ROUND_MODES`，值相同但独立维护。

**建议**: 从单一定义导出（`scheme.quant_scheme` → `formats.base` 或提取到常量模块）。

### L3 — `session/study_config.py` 文档仍是中文

与其他英文模块文档不一致。项目约定是英文（所有 ADR、模块文档均为英文），但 `study_config.py` 的 docstring 和注释是中文。

---

## 优势确认

以下设计决策经审视确认是正确的：

1. **三轴正交设计**: `QuantScheme = FormatBase × GranularitySpec × TransformBase` 干净，扩展新格式/transform 不改核心
2. **OpQuantConfig 两阶段模型**: storage（统一 elemwise）+ compute（per-role）消除 tuple pipeline 过度设计
3. **Output-Driven Observer**: 用户声明输出 → 系统推导 Observer，Observer 不暴露，设计正确
4. **格式注册表惰性初始化 + 线程安全**: `_ensure_initialized()` 双重检查锁定，`auto-parse` 从命名约定自动创建格式
5. **Lazy import 解决 session↔report 循环**: ADR-008 §5.2 的方案标准且有效
6. **Type guard 测试覆盖**: `__post_init__` 验证 + `pytest.raises` 匹配模式在 `test_quant_config.py` 中充分覆盖
7. **Frozen dataclass 安全**: `FormatBase._freeze()`, `OpQuantConfig(frozen=True)`, `QuantScheme(frozen=True)` 保证配置不可变
8. **SmoothQuant 权重融合**: `fuse_smoothquant_weights()` 将平滑 scale 吸收进权重，数学正确
9. **Per-block MX 的 calibrate() no-op**: `_needs_calibration()` 正确识别 MX per_block 并跳过——scales 动态计算

---

## 统计摘要

| 类别 | 数量 |
|------|------|
| 严重问题 | 3 |
| 重要问题 | 4 |
| 中等问题 | 3 |
| 低优先级 | 3 |
| **待修复总计** | **13** |
| 确认正确的设计决策 | 9 |

---

## 修复优先级建议

**P0 (本分支修复)**:
- C1: 删除 `OpQuantConfig.from_descriptor()`，统一反序列化路径
- C2: 修复 `STUDY_CONFIG` 的 `part_hierarchical` 缺失 transform 和 `lsq_pot` 不支持

**P1 (P7 前修复)**:
- C3: `to_op_config()` 支持 output compute
- I1: 重命名 `bfloat`/`fp` 字段
- I3: 统一 `scale_storage`/`scale_format` 命名
- M3: `QuantConfig` 添加 `axis` 字段

**P2 (下个重构周期)**:
- I2: `_utils/` 单文件包
- I4: `prescale_granularity` 条件填充
- M1: 命名倒置
- L1-L3: 重复逻辑、重复常量、文档语言
