# BANK + Sparse Static + Outlier Format — 实施计划

**日期**: 2026-05-14
**Branch**: `feature/refactor-src`
**ADR**: `docs/architecture/012-bank-sparse-static-outlier-format.md`

## 开发优先级

用户指定顺序：**Static → BANK → BANK+sparse → outlier_format**

> 替代方案（BANK first）在 ADR-012 中讨论。当前按用户优先级执行，在加 BANK 时补充 BANK+static 路径。

---

## Phase 1: Sparse 静态量化（Activation）

**目标**：删除 `NotImplementedError`，sparse 支持 `scale is not None` 的静态路径。mask 从 `torch.topk` 切换到 `abs(x) > threshold`。

### 1.1 FormatBase — 静态 sparse 方法

**文件**: `src/formats/base.py`

**`_quantize_per_tensor_sparse`**（改现有方法）：
- 当 `scale is not None` 时：scale shape 为 `(2,)`（`scale[0]=amax_o`, `scale[1]=amax_n`），额外接收 `threshold` 参数
- `mask = torch.abs(x) > threshold` 替代 `torch.topk`
- 两组各自用预存 scale 做 normalize → elemwise → rescale

**`_quantize_per_channel_sparse`**（改现有方法）：
- 静态时 scale shape 为 `(2, C, 1, ...)`，threshold shape 为 `(C, 1, ...)`
- 同样的 threshold-based mask

**`_quantize_per_block` → `_quantize_outlier_bank`**（改现有方法）：
- 静态时接收预存的 `scale_o`、`scale_n`（shared exponents）+ `threshold`
- 或者：per_block 的静态路径留后续（因为 per_block 稀疏的 scale_storage 当前死参数问题）

### 1.2 quantize() 接口扩展

**文件**: `src/quantize/elemwise.py`

`quantize()` 新增可选参数：
- `sparse_threshold: Optional[torch.Tensor] = None`：静态稀疏的 threshold
- 透传到 `format.quantize(..., threshold=sparse_threshold)`

### 1.3 FormatBase.quantize() dispatch

**文件**: `src/formats/base.py`

`quantize()` 方法签名新增 `threshold=None` 参数，透传到 sparse 子方法。

### 1.4 Calibration Pipeline — 记录 sparse threshold + scales

**文件**: `src/calibration/pipeline.py`

当 `static_input_scale=True` 且 `outlier_ratio > 0` 时：
- 在 hook 中额外记录：每个 granularity group 的 k-th magnitude（threshold）
- 存储 `_input_scale_o`、`_input_scale_n`、`_input_sparse_threshold` buffer 到模块

### 1.5 Session 集成

**文件**: `src/session/_session.py`

`run_quantization()` 中，校准后将 sparse 参数注入模块 buffer，推理时自动使用静态路径。

### 1.6 测试

- `scale is not None` + `outlier_ratio > 0` 不再 raise
- threshold-based mask 与 topk-based mask bit-exact 验证（当 threshold = 第 k 个 magnitude 时）
- 静态 sparse 的 shape 测试
- Session 集成：`static_input_scale=True` + `outlier_ratio > 0` 端到端

---

## Phase 2: BANK 粒度

**目标**：新增 `GranularityMode.BANK`，支持按 bank_axis 切分为粗粒度 scale 组。

### 2.1 GranularityMode + GranularitySpec

**文件**: `src/scheme/granularity.py`

```python
class GranularityMode(Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_BLOCK = "per_block"
    BANK = "bank"                    # NEW
    DYNAMIC_GROUP = "dynamic_group"
```

`GranularitySpec` 新增：
- `bank_size: int = 16`
- `bank_axis: int = -1`（默认与 a_axis/w_axis 同步）
- `__post_init__` 校验：`BANK` 要求 `bank_size > 0`，`channel_axis=0`（BANK 不用 channel_axis）

### 2.2 `_quantize_per_bank()`

**文件**: `src/formats/base.py`

```python
def _quantize_per_bank(self, x, granularity, round_mode,
                       allow_denorm=True, scale=None, scale_storage="pot"):
```

逻辑：
1. 沿 `bank_axis` reshape 为 `(..., num_banks, bank_size, ...)`
2. 对每个 bank 做 `amax = max(abs(x)) over bank dims`（保持 bank dim 为非 reduction dim）
3. 支持 `scale_storage`（fp32/pot）
4. normalize → elemwise → rescale
5. reshape 回原形状

**BANK 的 reduction 语义**（与 PER_BLOCK 的关键差异）：
- 对于 M×N tensor，bank_axis=-1，bank_size=16：
  - reshape: `(M, N) → (M, N/16, 16)`
  - amax dims: `(0, -1)` — 跨 M 维 + bank 内部 16
  - 结果: N/16 个 scale，每个覆盖 M×16 个元素
- PER_BLOCK 则会得到 M×(N/32) 个 scale——BANK 不细分 M 维。

### 2.3 quantize() dispatch

**文件**: `src/formats/base.py`

```python
elif mode == GranularityMode.BANK:
    return self._quantize_per_bank(x, granularity, round_mode,
                                    allow_denorm=allow_denorm,
                                    scale=scale, scale_storage=scale_storage)
```

### 2.4 QuantConfig + _resolve_granularity

**文件**: `src/session/_config.py`

- `_VALID_GRANULARITIES` 加 `"bank"`
- `_resolve_granularity()` 加 bank 分支，`bank_axis` 与 `axis` 参数同步
- `__post_init__` 校验：`"bank"` 时 bank_size 可用（可通过已有 w_block_size/a_block_size 参数，或新增 w_bank_size/a_bank_size）

> 注意：bank_size 和 block_size 复用同一个参数字段还是独立？建议当前复用（两者不共存），后续如有需要可分拆。

### 2.5 测试

- BANK 构造测试（bank_size/bank_axis 校验）
- bit-exact 测试（手算 BANK amax + quantize 期望值）
- BANK vs PER_BLOCK(axis=-1) 行为差异验证（scale 数量和值不同）
- 各种 shape (2D/3D/4D) 保持
- QuantConfig → Session 端到端

---

## Phase 3: BANK + sparse

**目标**：BANK 粒度支持 `outlier_ratio > 0`。

### 3.1 `_quantize_per_bank_sparse()`

**文件**: `src/formats/base.py`

逻辑（与 `_quantize_per_channel_sparse` 结构相同，group 定义变为 bank）：
1. 沿 bank_axis reshape
2. 每个 bank 内 topk（动态）或 threshold compare（静态）
3. 两组各自 amax → normalize → elemwise → rescale
4. mask merge → reshape 回原形状

### 3.2 quantize() dispatch

BANK + `outlier_ratio > 0` 分支调度到 `_quantize_per_bank_sparse`。

### 3.3 测试

- BANK+sparse bit-exact（手算）
- BANK+sparse 退化（k >= bank 内元素数）
- BANK+sparse static 路径
- Session 集成

---

## Phase 4: 可配置 Outlier Format

**目标**：sparse 的 outlier 组可以使用与 normal 组不同的格式。

### 4.1 QuantScheme 扩展

**文件**: `src/scheme/quant_scheme.py`

```python
@dataclass(frozen=True)
class QuantScheme:
    format: FormatBase
    granularity: GranularitySpec
    transform: TransformBase
    round_mode: str
    scale_storage: str
    outlier_format: Optional[FormatBase] = None  # NEW — None = use self.format
```

### 4.2 QuantConfig 扩展

**文件**: `src/session/_config.py`

```python
outlier_format: Optional[str] = None       # None = 用主格式
a_outlier_format: Optional[str] = None     # None = 跟随 outlier_format
```

`to_op_config()` 中解析 `outlier_format` → `FormatBase`，设置到 `QuantScheme`。

### 4.3 sparse 方法接收 outlier_format

**文件**: `src/formats/base.py`、`src/formats/_outlier_utils.py`

所有 sparse 方法（4 个）新增 `outlier_format=None` 参数：
- `outlier_format is None`：用 `self`（当前行为）
- 非 None：outlier 组用 `outlier_format.quantize_elemwise()`，normal 组仍用 `self.quantize_elemwise()`

### 4.4 透传路径

```
QuantConfig.outlier_format
  → QuantScheme.outlier_format
    → quantize() in elemwise.py（读出 scheme.outlier_format）
      → format.quantize(..., outlier_format=scheme.outlier_format)
        → _quantize_X_sparse(..., outlier_format=outlier_format)
```

### 4.5 测试

- `outlier_format=None` 向后兼容（与当前行为 bit-exact）
- `outlier_format="int8"` + 主格式 int4：outlier 组用 int8 量化
- `a_outlier_format` 独立覆盖测试
- QuantConfig → Session 端到端

---

## 涉及文件总览

| 文件 | P1 | P2 | P3 | P4 | 改动类型 |
|------|:--:|:--:|:--:|:--:|---------|
| `src/scheme/granularity.py` | | x | | | 新增 BANK mode + bank_size/bank_axis |
| `src/formats/base.py` | x | x | x | x | 静态 sparse + bank + bank sparse + outlier_format |
| `src/formats/_outlier_utils.py` | x | | | x | 静态 per_block sparse + outlier_format |
| `src/quantize/elemwise.py` | x | | | x | threshold 参数 + outlier_format 透传 |
| `src/scheme/quant_scheme.py` | | | | x | outlier_format 字段 |
| `src/session/_config.py` | | x | | x | BANK resolve + outlier_format/a_outlier_format |
| `src/calibration/pipeline.py` | x | | | | 静态 sparse threshold/scale 记录 |
| `src/session/_session.py` | x | | | | 静态 sparse buffer 注入 |

---

## 验证文档

| Phase | 验证文档 | 内容 |
|-------|---------|------|
| P1 | `docs/verification/020-sparse-static.md` | 静态 sparse 数学推导（per_tensor/per_channel） |
| P2 | `docs/verification/021-bank-granularity.md` | BANK 量化数学推导 |
| P3 | `docs/verification/022-bank-sparse.md` | BANK + sparse 数学推导 |

---

## E2E 回归门

每个 Phase 完成后必须通过：
- `pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q -m "not slow"`
- `PYTHONPATH=. python scripts/mnist_hadamard_study.py`
- `PYTHONPATH=. python scripts/transformer_agnews_study.py`
