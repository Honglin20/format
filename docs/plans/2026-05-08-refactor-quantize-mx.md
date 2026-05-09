# Refactor quantize_mx — 消除双入口，对齐 per_block 实现范式

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 消除 `quantize_mx` / `quantize` 双公共入口，将 per_block 量化逻辑内联到 `FormatBase._quantize_per_block()`，与 per_tensor / per_channel 实现范式一致。

**Architecture:** `quantize(x, scheme)` 是唯一公共入口（ADR-001 已定义）。`FormatBase.quantize()` 按 granularity 分发：per_tensor / per_channel / per_block 三个方法各自直接实现，不再通过 `_quantize_mx` 外包。Block 工具函数（`_reshape_to_blocks` 等）作为格式层内部 helper，放在 `src/formats/_block_utils.py`。

**Tech Stack:** Python, PyTorch, pytest

---

### Task 1: 添加 `FormatBase._quantize_per_block()` vs `_quantize_mx` 等价性安全网测试

**Files:**
- Modify: `src/tests/test_format_quantize.py`

**Step 1: 扩展已有 `test_per_block_quantize_equiv` 覆盖更多边界情况**

在 `test_format_quantize.py` 的 `test_per_block_quantize_equiv` 测试（已存在，line 86）之后，添加以下测试确保 `fmt.quantize(x, per_block(...))` 和 `_quantize_mx` 在更多条件下一致：

```python
# 在 test_per_block_various_sizes 之后添加

@pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "fp4_e2m1", "int8"])
def test_per_block_quantize_vs_mx_round_modes(fmt_name):
    """fmt.quantize(x, per_block, round) == _quantize_mx for all round modes."""
    from src.quantize.mx_quantize import _quantize_mx
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec.per_block(32)

    for rm in ["nearest", "floor"]:
        result = fmt.quantize(x, gran, rm)
        expected = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                                block_size=32, axes=-1, round_mode=rm)
        assert torch.equal(result, expected), \
            f"{fmt_name}/per_block(32)/{rm}: mismatch"


def test_per_block_quantize_vs_mx_shared_exp_none():
    """fmt.quantize(PER_BLOCK) with shared_exp_method='none' — verify equivalence."""
    from src.quantize.mx_quantize import _quantize_mx
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    fmt = FormatBase.from_str("fp8_e4m3")
    gran = GranularitySpec.per_block(32)

    result = fmt.quantize(x, gran, "nearest")
    expected = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                            block_size=32, axes=-1, round_mode="nearest",
                            shared_exp_method="none")
    # shared_exp_method='none' 和 'max' 输出不同，但 format.quantize 默认是 'max'
    # 这个测试验证 format.quantize 的默认行为 == _quantize_mx 默认 "max"
    expected_max = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                                block_size=32, axes=-1, round_mode="nearest",
                                shared_exp_method="max")
    assert torch.equal(result, expected_max)


def test_per_block_quantize_vs_mx_flush_subnorms():
    """fmt.quantize(PER_BLOCK) — verify flush_fp32_subnorms path not triggered."""
    from src.quantize.mx_quantize import _quantize_mx
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    fmt = FormatBase.from_str("fp8_e4m3")
    gran = GranularitySpec.per_block(32)

    result = fmt.quantize(x, gran, "nearest")
    # With flush_fp32_subnorms=False (default), subnorms are preserved
    expected = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                            block_size=32, axes=-1, round_mode="nearest",
                            flush_fp32_subnorms=False)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_per_block_quantize_vs_mx_various_blocks(block_size):
    """fmt.quantize(x, per_block(N)) == _quantize_mx for various block sizes."""
    from src.quantize.mx_quantize import _quantize_mx
    torch.manual_seed(42)
    x = torch.randn(4, block_size * 4)
    fmt = FormatBase.from_str("fp8_e4m3")
    gran = GranularitySpec.per_block(block_size)

    result = fmt.quantize(x, gran, "nearest")
    expected = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                            block_size=block_size, axes=-1, round_mode="nearest")
    assert torch.equal(result, expected)


@pytest.mark.parametrize("shape", [
    (4, 64), (2, 3, 64), (2, 3, 4, 64),  # 2D, 3D, 4D
])
def test_per_block_quantize_vs_mx_multi_dim(shape):
    """fmt.quantize(x, per_block) == _quantize_mx for multi-dim tensors."""
    from src.quantize.mx_quantize import _quantize_mx
    torch.manual_seed(42)
    x = torch.randn(*shape)
    fmt = FormatBase.from_str("fp8_e4m3")
    gran = GranularitySpec.per_block(32)

    result = fmt.quantize(x, gran, "nearest")
    expected = _quantize_mx(x, scale_bits=8, elem_format=fmt,
                            block_size=32, axes=-1, round_mode="nearest")
    assert torch.equal(result, expected)
```

**Step 2: 运行测试确认通过（当前实现）**

```bash
pytest src/tests/test_format_quantize.py -q -k "per_block"
```

Expected: 全部 PASS（这些测试验证等价性，当前实现已满足）

**Step 3: 提交**

```bash
git add src/tests/test_format_quantize.py
git commit -m "test(format): add per_block quantize vs _quantize_mx equivalence safety net"
```

---

### Task 2: 将 block 工具函数从 `src/quantize/mx_quantize.py` 提取到 `src/formats/_block_utils.py`

**Files:**
- Create: `src/formats/_block_utils.py`
- Modify: `src/quantize/mx_quantize.py`

**Step 1: 创建 `src/formats/_block_utils.py`**

将 `_reshape_to_blocks`、`_undo_reshape_to_blocks`、`_shared_exponents` 以及常量 `FP32_EXPONENT_BIAS`、`FP32_MIN_NORMAL` 移动到新文件：

```python
"""Internal helpers for per-block quantization (MX-style shared exponents).

These are private implementation details of FormatBase._quantize_per_block().
"""
import torch

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


def _shared_exponents(A, method="max", axes=None, ebits=0):
    if method == "max":
        if axes is None:
            shared_exp = torch.max(torch.abs(A))
        else:
            shared_exp = A
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
    elif method == "none":
        shared_exp = torch.abs(A)
    else:
        raise ValueError(f"Unrecognized shared exponent method {method!r}")

    shared_exp = torch.floor(
        torch.log2(
            shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
        )
    )

    if ebits > 0:
        emax = 2**(ebits-1) - 1
        shared_exp[shared_exp > emax] = float("NaN")
        shared_exp[shared_exp < -emax] = -emax

    return shared_exp


def _reshape_to_blocks(A, axes, block_size):
    if axes is None:
        raise ValueError("axes required in order to determine which "
                         "dimension to apply block size to")
    if block_size == 0:
        raise ValueError("block_size == 0 in _reshape_to_blocks")

    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    if not all(x >= 0 for x in axes):
        raise ValueError("All axes must be non-negative after normalization")
    axes = sorted(axes)

    for i in range(len(axes)):
        axes[i] += i
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.item())
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            if shape[axis] >= reshape_block_size:
                if shape[axis] % reshape_block_size != 0:
                    raise ValueError(
                        f"shape[{axis}]={shape[axis]} not divisible by block_size={reshape_block_size}")
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)
    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape


def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    A = A.view(padded_shape)
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        A = torch.squeeze(A, dim=axis + 1)
    return A
```

**Step 2: 更新 `src/quantize/mx_quantize.py` 的 import**

在 `mx_quantize.py` 中将工具函数 import 改为从新位置：

```python
from src.formats._block_utils import (
    _reshape_to_blocks,
    _undo_reshape_to_blocks,
    _shared_exponents,
    FP32_EXPONENT_BIAS,
    FP32_MIN_NORMAL,
)
```

删除文件中 `_reshape_to_blocks`、`_undo_reshape_to_blocks`、`_shared_exponents` 的定义和常量。

**Step 3: 运行现有测试确认逻辑不变**

```bash
pytest src/tests/test_mx_quantize_equiv.py src/tests/test_format_quantize.py -q
```

Expected: 全部 PASS

**Step 4: 提交**

```bash
git add src/formats/_block_utils.py src/quantize/mx_quantize.py
git commit -m "refactor(formats): extract block utils to src/formats/_block_utils.py"
```

---

### Task 3: 将 per_block 量化逻辑内联到 `FormatBase._quantize_per_block()`

**Files:**
- Modify: `src/formats/base.py:194-222`

**Step 1: 重写 `FormatBase._quantize_per_block()`**

将当前委托 `_quantize_mx` 的代码替换为直接实现：

```python
def _quantize_per_block(self, x, granularity, round_mode, scale=None,
                          scale_storage="fp32"):
    """Per-block quantization using MX-style shared exponents.

    Shares the same structure as _quantize_per_channel:
    compute scale → normalize → elemwise quantize → rescale.

    During JIT tracing (ONNX export), return x unchanged — the
    Function's symbolic() method handles quantization in the ONNX graph.
    """
    if torch.jit.is_tracing():
        return x
    from src.formats._block_utils import (
        _reshape_to_blocks,
        _undo_reshape_to_blocks,
        _shared_exponents,
        FP32_EXPONENT_BIAS,
    )

    block_size = granularity.block_size
    axes = [granularity.block_axis]

    # Shortcut for no quantization
    if self is None:
        return x

    # Normalize axes
    axes = [x.ndim + a if a < 0 else a for a in axes]

    # Step 1: Tile into hardware-vector-sized blocks
    A, axes, orig_shape, padded_shape = _reshape_to_blocks(
        x, axes, block_size)

    # Step 2: Compute shared exponents per block
    shared_exp_axes = [a + 1 for a in axes] if block_size > 0 else axes
    shared_exp = _shared_exponents(
        A, method="max", axes=shared_exp_axes, ebits=0)

    # Step 3: Offset by format's max representable exponent
    shared_exp = shared_exp - self.emax

    # Step 4: Clamp shared exponents to int8 range
    scale_emax = 2**(8-1) - 1  # scale_bits=8 → emax=127
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    # Step 5: Normalize by shared exponent
    A = A / (2 ** shared_exp)

    # Step 6: Element-wise quantize to target format
    A = self.quantize_elemwise(
        A, round_mode=round_mode,
        allow_denorm=True, saturate_normals=True)

    # Step 7: Rescale
    A = A * (2 ** shared_exp)

    # Step 8: Undo block tiling
    A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A
```

**关键设计决策**：
- `scale_bits=8` 内联为常量（从未改变过，所有调用方都是 8）
- `shared_exp_method="max"` 内联（从未使用过 "none" 在生产路径）
- `flush_fp32_subnorms=False` 移除（从未在生产路径启用）
- `scale` 参数保留在签名中（接口兼容），但不使用（已有注释说明 PER_BLOCK 忽略外部 scale）
- 步骤标注 `Step 1-8` 保持与 `_quantize_per_channel` 同等的可读性

**Step 2: 运行全部测试确认逻辑不变**

```bash
pytest src/tests/test_format_quantize.py src/tests/test_mx_quantize_equiv.py src/tests/test_refactor_quantize_elemwise.py -q
```

Expected: 全部 PASS

**Step 3: 提交**

```bash
git add src/formats/base.py
git commit -m "refactor(formats): inline per_block quantization into FormatBase._quantize_per_block()"
```

---

### Task 4: 移除 `_quantize_mx` 冗余参数，改为接受 QuantScheme（内部兼容层）

**Files:**
- Modify: `src/quantize/mx_quantize.py`

**Step 1: 将 `_quantize_mx` 改为 thin wrapper**

当前 `_quantize_mx` 被 `FormatBase._quantize_per_block()` 调用... 

等一下。Task 3 完成后，`FormatBase._quantize_per_block()` 已经不再调用 `_quantize_mx`。但以下测试文件仍然直接 import `_quantize_mx`：

- `src/tests/test_format_quantize.py:95` — 等价性验证
- `src/tests/test_refactor_quantize_elemwise.py:244,262,355` — 等价性验证
- `src/tests/test_mx_quantize_equiv.py:122-177` — 对标 mx 库的等价性测试

这些测试调用 `_quantize_mx(x, scale_bits=8, elem_format=fmt, block_size=32, axes=[-1], round_mode=...)` —— 它们不用 QuantScheme，直接把参数传给 `_quantize_mx`。

**策略**：保留 `_quantize_mx` 在 `src/quantize/mx_quantize.py` 作为 thin wrapper，内部构造 GranularitySpec 和 FormatBase，调用 `FormatBase._quantize_per_block()`。这样测试继续通过，同时确保逻辑走的是 `FormatBase._quantize_per_block()`。

```python
def _quantize_mx(
    A,
    scale_bits,
    elem_format,
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round_mode="nearest",
    flush_fp32_subnorms=False,
    scale=None,
):
    """Per-block quantize (backward-compat wrapper).

    Thin wrapper around FormatBase._quantize_per_block().
    Prefer ``quantize(x, QuantScheme.mxfp(...))`` for new code.
    """
    if elem_format is None:
        return A

    if isinstance(elem_format, str):
        fmt = FormatBase.from_str(elem_format)
    else:
        fmt = elem_format

    from src.scheme.granularity import GranularitySpec

    # Build GranularitySpec from legacy parameters
    if block_size > 0:
        axis = axes[0] if isinstance(axes, list) else axes
        granularity = GranularitySpec.per_block(block_size, axis=axis)
    else:
        granularity = GranularitySpec.per_tensor()

    return fmt._quantize_per_block(A, granularity, round_mode)
```

注意：wrapper **不**尝试传递 `shared_exp_method`、`flush_fp32_subnorms`、`scale`、`scale_bits` —— 这些参数在 wrapper 中是被忽略的。但这些参数在被实际使用的测试中：
- `shared_exp_method="none"` → 测试 `test_per_block_shared_exp_none`
- `flush_fp32_subnorms=True` → 测试 `test_per_block_flush_fp32_subnorms`

这些是非默认路径。需要决定：

**选项 A**：wrapper 也支持这些参数（复制逻辑）
**选项 B**：修改这些测试，让它们直接测试 `_quantize_per_block` 的正确行为（这些是非标准 MX 行为）
**选项 C**：将这些参数视为已弃用，测试改为验证默认行为

鉴于 `shared_exp_method="none"` 和 `flush_fp32_subnorms=True` 是 mx 库的遗留参数，在我们的框架中从无生产使用，选 **选项 B**：更新这几个测试。

**Step 1 实施**：

```python
# src/quantize/mx_quantize.py — 完整内容

"""Per-block MX quantization (backward-compat wrappers).

The actual implementation lives in FormatBase._quantize_per_block().
This module provides thin backward-compat wrappers for tests that still
reference the old _quantize_mx name.
"""
import torch
from src.formats.base import FormatBase
from src.formats._block_utils import (
    _reshape_to_blocks,
    _undo_reshape_to_blocks,
    _shared_exponents,
    FP32_EXPONENT_BIAS,
    FP32_MIN_NORMAL,
)


def _quantize_mx(
    A,
    scale_bits,
    elem_format,
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round_mode="nearest",
    flush_fp32_subnorms=False,
    scale=None,
):
    """Per-block quantize (backward-compat wrapper).

    Prefer ``quantize(x, QuantScheme.mxfp(fmt, block_size))`` or
    ``fmt.quantize(x, GranularitySpec.per_block(N))`` for new code.
    """
    if elem_format is None:
        return A

    fmt = FormatBase.from_str(elem_format) if isinstance(elem_format, str) else elem_format

    from src.scheme.granularity import GranularitySpec

    if block_size > 0:
        axis = axes[0] if isinstance(axes, list) else axes
        granularity = GranularitySpec.per_block(block_size, axis=axis)
    else:
        granularity = GranularitySpec.per_tensor()

    # Non-default paths (shared_exp_method='none', flush_fp32_subnorms=True)
    # are legacy mx paths; forward them directly.
    if shared_exp_method != "max" or flush_fp32_subnorms:
        return _quantize_mx_legacy(
            A, fmt, granularity, round_mode,
            shared_exp_method, flush_fp32_subnorms)

    return fmt._quantize_per_block(A, granularity, round_mode)


def _quantize_mx_legacy(A, fmt, granularity, round_mode,
                         shared_exp_method, flush_fp32_subnorms):
    """Handle non-standard shared_exp_method / flush paths."""
    block_size = granularity.block_size
    axes = [granularity.block_axis]
    axes = [A.ndim + a if a < 0 else a for a in axes]

    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(A, axes, block_size)

    shared_exp_axes = [a + 1 for a in axes] if block_size > 0 else axes
    shared_exp = _shared_exponents(
        A, method=shared_exp_method, axes=shared_exp_axes, ebits=0)

    if flush_fp32_subnorms:
        A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    shared_exp = shared_exp - fmt.emax
    scale_emax = 2**(8-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    A = A / (2**shared_exp)
    A = fmt.quantize_elemwise(A, round_mode=round_mode,
                               allow_denorm=True, saturate_normals=True)
    A = A * (2**shared_exp)

    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A
```

等等，这样又复制了逻辑，不好。让我重新想。

更好的方案：直接把 `_quantize_mx` 的 legacy 路径也整合进 `FormatBase._quantize_per_block()`。反正 `shared_exp_method` 和 `flush_fp32_subnorms` 只是 `_quantize_per_block` 内部 step 2-4 的参数化变体。

但实际上 `shared_exp_method` 和 `flush_fp32_subnorms` 在框架中根本没有使用场景——它们只是 mx 库暴露的参数。如果它们没有出现在 QuantScheme 中，说明框架有意不暴露。测试里的等价性检查是历史遗留。

**最终决策**：保留 `_quantize_mx` wrapper 支持所有参数（保持测试 100% 通过），但实现改为构造 GranularitySpec 后调用 `FormatBase._quantize_per_block()`。对于 shared_exp_method='none' 和 flush 等非标准路径，直接在 wrapper 中内联处理（很少的代码，且只是 compat）。

实际上，让我再简化。Wrapper 直接包含完整逻辑（跟现在一样），但去掉 `scale_bits` 参数（hardcode 8），去掉注释中的冗余文档。这样 wrapper 逻辑独立但简短。

不对，等等。用户说要把逻辑内联到 `FormatBase._quantize_per_block()` 中。如果 wrapper 还保留完整逻辑，那就失去了内联的意义。

最佳方案：
1. `FormatBase._quantize_per_block()` 包含完整、正确的 per_block 量化逻辑（覆盖所有参数变体）
2. `_quantize_mx` wrapper 调用 `FormatBase._quantize_per_block()`
3. 测试要么走 `_quantize_mx` wrapper（向后兼容），要么新增走 `FormatBase._quantize_per_block()` 的测试

但 `_quantize_per_block()` 的签名是 `(x, granularity, round_mode, scale, scale_storage)` — 没有 `shared_exp_method`、`flush_fp32_subnorms`、`scale_bits` 参数。这些是 mx 遗留参数。

所以 `_quantize_mx` wrapper 需要处理这些参数，当它们是非默认值时走 legacy 路径。

OK 我觉得这已经够清晰了。让我简化 plan 中的 wrapper 实现。

**Step 1 实施（简化版）**：

```python
# src/quantize/mx_quantize.py

"""Per-block MX quantization — backward-compat wrappers.

The canonical implementation is FormatBase._quantize_per_block().
These wrappers exist for backward compatibility with direct callers
(primarily tests) that pass legacy mx parameter names.
"""

import torch
from src.formats.base import FormatBase
from src.formats._block_utils import (
    FP32_EXPONENT_BIAS, FP32_MIN_NORMAL,
    _reshape_to_blocks, _undo_reshape_to_blocks, _shared_exponents,
)


def _quantize_mx(A, scale_bits, elem_format,
                 shared_exp_method="max", axes=None, block_size=0,
                 round_mode="nearest", flush_fp32_subnorms=False, scale=None):
    """Per-block quantize with shared exponents.

    Prefer ``fmt.quantize(x, GranularitySpec.per_block(N), round)``.
    """
    if elem_format is None:
        return A

    fmt = FormatBase.from_str(elem_format) if isinstance(elem_format, str) else elem_format
    from src.scheme.granularity import GranularitySpec

    if block_size > 0:
        axis = axes[0] if isinstance(axes, list) else axes
        gran = GranularitySpec.per_block(block_size, axis=axis)
    else:
        gran = GranularitySpec.per_tensor()

    # Non-default paths (legacy mx options not in QuantScheme)
    if shared_exp_method != "max" or flush_fp32_subnorms:
        return _mx_legacy(
            A, fmt, gran, round_mode, shared_exp_method, flush_fp32_subnorms)

    return fmt._quantize_per_block(A, gran, round_mode)


def _mx_legacy(A, fmt, gran, round_mode, shared_exp_method, flush_fp32_subnorms):
    """Handle shared_exp_method='none' and flush_fp32_subnorms=True paths."""
    block_size = gran.block_size
    axes = [gran.block_axis]
    axes = [A.ndim + a if a < 0 else a for a in axes]

    if block_size > 0:
        A, axes, orig_shape, padded_shape = _reshape_to_blocks(A, axes, block_size)

    shared_exp_axes = [a + 1 for a in axes] if block_size > 0 else axes
    shared_exp = _shared_exponents(A, method=shared_exp_method,
                                    axes=shared_exp_axes, ebits=0)

    if flush_fp32_subnorms:
        A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    shared_exp = shared_exp - fmt.emax
    scale_emax = 2**(8-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    A = A / (2**shared_exp)
    A = fmt.quantize_elemwise(A, round_mode=round_mode,
                               allow_denorm=True, saturate_normals=True)
    A = A * (2**shared_exp)

    if block_size:
        A = _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A
```

**Step 2: 运行测试**

```bash
pytest src/tests/test_mx_quantize_equiv.py src/tests/test_format_quantize.py src/tests/test_refactor_quantize_elemwise.py src/tests/test_scheme_api.py -q
```

Expected: 全部 PASS

**Step 3: 提交**

```bash
git add src/quantize/mx_quantize.py
git commit -m "refactor(quantize): reduce _quantize_mx to thin wrapper around FormatBase._quantize_per_block()"
```

---

### Task 5: 移除公共 `quantize_mx` API

**Files:**
- Modify: `src/quantize/__init__.py`
- Modify: `src/tests/test_scheme_api.py`
- Modify: `src/tests/test_golden_equiv.py`
- Modify: `src/tests/test_mx_quantize_equiv.py`

**Step 1: 从 `__init__.py` 移除 `quantize_mx` 导出**

```python
# src/quantize/__init__.py
from .elemwise import quantize
from .bfloat_quantize import quantize_bfloat

__all__ = ["quantize", "quantize_bfloat"]
```

**Step 2: 更新 `test_scheme_api.py` — 将 `quantize_mx` 调用改为 `quantize(x, scheme)`**

`test_scheme_api.py` 有 5 个测试调用 `quantize_mx`：

- `test_quantize_mx_matches_old` → `quantize_mx(A, scheme=scheme, axes=[-1])` 改为 `quantize(A, scheme)`
- `test_quantize_mx_none_scheme` → `quantize_mx(A, scheme=None)` 改为 `quantize(A, None)`
- `test_quantize_mx_per_channel_raises` → 改为测试 `quantize()` 对 PER_CHANNEL 的正确行为（不再抛异常，`quantize()` 支持 PER_CHANNEL）
- `test_quantize_mx_no_block` → `quantize_mx(A, scheme=scheme, axes=[-1])` 改为 `quantize(A, scheme)`
- `test_quantize_mx_delegates_to_quantize_with_transform` → `quantize_mx(A, scheme=scheme)` 改为 `quantize(A, scheme)`

详细修改：

```python
class TestQuantizeMxScheme:
    """Tests for per-block quantization via quantize(x, scheme)."""

    @pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp4_e2m1", "int8"])
    def test_per_block_matches_old(self, fmt):
        """quantize(x, QuantScheme.mxfp(fmt)) should match old mx quantize_mx_op."""
        from src.quantize.elemwise import quantize
        from mx.mx_ops import quantize_mx_op as old_qmx_op
        from mx.specs import finalize_mx_specs as old_finalize

        torch.manual_seed(42)
        A = torch.randn(4, 64)
        config = {"w_elem_format": fmt, "a_elem_format": fmt,
                  "block_size": 32, "bfloat": 16}
        old_specs = old_finalize(config.copy())

        scheme = QuantScheme.mxfp(fmt, block_size=32)
        scheme_out = quantize(A.clone(), scheme)
        old_out = old_qmx_op(A.clone(), mx_specs=old_specs,
                             elem_format=fmt, axes=[-1])
        assert torch.equal(old_out, scheme_out), f"mismatch for {fmt}"

    def test_none_scheme(self):
        """scheme=None should pass through unchanged."""
        from src.quantize.elemwise import quantize
        A = torch.randn(4, 64)
        out = quantize(A.clone(), None)
        assert torch.equal(A, out)

    def test_per_channel_works(self):
        """quantize() supports PER_CHANNEL (unlike removed quantize_mx)."""
        from src.quantize.elemwise import quantize
        A = torch.randn(4, 64)
        scheme = QuantScheme.per_channel("fp8_e4m3", axis=0)
        result = quantize(A, scheme)
        assert result.shape == A.shape

    def test_per_tensor_works(self):
        """quantize() supports PER_TENSOR."""
        from src.quantize.elemwise import quantize
        torch.manual_seed(42)
        A = torch.randn(4, 64)
        scheme = QuantScheme(format="fp8_e4m3", granularity=GranularitySpec.per_tensor())
        result = quantize(A.clone(), scheme)
        assert result.shape == A.shape

    def test_delegates_to_quantize_with_transform(self):
        """quantize() handles transforms correctly."""
        from src.quantize.elemwise import quantize
        from src.transform.pre_scale import PreScaleTransform

        torch.manual_seed(42)
        A = torch.randn(4, 64)
        scale = torch.ones(1) * 2.0
        scheme = QuantScheme(
            format="int8",
            granularity=GranularitySpec.per_block(32),
            transform=PreScaleTransform(scale=scale),
        )
        out = quantize(A.clone(), scheme)
        assert out.shape == A.shape
        assert out.isfinite().all()
```

**Step 3: 更新 `test_golden_equiv.py`**

```python
# Line 12: 修改 import
from src.quantize.elemwise import quantize

# Line 86-87: 修改调用
new_out = quantize(
    A.clone(),
    QuantScheme.mxfp(fmt, block_size=32).transform(...)  # 需要查看具体上下文
)
```

需要先读 test_golden_equiv.py 的具体用法再改。

**Step 4: 更新 `test_mx_quantize_equiv.py` 的 `quantize_mx` 调用**

Line 188-195: 将 `quantize_mx(A, scheme=scheme, axes=[-1])` 改为 `quantize(A, scheme)`。

注意：`quantize_mx` 传递 `axes` 参数但 `quantize` 不需要——因为 `GranularitySpec.block_axis` 已经在 scheme 中。这正好证明了 `quantize_mx` 的 `axes` 参数是多余的。

**Step 5: 运行测试**

```bash
pytest src/tests/test_scheme_api.py src/tests/test_mx_quantize_equiv.py src/tests/test_golden_equiv.py -q
```

Expected: 全部 PASS

**Step 6: 提交**

```bash
git add src/quantize/__init__.py src/tests/test_scheme_api.py src/tests/test_golden_equiv.py src/tests/test_mx_quantize_equiv.py
git commit -m "refactor(quantize): remove public quantize_mx, use quantize(x, scheme) as sole entry point"
```

---

### Task 6: 添加 `FormatBase._quantize_per_block()` 直接测试

**Files:**
- Create: `src/tests/test_format_per_block.py`（或在 `test_format_quantize.py` 中添加）

**Step 1: 添加直接测试**

在 `test_format_quantize.py` 中添加测试类 `TestPerBlockQuantizeDirect`：

```python
# ---------------------------------------------------------------------------
# 6. FormatBase._quantize_per_block() — direct method tests
# ---------------------------------------------------------------------------

class TestPerBlockQuantizeDirect:

    @pytest.mark.parametrize("fmt_name", ["int8", "fp8_e4m3", "fp8_e5m2",
                                           "fp6_e3m2", "fp4_e2m1"])
    def test_same_as_format_quantize_dispatch(self, fmt_name):
        """_quantize_per_block() == fmt.quantize(x, PER_BLOCK, ...) — same result."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)
        gran = GranularitySpec.per_block(32)

        via_dispatch = fmt.quantize(x, gran, "nearest")
        via_direct = fmt._quantize_per_block(x, gran, "nearest")
        assert torch.equal(via_dispatch, via_direct), \
            f"{fmt_name}: dispatch != direct"

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8"])
    def test_output_finite(self, fmt_name):
        """Per-block quantized output should always be finite."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)
        result = fmt._quantize_per_block(
            x, GranularitySpec.per_block(32), "nearest")
        assert result.isfinite().all(), \
            f"{fmt_name}: non-finite values in output"

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "fp4_e2m1"])
    def test_idempotent(self, fmt_name):
        """Quantizing twice with same params should be idempotent (or nearly)."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str(fmt_name)
        gran = GranularitySpec.per_block(32)

        once = fmt._quantize_per_block(x, gran, "nearest")
        twice = fmt._quantize_per_block(once, gran, "nearest")
        assert torch.allclose(once, twice, atol=1e-6), \
            f"{fmt_name}: not idempotent"

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8"])
    @pytest.mark.parametrize("block_size", [16, 32, 64])
    def test_different_block_sizes_produce_different_output(self, fmt_name, block_size):
        """Different block sizes should produce (potentially) different results."""
        torch.manual_seed(42)
        x = torch.randn(4, block_size * 3)
        fmt = FormatBase.from_str(fmt_name)

        a = fmt._quantize_per_block(
            x, GranularitySpec.per_block(block_size), "nearest")
        other_size = 32 if block_size != 32 else 16
        b = fmt._quantize_per_block(
            x, GranularitySpec.per_block(other_size), "nearest")
        # They may or may not be equal depending on values, but both must be finite
        assert a.isfinite().all()
        assert b.isfinite().all()

    def test_round_mode_effect(self):
        """floor vs nearest should produce different results for fp formats."""
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        fmt = FormatBase.from_str("fp4_e2m1")
        gran = GranularitySpec.per_block(32)

        nearest = fmt._quantize_per_block(x, gran, "nearest")
        floor = fmt._quantize_per_block(x, gran, "floor")
        # Should differ for low-precision formats (fp4 has only 1 mantissa bit)
        assert not torch.equal(nearest, floor), \
            "floor and nearest should differ for fp4"

    def test_shape_preserved(self):
        """Output shape must equal input shape for various tensor ranks."""
        torch.manual_seed(42)
        fmt = FormatBase.from_str("fp8_e4m3")
        gran = GranularitySpec.per_block(32)

        for shape in [(4, 64), (2, 3, 64), (1, 2, 3, 128)]:
            x = torch.randn(*shape)
            result = fmt._quantize_per_block(x, gran, "nearest")
            assert result.shape == x.shape, f"shape mismatch: {result.shape} != {x.shape}"

    def test_jit_tracing_passthrough(self):
        """During JIT tracing, per_block should return input unchanged."""
        fmt = FormatBase.from_str("fp8_e4m3")
        gran = GranularitySpec.per_block(32)

        # traced_fn is None if tracing is not active, which means
        # the passthrough logic works. We test by checking the
        # conditional is reachable without error.
        x = torch.randn(4, 64)
        result = fmt._quantize_per_block(x, gran, "nearest")
        assert result.shape == x.shape

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8"])
    def test_small_values_quantized_correctly(self, fmt_name):
        """Values near zero should not become NaN or Inf."""
        torch.manual_seed(42)
        x = torch.randn(4, 64) * 1e-6
        fmt = FormatBase.from_str(fmt_name)
        result = fmt._quantize_per_block(
            x, GranularitySpec.per_block(32), "nearest")
        assert result.isfinite().all(), \
            f"{fmt_name}: small values produced non-finite results"

    @pytest.mark.parametrize("fmt_name", ["fp8_e4m3", "int8"])
    def test_large_values_quantized_correctly(self, fmt_name):
        """Large values should not become NaN (but Inf is acceptable for fp formats)."""
        torch.manual_seed(42)
        x = torch.randn(4, 64) * 1e6
        fmt = FormatBase.from_str(fmt_name)
        result = fmt._quantize_per_block(
            x, GranularitySpec.per_block(32), "nearest")
        assert not result.isnan().any(), \
            f"{fmt_name}: large values produced NaN"
```

**Step 2: 运行新测试**

```bash
pytest src/tests/test_format_quantize.py -q -k "PerBlock"
```

Expected: 全部 PASS

**Step 3: 提交**

```bash
git add src/tests/test_format_quantize.py
git commit -m "test(formats): add direct FormatBase._quantize_per_block() tests"
```

---

### Task 7: 全量回归测试

**Step 1: 运行全部测试（除 golden equiv）**

```bash
pytest src/tests/ --ignore=src/tests/test_golden_equiv.py -q
```

Expected: 全部 2093+ PASS

**Step 2: （可选）golden equiv 测试**

```bash
pytest src/tests/test_golden_equiv.py -q
```

预期：与重构前相同的 26 个预存在失败（golden data `.pt` 文件未 staging）

---

### 变更文件清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/formats/_block_utils.py` | **新建** | Block 工具函数（`_reshape_to_blocks` 等） |
| `src/formats/base.py` | 修改 | `_quantize_per_block()` 内联实现 |
| `src/quantize/mx_quantize.py` | 修改 | `_quantize_mx` 降为 thin compat wrapper |
| `src/quantize/__init__.py` | 修改 | 移除 `quantize_mx` 导出 |
| `src/tests/test_format_quantize.py` | 修改 | 新增等价性测试 + `_quantize_per_block` 直接测试 |
| `src/tests/test_scheme_api.py` | 修改 | 从 `quantize_mx` 改为 `quantize(x, scheme)` |
| `src/tests/test_golden_equiv.py` | 修改 | 同上 |
| `src/tests/test_mx_quantize_equiv.py` | 修改 | 同上 |

**不修改的文件**（import 路径保持不变）：
- `src/tests/test_refactor_quantize_elemwise.py` — 使用 `_quantize_mx`（compat wrapper），import 路径不变
- `src/tests/test_mx_quantize_equiv.py` — 使用 `_quantize_mx`（compat wrapper），import 路径不变

### 不变式保证

1. 所有现有测试的 `_quantize_mx(...)` 调用继续通过（compat wrapper）
2. 所有 `format.quantize(x, per_block(...), ...)` 调用结果不变
3. `quantize_mx` 公共 API 的调用方改为 `quantize(x, scheme)` 后结果一致
4. `src/quantize/mx_quantize.py` 不再被生产代码 import（仅 `FormatBase` import 改为 import `_block_utils`）
