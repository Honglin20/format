"""
MX block quantization: shared_exponents, reshape_to_blocks,
quantize_mx, quantize_mx_op.

Rewritten from mx/mx_ops.py. Key changes:
- Uses FormatBase.from_str() instead of _get_format_params() / ElemFormat
- Uses safe_lshift/safe_rshift from src/quantize/elemwise.py
- Parameter 'round' renamed to 'round_mode'
- Custom CUDA path omitted (Python/CPU path only for now)
- FP32 constants defined locally instead of imported from mx/formats.py
"""
import torch
from src.formats.base import FormatBase
from src.quantize.elemwise import safe_lshift, safe_rshift, quantize_elemwise_core
from src.specs.specs import mx_assert_test

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


# ---------------------------------------------------------------------------
# Shared exponents
# ---------------------------------------------------------------------------

def shared_exponents(A, method="max", axes=None, ebits=0):
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

    # log2(shared_exp) and truncate to integer
    shared_exp = torch.floor(
        torch.log2(
            shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
        )
    )

    # Restrict to [-emax, emax] range
    if ebits > 0:
        emax = 2**(ebits-1) - 1
        shared_exp[shared_exp > emax] = float("NaN")
        shared_exp[shared_exp < -emax] = -emax

    return shared_exp


# ---------------------------------------------------------------------------
# Block reshaping
# ---------------------------------------------------------------------------

def reshape_to_blocks(A, axes, block_size):
    if axes is None:
        raise ValueError("axes required in order to determine which "
                         "dimension to apply block size to")
    if block_size == 0:
        raise ValueError("block_size == 0 in reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
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
                assert shape[axis] % reshape_block_size == 0
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


def undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    A = A.view(padded_shape)
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        A = torch.squeeze(A, dim=axis + 1)
    return A


# ---------------------------------------------------------------------------
# Core MX quantization
# ---------------------------------------------------------------------------

def quantize_mx(
    A,
    scale_bits,
    elem_format,
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round_mode="nearest",
    flush_fp32_subnorms=False,
):
    # Shortcut for no quantization
    if elem_format is None:
        return A

    assert scale_bits > 0

    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + A.ndim if x < 0 else x for x in axes]

    # Get format parameters from FormatBase
    if isinstance(elem_format, str):
        fmt = FormatBase.from_str(elem_format)
    else:
        fmt = elem_format
    ebits = fmt.ebits
    mbits = fmt.mbits
    emax = fmt.emax
    max_norm = fmt.max_norm

    # Perform tiling to the hardware vector size
    if block_size > 0:
        A, axes, orig_shape, padded_shape = reshape_to_blocks(A, axes, block_size)

    # Quantize
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    # Get shared exponents
    shared_exp = shared_exponents(
        A, method=shared_exp_method, axes=shared_exp_axes, ebits=0,
    )

    # Flush subnormal FP32 inputs to zero
    if flush_fp32_subnorms:
        A = A * (shared_exp > -FP32_EXPONENT_BIAS).type(A.dtype)

    # Offset the max exponent by the largest representable exponent
    shared_exp = shared_exp - emax

    scale_emax = 2**(scale_bits-1) - 1
    shared_exp[shared_exp > scale_emax] = float("NaN")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    A = A / (2**shared_exp)

    A = quantize_elemwise_core(
            A, mbits, ebits, max_norm, round_mode=round_mode,
            allow_denorm=True, saturate_normals=True)

    A = A * (2**shared_exp)

    # Undo tile reshaping
    if block_size:
        A = undo_reshape_to_blocks(A, padded_shape, orig_shape, axes)

    return A


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantize_mx_op(
    A,
    mx_specs,
    elem_format=None,
    block_size=None,
    axes=None,
    round_mode="nearest",
):
    mx_assert_test(mx_specs)

    if elem_format is None:
        return A

    if block_size is None:
        block_size = mx_specs["block_size"]

    if mx_specs["scale_bits"] == 0:
        scale_bits = 8
    else:
        scale_bits = mx_specs["scale_bits"]

    return quantize_mx(
            A, scale_bits,
            elem_format, block_size=block_size,
            axes=axes, round_mode=round_mode,
            shared_exp_method=mx_specs["shared_exp_method"],
            flush_fp32_subnorms=mx_specs["mx_flush_fp32_subnorms"])
