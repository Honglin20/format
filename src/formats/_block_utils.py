"""Internal helpers for per-block quantization (MX-style shared exponents).

These are private implementation details of FormatBase._quantize_per_block().
"""
import torch

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


# ---------------------------------------------------------------------------
# Shared exponents
# ---------------------------------------------------------------------------

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

def _reshape_to_blocks(A, axes, block_size):
    if axes is None:
        raise ValueError("axes required in order to determine which "
                         "dimension to apply block size to")
    if block_size == 0:
        raise ValueError("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    if not all(x >= 0 for x in axes):
        raise ValueError("All axes must be non-negative after normalization")
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
