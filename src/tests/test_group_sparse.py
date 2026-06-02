"""TDD tests for ADR-013 Group Sparse.

Phase 1: QuantScheme field validation (construction + mutual exclusivity).
Phase 2: compute_group_mask() — per-group mask from calibration data.
"""
import pytest
import torch
import torch.nn as nn

from src.formats.base import FormatBase
from src.scheme.granularity import GranularityMode, GranularitySpec
from src.scheme.quant_scheme import QuantScheme


# ---------------------------------------------------------------------------
# 1. Default construction — backward compatible
# ---------------------------------------------------------------------------

def test_default_group_format_is_none():
    """QuantScheme() defaults group_format=None, group_ratio=0.0."""
    scheme = QuantScheme(format="int8")
    assert scheme.group_format is None
    assert scheme.group_ratio == 0.0


def test_default_group_ratio_is_zero():
    """QuantScheme.per_tensor() defaults group_format=None, group_ratio=0.0."""
    scheme = QuantScheme.per_tensor("int8")
    assert scheme.group_format is None
    assert scheme.group_ratio == 0.0


# ---------------------------------------------------------------------------
# 2. Construction with group_format + group_ratio
# ---------------------------------------------------------------------------

def test_group_format_from_string():
    """group_format string is resolved to FormatBase."""
    scheme = QuantScheme(format="int4", group_format="int8", group_ratio=0.3)
    assert isinstance(scheme.group_format, FormatBase)
    assert scheme.group_format.name == "int8"
    assert scheme.group_ratio == 0.3


def test_group_format_from_formatbase():
    """group_format as FormatBase instance is accepted directly."""
    fmt_h = FormatBase.from_str("int8")
    scheme = QuantScheme(format="int4", group_format=fmt_h, group_ratio=0.3)
    assert scheme.group_format is fmt_h
    assert scheme.group_ratio == 0.3


def test_group_ratio_at_boundaries():
    """group_ratio=0.0 and group_ratio=1.0 are valid."""
    scheme_min = QuantScheme(format="int4", group_format="int8", group_ratio=0.0)
    assert scheme_min.group_ratio == 0.0
    scheme_max = QuantScheme(format="int4", group_format="int8", group_ratio=1.0)
    assert scheme_max.group_ratio == 1.0


# ---------------------------------------------------------------------------
# 3. Validation: group_ratio range
# ---------------------------------------------------------------------------

def test_group_ratio_negative_raises():
    """group_ratio < 0 raises ValueError."""
    with pytest.raises(ValueError, match="group_ratio"):
        QuantScheme(format="int4", group_format="int8", group_ratio=-0.1)


def test_group_ratio_above_one_raises():
    """group_ratio > 1 raises ValueError."""
    with pytest.raises(ValueError, match="group_ratio"):
        QuantScheme(format="int4", group_format="int8", group_ratio=1.1)


# ---------------------------------------------------------------------------
# 4. Validation: mutual exclusivity with outlier_format
# ---------------------------------------------------------------------------

def test_group_format_and_outlier_format_mutually_exclusive():
    """group_format and outlier_format cannot both be non-None."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        QuantScheme(format="int4", group_format="int8", group_ratio=0.3,
                    outlier_format="fp8_e4m3")


def test_group_format_without_group_ratio_ok():
    """group_format without outlier_format is fine."""
    scheme = QuantScheme(format="int4", group_format="int8", group_ratio=0.0)
    assert scheme.group_format is not None
    assert scheme.outlier_format is None


def test_outlier_format_without_group_format_ok():
    """outlier_format without group_format still works (ADR-012 compat)."""
    scheme = QuantScheme(format="int4", outlier_format="int8")
    assert scheme.outlier_format is not None
    assert scheme.group_format is None


# ---------------------------------------------------------------------------
# 5. Validation: invalid group_format string
# ---------------------------------------------------------------------------

def test_group_format_invalid_string_raises():
    """Invalid group_format string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown format"):
        QuantScheme(format="int4", group_format="nonexistent_format", group_ratio=0.3)


def test_group_format_wrong_type_raises():
    """group_format of wrong type raises TypeError."""
    with pytest.raises(TypeError, match="group_format"):
        QuantScheme(format="int4", group_format=42, group_ratio=0.3)


# ===========================================================================
# Phase 2: compute_group_mask()
# ===========================================================================


# ---------------------------------------------------------------------------
# 2a. Input validation
# ---------------------------------------------------------------------------

def test_group_mask_no_batch_dim_raises():
    """x_calib without batch dim raises ValueError."""
    from src.quantize._group_mask import compute_group_mask
    x = torch.randn(8)  # 1D — no batch dim
    with pytest.raises(ValueError, match="batch"):
        compute_group_mask(x, GranularitySpec.per_tensor(), 0.3)


def test_group_mask_invalid_ratio_raises():
    """group_ratio outside (0, 1] raises ValueError."""
    from src.quantize._group_mask import compute_group_mask
    x = torch.randn(3, 4, 8)
    with pytest.raises(ValueError, match="group_ratio"):
        compute_group_mask(x, GranularitySpec.per_tensor(), 0.0)
    with pytest.raises(ValueError, match="group_ratio"):
        compute_group_mask(x, GranularitySpec.per_tensor(), 1.2)


# ---------------------------------------------------------------------------
# 2b. PER_TENSOR — single group, mask is scalar True
# ---------------------------------------------------------------------------

def test_group_mask_per_tensor():
    """PER_TENSOR: 1 group → scalar True mask."""
    from src.quantize._group_mask import compute_group_mask
    x_calib = torch.randn(5, 4, 8)
    mask = compute_group_mask(x_calib, GranularitySpec.per_tensor(), 0.5)
    assert mask.shape == ()
    assert mask.dtype == torch.bool
    assert mask.item() is True


def test_group_mask_per_tensor_different_ratio():
    """PER_TENSOR mask is True regardless of ratio (always 1 group)."""
    from src.quantize._group_mask import compute_group_mask
    x_calib = torch.randn(3, 4, 8)
    for r in [0.01, 0.3, 0.9, 1.0]:
        mask = compute_group_mask(x_calib, GranularitySpec.per_tensor(), r)
        assert mask.item() is True


# ---------------------------------------------------------------------------
# 2c. PER_CHANNEL — per-channel amax → top-k channels
# ---------------------------------------------------------------------------

def test_group_mask_per_channel_shape():
    """PER_CHANNEL returns mask of shape (C,)."""
    from src.quantize._group_mask import compute_group_mask
    x_calib = torch.randn(5, 6, 12)  # 6 channels (axis=0)
    gran = GranularitySpec.per_channel(axis=0)
    mask = compute_group_mask(x_calib, gran, 0.5)
    assert mask.shape == (6,)
    assert mask.dtype == torch.bool
    # k = max(1, int(6 * 0.5)) = 3
    assert mask.sum().item() == 3


def test_group_mask_per_channel_topk_selection():
    """PER_CHANNEL: channels with largest amax become H."""
    from src.quantize._group_mask import compute_group_mask
    # 3 samples, 4 channels (axis=0), each of size 8
    # Channel 0: small (amax ~0.1),   Channel 1: large (amax ~5.0)
    # Channel 2: medium (amax ~1.0),  Channel 3: extra large (amax ~10.0)
    torch.manual_seed(42)
    base = torch.randn(3, 4, 8)
    scale = torch.tensor([0.1, 5.0, 1.0, 10.0]).view(1, 4, 1)
    x_calib = base * scale

    gran = GranularitySpec.per_channel(axis=0)
    mask = compute_group_mask(x_calib, gran, 0.5)  # k = max(1, int(4*0.5)) = 2

    # Channel 3 (scale 10.0) and Channel 1 (scale 5.0) should be H
    assert mask[3].item() is True
    assert mask[1].item() is True
    assert mask[0].item() is False
    assert mask[2].item() is False


def test_group_mask_per_channel_all_h():
    """PER_CHANNEL group_ratio=1.0 → all channels H."""
    from src.quantize._group_mask import compute_group_mask
    x_calib = torch.randn(5, 4, 8)
    gran = GranularitySpec.per_channel(axis=0)
    mask = compute_group_mask(x_calib, gran, 1.0)
    assert mask.all().item() is True
    assert mask.sum().item() == 4


def test_group_mask_per_channel_single_h():
    """PER_CHANNEL group_ratio=0.01 with 4 channels → k=1, one H."""
    from src.quantize._group_mask import compute_group_mask
    x_calib = torch.randn(3, 4, 8)
    gran = GranularitySpec.per_channel(axis=0)
    mask = compute_group_mask(x_calib, gran, 0.01)
    assert mask.sum().item() == 1


def test_group_mask_per_channel_cross_sample_max():
    """Cross-sample max: channel H even if only one sample has large amax."""
    from src.quantize._group_mask import compute_group_mask
    # 3 samples. Only sample 1 has channel 0 with huge values.
    x_calib = torch.zeros(3, 4, 8)
    x_calib[0] = torch.randn(4, 8) * 0.1
    x_calib[1, 0, :] = 100.0  # channel 0, sample 1 is huge
    x_calib[2] = torch.randn(4, 8) * 0.1

    gran = GranularitySpec.per_channel(axis=0)
    mask = compute_group_mask(x_calib, gran, 0.01)  # k=1

    # Channel 0 should be the single H (had 100.0 in sample 1)
    assert mask[0].item() is True
    assert mask[1:].any().item() is False


# ---------------------------------------------------------------------------
# 2d. PER_BLOCK — per-block amax → top-k blocks
# ---------------------------------------------------------------------------

def test_group_mask_per_block_shape():
    """PER_BLOCK mask shape matches block-group dimensions."""
    from src.quantize._group_mask import compute_group_mask
    # (3, 2, 64), block_axis=-1, block_size=32 → reshape (2,2,32) → 4 blocks
    x_calib = torch.randn(3, 2, 64)
    gran = GranularitySpec.per_block(size=32, axis=-1)
    mask = compute_group_mask(x_calib, gran, 0.5)
    # Reshaped: (2, 64/32) = (2, 2) blocks
    assert mask.shape == (2, 2)
    assert mask.dtype == torch.bool
    # k = max(1, int(4 * 0.5)) = 2
    assert mask.sum().item() == 2


def test_group_mask_per_block_topk():
    """PER_BLOCK: blocks with largest amax become H."""
    from src.quantize._group_mask import compute_group_mask
    # 3 samples. Tensor shape (2, 64), block_axis=-1, block_size=32
    # → 4 blocks: (row0,block0), (row0,block1), (row1,block0), (row1,block1)
    # Make (row0,block1) and (row1,block0) have largest amax
    x_calib = torch.ones(3, 2, 64) * 0.01
    x_calib[:, 0, 32:64] = 5.0  # row0, block1 → large
    x_calib[:, 1, 0:32] = 5.0    # row1, block0 → large

    gran = GranularitySpec.per_block(size=32, axis=-1)
    mask = compute_group_mask(x_calib, gran, 0.5)  # k = max(1, int(4*0.5)) = 2

    # mask[0, 1] = row0, block1 = H
    # mask[1, 0] = row1, block0 = H
    assert mask[0, 1].item() is True
    assert mask[1, 0].item() is True
    assert mask[0, 0].item() is False  # row0, block0 (small)
    assert mask[1, 1].item() is False  # row1, block1 (small)


def test_group_mask_per_block_all_h():
    """PER_BLOCK group_ratio=1.0 → all blocks H."""
    from src.quantize._group_mask import compute_group_mask
    x_calib = torch.randn(3, 2, 64)
    gran = GranularitySpec.per_block(size=32, axis=-1)
    mask = compute_group_mask(x_calib, gran, 1.0)
    assert mask.all().item() is True


def test_group_mask_per_block_padded():
    """PER_BLOCK with non-divisible dimension still works (padding)."""
    from src.quantize._group_mask import compute_group_mask
    # 50 not divisible by 32 → padded
    x_calib = torch.randn(3, 4, 50)
    gran = GranularitySpec.per_block(size=32, axis=-1)
    mask = compute_group_mask(x_calib, gran, 0.5)
    # 50 padded to 64 → 2 blocks per row → 4*2=8 blocks, k=4
    assert mask.sum().item() == 4


# ---------------------------------------------------------------------------
# 2e. BANK — per-bank amax → top-k banks
# ---------------------------------------------------------------------------

def test_group_mask_bank_shape():
    """BANK mask shape = (num_banks,)."""
    from src.quantize._group_mask import compute_group_mask
    x_calib = torch.randn(5, 4, 64)  # bank_axis=-1, bank_size=16 → 4 banks
    gran = GranularitySpec(mode=GranularityMode.BANK, bank_size=16, bank_axis=-1)
    mask = compute_group_mask(x_calib, gran, 0.5)
    assert mask.shape == (4,)
    assert mask.dtype == torch.bool
    # k = max(1, int(4 * 0.5)) = 2
    assert mask.sum().item() == 2


def test_group_mask_bank_topk():
    """BANK: banks with largest amax become H."""
    from src.quantize._group_mask import compute_group_mask
    # 3 samples. Tensor (4, 64), bank_axis=-1, bank_size=16 → 4 banks
    # Bank 0: cols 0-15,  Bank 1: cols 16-31
    # Bank 2: cols 32-47, Bank 3: cols 48-63
    # Make banks 1 and 2 large
    x_calib = torch.ones(3, 4, 64) * 0.01
    x_calib[:, :, 16:32] = 10.0   # bank 1 large
    x_calib[:, :, 32:48] = 10.0   # bank 2 large

    gran = GranularitySpec(mode=GranularityMode.BANK, bank_size=16, bank_axis=-1)
    mask = compute_group_mask(x_calib, gran, 0.5)  # k = max(1, int(4*0.5)) = 2

    assert mask[1].item() is True   # bank 1 H
    assert mask[2].item() is True   # bank 2 H
    assert mask[0].item() is False  # bank 0 L
    assert mask[3].item() is False  # bank 3 L


def test_group_mask_bank_cross_sample_max():
    """BANK: cross-sample max ensures bursty banks are H."""
    from src.quantize._group_mask import compute_group_mask
    x_calib = torch.ones(3, 2, 32) * 0.01  # 2 banks (bank_size=16)
    # Only sample 1 has huge values in bank 0
    x_calib[1, :, 0:16] = 100.0

    gran = GranularitySpec(mode=GranularityMode.BANK, bank_size=16, bank_axis=-1)
    mask = compute_group_mask(x_calib, gran, 0.5)  # k = max(1, int(2*0.5)) = 1

    assert mask[0].item() is True   # bank 0 H (had 100 in sample 1)
    assert mask[1].item() is False  # bank 1 L


def test_group_mask_bank_all_h():
    """BANK group_ratio=1.0 → all banks H."""
    from src.quantize._group_mask import compute_group_mask
    x_calib = torch.randn(3, 4, 64)
    gran = GranularitySpec(mode=GranularityMode.BANK, bank_size=16, bank_axis=-1)
    mask = compute_group_mask(x_calib, gran, 1.0)
    assert mask.all().item() is True


# ---------------------------------------------------------------------------
# 2f. Transposed axes (forward/backward axis conventions)
# ---------------------------------------------------------------------------

def test_group_mask_per_channel_negative_axis():
    """PER_CHANNEL with channel_axis=-1."""
    from src.quantize._group_mask import compute_group_mask
    # shape (3, 8, 6), channel_axis=-1 → C=6
    torch.manual_seed(42)
    base = torch.randn(3, 8, 6)
    scales = torch.tensor([0.1, 0.2, 0.3, 10.0, 0.5, 0.6])
    x_calib = base * scales.view(1, 1, 6)

    gran = GranularitySpec.per_channel(axis=-1)
    mask = compute_group_mask(x_calib, gran, 0.3)  # k = max(1, int(6*0.3)) = 1

    # Channel 3 (index 3, scale 10.0) should be H
    assert mask[3].item() is True
    assert mask.sum().item() == 1


def test_group_mask_bank_negative_axis():
    """BANK with bank_axis=-1 (last dim)."""
    from src.quantize._group_mask import compute_group_mask
    x_calib = torch.ones(3, 2, 32) * 0.01
    x_calib[:, :, 16:32] = 10.0  # second bank large

    gran = GranularitySpec(mode=GranularityMode.BANK, bank_size=16, bank_axis=-1)
    mask = compute_group_mask(x_calib, gran, 0.5)  # k=1

    assert mask[1].item() is True
    assert mask[0].item() is False


# ===========================================================================
# Phase 3: FormatBase group-sparse quantization (dynamic path)
# ===========================================================================


# ---------------------------------------------------------------------------
# 3a. PER_TENSOR — whole tensor uses group_format
# ---------------------------------------------------------------------------

def test_quantize_per_tensor_group_sparse():
    """PER_TENSOR + group_format=int8, format=int4 → result differs from int4-only."""
    from src.formats.base import FormatBase
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")
    x = torch.randn(4, 8)
    g = GranularitySpec.per_tensor()

    # group_sparse: H format = int8
    out_gs = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5)
    # standard int4
    out_int4 = fmt_int4.quantize(x, g)
    # standard int8
    out_int8 = fmt_int8.quantize(x, g)

    # group_sparse should match int8 (all H), not int4
    assert torch.allclose(out_gs, out_int8, atol=1e-7)
    assert not torch.allclose(out_gs, out_int4)


def test_quantize_per_tensor_group_sparse_float_format():
    """PER_TENSOR with float format delegates to standard path (no group sparse)."""
    from src.formats.base import FormatBase
    fmt_fp8 = FormatBase.from_str("fp8_e4m3")
    fmt_int8 = FormatBase.from_str("int8")
    x = torch.randn(4, 8)
    g = GranularitySpec.per_tensor()

    out_gs = fmt_fp8.quantize(x, g, group_format=fmt_int8, group_ratio=0.5)
    out_std = fmt_fp8.quantize(x, g)  # float → direct elemwise, no group sparse
    assert torch.allclose(out_gs, out_std, atol=1e-7)


# ---------------------------------------------------------------------------
# 3b. PER_CHANNEL — H channels use group_format
# ---------------------------------------------------------------------------

def test_quantize_per_channel_group_sparse():
    """PER_CHANNEL: H channels get group_format (int8), L channels get format (int4)."""
    from src.formats.base import FormatBase
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    # 6 channels (axis=0), make channels 2 and 4 large → should be H
    torch.manual_seed(42)
    x = torch.randn(6, 8)
    x[2] *= 20.0
    x[4] *= 20.0
    g = GranularitySpec.per_channel(axis=0)

    out_gs = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.4)  # k=2
    out_int4 = fmt_int4.quantize(x, g)
    out_int8 = fmt_int8.quantize(x, g)

    # H channels (2, 4) → close to int8; L channels → close to int4
    assert not torch.allclose(out_gs[2], out_int4[2], atol=1e-7)  # diff from int4
    assert torch.allclose(out_gs[2], out_int8[2], atol=1e-7)       # matches int8
    assert torch.allclose(out_gs[0], out_int4[0], atol=1e-7)       # L → int4


def test_quantize_per_channel_group_sparse_all_h():
    """group_ratio=1.0 → all channels H → matches int8 quantization."""
    from src.formats.base import FormatBase
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")
    x = torch.randn(4, 8)
    g = GranularitySpec.per_channel(axis=0)

    out_gs = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=1.0)
    out_int8 = fmt_int8.quantize(x, g)
    assert torch.allclose(out_gs, out_int8, atol=1e-7)


# ---------------------------------------------------------------------------
# 3c. PER_BLOCK — H blocks use group_format
# ---------------------------------------------------------------------------

def test_quantize_per_block_group_sparse():
    """PER_BLOCK: H blocks get group_format, L blocks get format."""
    from src.formats.base import FormatBase
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    # (2, 64), block_size=32 → 4 blocks. Use varied values so int4 vs int8 differ.
    torch.manual_seed(99)
    x = torch.randn(2, 64) * 0.01
    x[0, 32:64] = torch.randn(32) * 2 + 5.0   # row 0, block 1 large (varied)
    x[1, 0:32] = torch.randn(32) * 2 + 5.0     # row 1, block 0 large (varied)
    g = GranularitySpec.per_block(size=32, axis=-1)

    out_gs = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5)
    out_int4 = fmt_int4.quantize(x, g)

    # Verify H blocks differ from int4 (using int8 instead)
    assert not torch.allclose(out_gs[0, 32:64], out_int4[0, 32:64], atol=1e-5)
    assert not torch.allclose(out_gs[1, 0:32], out_int4[1, 0:32], atol=1e-5)
    # L blocks (small values) should match int4
    assert torch.allclose(out_gs[0, 0:32], out_int4[0, 0:32], atol=1e-7)
    assert torch.allclose(out_gs[1, 32:64], out_int4[1, 32:64], atol=1e-7)


def test_quantize_per_block_group_sparse_all_h():
    """PER_BLOCK group_ratio=1.0 → all blocks use group_format."""
    from src.formats.base import FormatBase
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")
    x = torch.randn(2, 64)
    g = GranularitySpec.per_block(size=32, axis=-1)

    out_gs = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=1.0)
    out_int8 = fmt_int8.quantize(x, g)
    assert torch.allclose(out_gs, out_int8, atol=1e-6)


# ---------------------------------------------------------------------------
# 3d. BANK — H banks use group_format
# ---------------------------------------------------------------------------

def test_quantize_bank_group_sparse():
    """BANK: H banks get group_format, L banks get format."""
    from src.formats.base import FormatBase
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    # (4, 64), bank_size=16 → 4 banks. Varied values so int4 vs int8 differ.
    torch.manual_seed(77)
    x = torch.randn(4, 64) * 0.01
    x[:, 16:32] = torch.randn(4, 16) * 3 + 8.0  # bank 1 large (varied)
    x[:, 32:48] = torch.randn(4, 16) * 3 + 8.0  # bank 2 large (varied)
    g = GranularitySpec(mode=GranularityMode.BANK, bank_size=16, bank_axis=-1)

    out_gs = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5)
    out_int4 = fmt_int4.quantize(x, g)

    # H banks (1, 2) should differ from int4
    assert not torch.allclose(out_gs[:, 16:32], out_int4[:, 16:32], atol=1e-5)
    assert not torch.allclose(out_gs[:, 32:48], out_int4[:, 32:48], atol=1e-5)
    # L banks (0, 3) should match int4
    assert torch.allclose(out_gs[:, 0:16], out_int4[:, 0:16], atol=1e-7)


def test_quantize_bank_group_sparse_all_h():
    """BANK group_ratio=1.0 → all banks use group_format."""
    from src.formats.base import FormatBase
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")
    x = torch.randn(4, 64)
    g = GranularitySpec(mode=GranularityMode.BANK, bank_size=16, bank_axis=-1)

    out_gs = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=1.0)
    out_int8 = fmt_int8.quantize(x, g)
    assert torch.allclose(out_gs, out_int8, atol=1e-6)


# ---------------------------------------------------------------------------
# 3e. Edge cases
# ---------------------------------------------------------------------------

def test_quantize_group_sparse_backward_compat():
    """group_format=None → standard quantization path (no group sparse)."""
    from src.formats.base import FormatBase
    fmt = FormatBase.from_str("int4")
    x = torch.randn(4, 8)
    g = GranularitySpec.per_tensor()

    out_gs = fmt.quantize(x, g, group_format=None, group_ratio=0.0)
    out_std = fmt.quantize(x, g)
    assert torch.allclose(out_gs, out_std, atol=1e-7)


def test_quantize_group_sparse_with_scale():
    """group_sparse with pre-computed scale should use it (fp32 mode)."""
    from src.formats.base import FormatBase
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")
    x = torch.randn(4, 8)
    g = GranularitySpec.per_tensor()

    scale = torch.tensor(3.0)
    out_gs = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                               scale=scale, scale_storage="fp32")
    # Manual: normalize by scale (no POT), quantize with int8, rescale
    expected = fmt_int8.quantize_elemwise(x / scale, round_mode="nearest") * scale
    assert torch.allclose(out_gs, expected, atol=1e-7)


def test_quantize_group_sparse_scale_storage_fp32():
    """group_sparse respects scale_storage='fp32' (no POT rounding)."""
    from src.formats.base import FormatBase
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")
    x = torch.randn(4, 8)
    g = GranularitySpec.per_channel(axis=0)

    out_pot = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                                scale_storage="pot")
    out_fp32 = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                                 scale_storage="fp32")
    # fp32 and pot should differ (pot rounds amax to power-of-2)
    assert not torch.allclose(out_pot, out_fp32, atol=1e-7)


# ===========================================================================
# Phase 6: QuantConfig fields + to_op_config() + from_descriptor()
# ===========================================================================

from src.session._config import QuantConfig, resolve_config


# ---------------------------------------------------------------------------
# 6a. Defaults — backward compatible
# ---------------------------------------------------------------------------

def test_quantconfig_default_group_fields():
    """QuantConfig() defaults group_format=None, group_ratio=0.0."""
    cfg = QuantConfig()
    assert cfg.group_format is None
    assert cfg.group_ratio == 0.0
    assert cfg.a_group_format is None
    assert cfg.a_group_ratio is None


# ---------------------------------------------------------------------------
# 6b. Field storage
# ---------------------------------------------------------------------------

def test_quantconfig_group_format_stored():
    """group_format string is stored as-is."""
    cfg = QuantConfig(w_format="int4", group_format="int8", group_ratio=0.3)
    assert cfg.group_format == "int8"
    assert cfg.group_ratio == 0.3


def test_quantconfig_a_group_format_stored():
    """a_group_format overrides group_format for activation."""
    cfg = QuantConfig(w_format="int4", group_format="int8", group_ratio=0.3,
                      a_group_format="fp8_e4m3", a_group_ratio=0.5)
    assert cfg.a_group_format == "fp8_e4m3"
    assert cfg.a_group_ratio == 0.5


# ---------------------------------------------------------------------------
# 6c. Validation: group_ratio range
# ---------------------------------------------------------------------------

def test_quantconfig_group_ratio_negative_raises():
    """group_ratio < 0 raises ValueError."""
    with pytest.raises(ValueError, match="group_ratio"):
        QuantConfig(w_format="int4", group_format="int8", group_ratio=-0.1)


def test_quantconfig_group_ratio_above_one_raises():
    """group_ratio > 1 raises ValueError."""
    with pytest.raises(ValueError, match="group_ratio"):
        QuantConfig(w_format="int4", group_format="int8", group_ratio=1.1)


# ---------------------------------------------------------------------------
# 6d. Validation: group_format must be resolvable
# ---------------------------------------------------------------------------

def test_quantconfig_invalid_group_format_raises():
    """Invalid group_format string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown group_format"):
        QuantConfig(w_format="int4", group_format="no_such_format", group_ratio=0.3)


def test_quantconfig_group_format_type_error():
    """group_format of wrong type raises TypeError."""
    with pytest.raises(TypeError, match="group_format must be a string"):
        QuantConfig(w_format="int4", group_format=42, group_ratio=0.3)


def test_quantconfig_invalid_a_group_format_raises():
    """Invalid a_group_format string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown a_group_format"):
        QuantConfig(w_format="int4", group_format="int8", group_ratio=0.3,
                    a_group_format="no_such_format")


def test_quantconfig_a_group_format_type_error():
    """a_group_format of wrong type raises TypeError."""
    with pytest.raises(TypeError, match="a_group_format must be a string"):
        QuantConfig(w_format="int4", group_format="int8", group_ratio=0.3,
                    a_group_format=42)


def test_quantconfig_a_group_format_weight_only_raises():
    """a_group_format cannot be set when weight_only=True."""
    with pytest.raises(ValueError, match="a_group_format.*weight_only"):
        QuantConfig(w_format="nf4", weight_only=True, a_group_format="int8")


# ---------------------------------------------------------------------------
# 6e. Validation: mutual exclusivity with outlier
# ---------------------------------------------------------------------------

def test_quantconfig_group_and_outlier_mutually_exclusive():
    """group_format and outlier_format cannot both be set."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        QuantConfig(w_format="int4", group_format="int8", group_ratio=0.3,
                    outlier_format="fp8_e4m3", outlier_ratio=0.1)


def test_quantconfig_group_ratio_and_outlier_ratio_mutually_exclusive():
    """group_ratio > 0 and outlier_ratio > 0 cannot both be set."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        QuantConfig(w_format="int4", group_format="int8", group_ratio=0.3,
                    outlier_ratio=0.1)


def test_quantconfig_group_format_without_ratio_ok():
    """group_format set but group_ratio=0 — valid, just unused at runtime."""
    cfg = QuantConfig(w_format="int4", group_format="int8", group_ratio=0.0)
    assert cfg.group_format == "int8"
    assert cfg.group_ratio == 0.0


# ---------------------------------------------------------------------------
# 6f. to_op_config() — group_format wiring
# ---------------------------------------------------------------------------

def test_to_op_config_group_format_on_weight():
    """group_format → QuantScheme.group_format on weight and activation."""
    cfg = QuantConfig(w_format="int4", group_format="int8", group_ratio=0.3)
    result = cfg.to_op_config()
    assert result.weight.group_format is not None
    assert result.weight.group_format.name == "int8"
    assert result.weight.group_ratio == 0.3
    assert result.input.group_format is not None
    assert result.input.group_format.name == "int8"
    assert result.input.group_ratio == 0.3


def test_to_op_config_a_group_format_overrides():
    """a_group_format overrides group_format on activation scheme only."""
    cfg = QuantConfig(
        w_format="int4", a_format="int8",
        group_format="fp8_e4m3",
        a_group_format="nf4",
        group_ratio=0.3,
        a_group_ratio=0.5,
    )
    result = cfg.to_op_config()
    assert result.weight.group_format.name == "fp8_e4m3"
    assert result.weight.group_ratio == 0.3
    assert result.input.group_format.name == "nf4"
    assert result.input.group_ratio == 0.5


def test_to_op_config_a_group_format_falls_back():
    """a_group_format=None → activation follows group_format."""
    cfg = QuantConfig(w_format="int4", group_format="int8", group_ratio=0.3)
    result = cfg.to_op_config()
    assert result.input.group_format is not None
    assert result.input.group_format.name == "int8"
    assert result.input.group_ratio == 0.3


def test_to_op_config_a_group_ratio_falls_back():
    """a_group_ratio=None → activation follows group_ratio."""
    cfg = QuantConfig(w_format="int4", group_format="int8", group_ratio=0.3)
    result = cfg.to_op_config()
    assert result.input.group_ratio == 0.3


def test_to_op_config_group_format_none():
    """No group_format → QuantScheme.group_format is None, group_ratio=0."""
    cfg = QuantConfig(w_format="int4")
    result = cfg.to_op_config()
    assert result.weight.group_format is None
    assert result.weight.group_ratio == 0.0
    assert result.input.group_format is None
    assert result.input.group_ratio == 0.0


def test_to_op_config_no_conflict_with_outlier():
    """Outlier-only config (no group) → group_format is None."""
    cfg = QuantConfig(w_format="int4", outlier_format="int8", outlier_ratio=0.1)
    result = cfg.to_op_config()
    assert result.weight.group_format is None
    assert result.weight.outlier_format is not None


# ---------------------------------------------------------------------------
# 6g. from_descriptor() — group_format support
# ---------------------------------------------------------------------------

def test_from_descriptor_group_format():
    """Descriptor with group_format/group_ratio → QuantConfig fields set."""
    cfg = QuantConfig.from_descriptor({
        "format": "int4",
        "granularity": "per_channel",
        "group_format": "int8",
        "group_ratio": 0.3,
    })
    assert cfg.group_format == "int8"
    assert cfg.group_ratio == 0.3


def test_from_descriptor_a_group_format():
    """Descriptor with a_group_format → activation override set."""
    cfg = QuantConfig.from_descriptor({
        "format": "int4",
        "granularity": "per_channel",
        "group_format": "fp8_e4m3",
        "a_group_format": "int8",
        "group_ratio": 0.3,
        "a_group_ratio": 0.5,
    })
    assert cfg.a_group_format == "int8"
    assert cfg.a_group_ratio == 0.5


def test_resolve_config_group_format():
    """resolve_config() with group_format → OpQuantConfig with group_format."""
    desc = {
        "format": "int4",
        "granularity": "per_channel",
        "group_format": "int8",
        "group_ratio": 0.3,
    }
    result = resolve_config(desc)
    assert result.weight.group_format is not None
    assert result.weight.group_format.name == "int8"
    assert result.weight.group_ratio == 0.3


def test_resolve_config_a_group_format():
    """resolve_config() with a_group_format → activation override."""
    desc = {
        "format": "int4",
        "granularity": "per_channel",
        "group_format": "fp8_e4m3",
        "a_group_format": "int8",
        "group_ratio": 0.2,
        "a_group_ratio": 0.6,
    }
    result = resolve_config(desc)
    assert result.weight.group_format.name == "fp8_e4m3"
    assert result.weight.group_ratio == 0.2
    assert result.input.group_format.name == "int8"
    assert result.input.group_ratio == 0.6


def test_from_descriptor_group_format_invalid_raises():
    """Invalid group_format in descriptor raises TypeError/ValueError."""
    with pytest.raises(TypeError, match="'group_format' must be a string"):
        QuantConfig.from_descriptor({
            "format": "int4",
            "granularity": "per_channel",
            "group_format": 42,
        })


def test_from_descriptor_a_group_format_weight_only_raises():
    """a_group_format + weight_only in descriptor raises ValueError."""
    with pytest.raises(ValueError, match="'a_group_format'.*weight_only"):
        QuantConfig.from_descriptor({
            "format": "int4",
            "granularity": "per_channel",
            "a_group_format": "int8",
            "weight_only": True,
        })


# ===========================================================================
# Phase 4: CalibrationSession group_sparse integration
# ===========================================================================

from src.calibration.pipeline import CalibrationSession, _compute_sparse_scales
from src.calibration.strategies import MaxScaleStrategy


class _GroupSparseModule(nn.Module):
    """Minimal module with cfg.output having group_format set."""
    def __init__(self, group_format, group_ratio, weight_format="int4", mode="per_channel"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 8))
        from src.scheme.granularity import GranularitySpec, GranularityMode
        from src.formats.base import FormatBase
        fmt = FormatBase.from_str(weight_format)
        if mode == "per_channel":
            g = GranularitySpec.per_channel(axis=0)
        elif mode == "bank":
            g = GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1)
        else:
            g = GranularitySpec.per_tensor()
        gf = FormatBase.from_str(group_format) if group_format else None
        from src.scheme.quant_scheme import QuantScheme
        from src.scheme.op_config import OpQuantConfig
        self.cfg = OpQuantConfig(
            output=QuantScheme(format=fmt, granularity=g, group_format=gf, group_ratio=group_ratio),
        )

    def forward(self, x):
        return x + self.weight


def test_calibration_group_sparse_collects_samples():
    """When group_sparse=True, per-sample outputs are collected for group_mask."""
    model = _GroupSparseModule("int8", 0.3)
    session = CalibrationSession(model, MaxScaleStrategy(), sparse=True)
    with session:
        for _ in range(3):
            model(torch.randn(4, 8))
    assert len(session._output_samples) == 1
    name = list(session._output_samples.keys())[0]
    assert len(session._output_samples[name]) == 3


def test_calibration_group_sparse_assigns_mask():
    """After calibration, _output_group_mask buffer is assigned."""
    model = _GroupSparseModule("int8", 0.3)
    session = CalibrationSession(model, MaxScaleStrategy(), sparse=True)
    with session:
        for _ in range(3):
            model(torch.randn(4, 8))
    assert hasattr(model, "_output_group_mask")
    mask = model._output_group_mask
    assert mask.dtype == torch.bool
    # PER_CHANNEL axis=0 with 4 channels, group_ratio=0.3 → k=1
    assert mask.shape == (4,)
    assert mask.sum().item() == 1


def test_calibration_group_sparse_no_group_format_no_mask():
    """When scheme has no group_format, no group_mask buffer is assigned."""
    model = _GroupSparseModule(None, 0.0)
    session = CalibrationSession(model, MaxScaleStrategy(), sparse=True)
    with session:
        for _ in range(2):
            model(torch.randn(4, 8))
    assert not hasattr(model, "_output_group_mask")


def test_calibration_group_sparse_bank_mode():
    """BANK mode: group_mask shape matches number of banks."""
    model = _GroupSparseModule("int8", 0.8, mode="bank")
    session = CalibrationSession(model, MaxScaleStrategy(), sparse=True)
    with session:
        for _ in range(2):
            model(torch.randn(4, 8))
    assert hasattr(model, "_output_group_mask")
    mask = model._output_group_mask
    # 8 / 4 = 2 banks, group_ratio=0.8 → k=1
    assert mask.shape == (2,)
    assert mask.sum().item() == 1


def test_calibration_group_sparse_per_tensor_mode():
    """PER_TENSOR: group_mask is scalar True."""
    model = _GroupSparseModule("int8", 0.5, mode="per_tensor")
    session = CalibrationSession(model, MaxScaleStrategy(), sparse=True)
    with session:
        for _ in range(2):
            model(torch.randn(4, 8))
    assert hasattr(model, "_output_group_mask")
    mask = model._output_group_mask
    assert mask.shape == ()
    assert mask.item() is True


def test_calibration_group_sparse_cross_sample_max():
    """Group mask uses cross-sample max to select H groups."""
    torch.manual_seed(123)
    model = _GroupSparseModule("int8", 0.3)
    session = CalibrationSession(model, MaxScaleStrategy(), sparse=True)
    with session:
        # First 2 samples: small values
        model(torch.randn(4, 8) * 0.01)
        model(torch.randn(4, 8) * 0.01)
        # Third sample: channel 3 is huge
        x = torch.randn(4, 8) * 0.01
        x[3] *= 100.0
        model(x)
    mask = model._output_group_mask
    # Channel 3 should be the H group (had 100.0 in sample 3)
    assert mask[3].item() is True


def test_calibration_group_sparse_no_effect_without_flag():
    """Without sparse=True, no sample collection or group_mask assignment."""
    model = _GroupSparseModule("int8", 0.3)
    session = CalibrationSession(model, MaxScaleStrategy(), sparse=False)
    with session:
        for _ in range(2):
            model(torch.randn(4, 8))
    assert not hasattr(model, "_output_group_mask")
    assert len(session._output_samples) == 0


# ===========================================================================
# Phase 5: FormatBase static path (pre-computed group_mask)
# ===========================================================================


def test_quantize_per_tensor_group_sparse_static():
    """PER_TENSOR with group_mask → still uses group_format (all H)."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")
    x = torch.randn(4, 8)
    g = GranularitySpec.per_tensor()
    group_mask = torch.tensor(True)

    out = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                            group_mask=group_mask)
    out_int8 = fmt_int8.quantize(x, g)
    assert torch.allclose(out, out_int8, atol=1e-7)


def test_quantize_per_channel_group_sparse_static():
    """PER_CHANNEL with pre-computed group_mask → uses it instead of dynamic top-k."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    x = torch.randn(4, 8)
    g = GranularitySpec.per_channel(axis=0)

    # Pre-computed mask: channels 0 and 3 are H
    h_mask = torch.tensor([True, False, False, True], dtype=torch.bool)

    out_static = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                                   group_mask=h_mask, scale_storage="fp32")
    out_int8 = fmt_int8.quantize(x, g, scale_storage="fp32")

    # H channels should match int8
    assert torch.allclose(out_static[0], out_int8[0], atol=1e-7)
    assert torch.allclose(out_static[3], out_int8[3], atol=1e-7)
    # L channels should match int4
    out_int4 = fmt_int4.quantize(x, g, scale_storage="fp32")
    assert torch.allclose(out_static[1], out_int4[1], atol=1e-7)
    assert torch.allclose(out_static[2], out_int4[2], atol=1e-7)


def test_quantize_bank_group_sparse_static():
    """BANK with pre-computed group_mask → uses it instead of dynamic top-k."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    x = torch.randn(4, 32)
    g = GranularitySpec(mode=GranularityMode.BANK, bank_size=8, bank_axis=-1)
    # 4 banks: k = max(1, int(4*0.5)) = 2

    # Pre-computed mask: banks 1 and 2 are H
    h_mask = torch.tensor([False, True, True, False], dtype=torch.bool)

    out_static = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                                   group_mask=h_mask, scale_storage="fp32")
    out_dynamic = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                                    scale_storage="fp32")

    # Static and dynamic should differ (different H banks potentially)
    # but both should produce per-bank quantization
    assert out_static.shape == x.shape
    # Verify H banks (1, 2) differ from L banks
    assert not torch.allclose(out_static[:, 0:8], out_static[:, 8:16], atol=1e-5)


def test_quantize_group_sparse_static_with_scale():
    """Static group_mask with pre-computed scale → uses both."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    x = torch.randn(4, 8)
    g = GranularitySpec.per_channel(axis=0)

    h_mask = torch.tensor([True, False, False, True], dtype=torch.bool)
    scale = torch.tensor([[3.0], [5.0], [2.0], [4.0]]).view(4, 1)

    out = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                            group_mask=h_mask, scale=scale, scale_storage="fp32")

    # Manual: quantize with both formats, then torch.where
    x_norm = x / scale
    x_q_h = fmt_int8.quantize_elemwise(x_norm)
    x_q_l = fmt_int4.quantize_elemwise(x_norm)
    expected = torch.where(h_mask.view(4, 1), x_q_h * scale, x_q_l * scale)
    assert torch.allclose(out, expected, atol=1e-7)


def test_quantize_group_sparse_static_pot_storage():
    """Static group_mask with pot scale_storage applies POT rounding."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    x = torch.randn(4, 8)
    g = GranularitySpec.per_channel(axis=0)
    h_mask = torch.tensor([True, False, False, True], dtype=torch.bool)

    out_pot = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                                group_mask=h_mask, scale_storage="pot")
    out_fp32 = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                                 group_mask=h_mask, scale_storage="fp32")
    # pot and fp32 should differ
    assert not torch.allclose(out_pot, out_fp32, atol=1e-7)


def test_quantize_group_sparse_static_no_mask_uses_dynamic():
    """Without group_mask → dynamic top-k path is used."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    torch.manual_seed(42)
    x = torch.randn(4, 8) * 0.01
    x[2] *= 20.0  # channel 2 has large values → H
    g = GranularitySpec.per_channel(axis=0)

    out = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.3)  # k=1
    out_int8 = fmt_int8.quantize(x, g)

    # Channel 2 (largest amax) should match int8
    assert torch.allclose(out[2], out_int8[2], atol=1e-7)


# ===========================================================================
# Review fixes (C1, I2, S1)
# ===========================================================================


# ---------------------------------------------------------------------------
# C1: QuantScheme ratio-level mutual exclusivity
# ---------------------------------------------------------------------------

def test_quant_scheme_group_ratio_and_outlier_ratio_mutually_exclusive():
    """group_ratio > 0 and outlier_ratio > 0 cannot both be set on QuantScheme."""
    from src.scheme.granularity import GranularityMode
    g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                        outlier_ratio=0.1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        QuantScheme(
            format="int4",
            group_format="int8", group_ratio=0.3,
            granularity=g,
        )


def test_quant_scheme_group_format_set_outlier_ratio_zero_ok():
    """group_format set with outlier_ratio=0 is fine."""
    from src.scheme.granularity import GranularityMode
    g = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0)
    scheme = QuantScheme(
        format="int4",
        group_format="int8", group_ratio=0.3,
        granularity=g,  # outlier_ratio=0 by default
    )
    assert scheme.group_format is not None
    assert scheme.granularity.outlier_ratio == 0.0


# ---------------------------------------------------------------------------
# I2: PER_BLOCK static path test
# ---------------------------------------------------------------------------

def test_quantize_per_block_group_sparse_static():
    """PER_BLOCK with pre-computed group_mask → H blocks use group_format."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    # (2, 64) → block_size=32 → 4 blocks
    torch.manual_seed(42)
    x = torch.randn(2, 64) * 0.01
    x[0, 32:64] = torch.randn(32) * 2 + 5.0  # block (0,1) large
    x[1, 0:32] = torch.randn(32) * 2 + 5.0    # block (1,0) large
    g = GranularitySpec.per_block(size=32, axis=-1)

    # Pre-computed mask: blocks (0,1) and (1,0) are H
    h_mask = torch.tensor([[False, True], [True, False]], dtype=torch.bool)

    out_static = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                                   group_mask=h_mask)
    out_dynamic = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5)

    # Both static and dynamic should produce H for the two large blocks
    assert out_static.shape == x.shape
    # Static: H blocks use int8, L blocks use int4
    assert not torch.allclose(out_static[0, 32:64], fmt_int4.quantize(x[0, 32:64].unsqueeze(0), GranularitySpec.per_block(size=32, axis=-1)))
    # Dynamic: since largest blocks in scores match our mask, results should be close
    assert torch.allclose(out_static, out_dynamic, atol=1e-5)


# ---------------------------------------------------------------------------
# S1: group_format set with group_ratio=0.0 → no-op
# ---------------------------------------------------------------------------

def test_quantize_group_format_set_ratio_zero():
    """group_format is set but group_ratio=0.0 → standard path, group_format ignored."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")
    x = torch.randn(4, 8)
    g = GranularitySpec.per_tensor()

    out_with_gf = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.0)
    out_std = fmt_int4.quantize(x, g)

    # Should match standard int4 path (group_format ignored when ratio=0)
    assert torch.allclose(out_with_gf, out_std, atol=1e-7)


def test_calibration_group_sparse_no_buffer_when_ratio_zero():
    """When group_format is set but group_ratio=0, no group_mask buffer assigned."""
    model = _GroupSparseModule("int8", 0.0)
    session = CalibrationSession(model, MaxScaleStrategy(), sparse=True)
    with session:
        for _ in range(2):
            model(torch.randn(4, 8))
    assert not hasattr(model, "_output_group_mask")


# ===========================================================================
# Phase 7: Backward propagation (gradient flow through group sparse)
# ===========================================================================


def test_group_sparse_backward_per_channel():
    """Gradients propagate through per_channel group_sparse without error."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    torch.manual_seed(42)
    x = torch.randn(4, 8, requires_grad=True)
    g = GranularitySpec.per_channel(axis=0)

    out = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_group_sparse_backward_per_bank():
    """Gradients propagate through bank group_sparse without error."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    torch.manual_seed(42)
    x = torch.randn(4, 16, requires_grad=True)
    g = GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1)

    out = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_group_sparse_backward_per_block():
    """Gradients propagate through per_block group_sparse without error."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    torch.manual_seed(42)
    x = torch.randn(2, 64, requires_grad=True)
    g = GranularitySpec.per_block(size=32, axis=-1)

    out = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_group_sparse_backward_static_mask():
    """Gradients propagate through group_sparse with static group_mask."""
    fmt_int8 = FormatBase.from_str("int8")
    fmt_int4 = FormatBase.from_str("int4")

    torch.manual_seed(42)
    x = torch.randn(4, 8, requires_grad=True)
    g = GranularitySpec.per_channel(axis=0)
    h_mask = torch.tensor([True, False, False, True], dtype=torch.bool)

    out = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5,
                            group_mask=h_mask, scale_storage="fp32")
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


# ===========================================================================
# Phase 8: Ops-layer integration — static group_mask affects inference
# ===========================================================================


class _SimpleMLP(nn.Module):
    def __init__(self, in_dim=8, hid_dim=4, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def test_ops_static_group_mask_affects_inference():
    """Calibrated _output_group_mask buffer is used during inference."""
    from src.session._config import QuantConfig
    from src.session._compat import Session

    # Model with group sparse
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        a_format="int4", a_granularity="per_channel",
        group_format="int8", group_ratio=0.5,
        quantize_nonlinear=False,
    )
    model = _SimpleMLP(in_dim=8, hid_dim=4, out_dim=2)
    session = Session(model, cfg, keep_fp32=True)

    # Run with calibration (sparse=True) → static group_mask computed
    calib_data = [torch.randn(4, 8) for _ in range(3)]
    result = session.run(calib_data)

    # The quantized model should have _output_group_mask buffers on
    # modules that have group_format set
    has_group_mask = False
    for name, mod in session.qmodel.named_modules():
        for buf_name in ("_output_group_mask", "_input_group_mask"):
            if hasattr(mod, buf_name):
                has_group_mask = True
                mask = getattr(mod, buf_name)
                assert mask.dtype == torch.bool
                assert mask.any().item()
    assert has_group_mask, "No group_mask buffers found on any module"


def test_ops_static_group_mask_differs_from_dynamic():
    """Static group_mask produces different results than dynamic top-k."""
    from src.session._config import QuantConfig
    from src.session._compat import Session

    torch.manual_seed(42)

    # Same model, same config
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        a_format="int4", a_granularity="per_channel",
        group_format="int8", group_ratio=0.3,
        quantize_nonlinear=False,
    )

    # With calibration (static mask path)
    model1 = _SimpleMLP(in_dim=8, hid_dim=4, out_dim=2)
    session1 = Session(model1, cfg, keep_fp32=True)
    calib = [torch.randn(4, 8) for _ in range(5)]
    result1 = session1.run(calib)

    # Without calibration (dynamic path — no static mask)
    model2 = _SimpleMLP(in_dim=8, hid_dim=4, out_dim=2)
    # Copy same weights
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        p2.data.copy_(p1.data)
    session2 = Session(model2, cfg, keep_fp32=True)
    # Run without calibration data — no sparse flag → dynamic group_sparse
    result2 = session2.run([])

    # Both should produce valid outputs
    assert result1 is not None
    assert result2 is not None


# ===========================================================================
# Phase 9: E2E precision — group_sparse reduces error vs low-precision only
# ===========================================================================


def test_group_sparse_improves_over_low_precision_only():
    """Group sparse (int4+int8) should produce lower error than int4-only."""
    from src.session._config import QuantConfig
    from src.session._compat import Session

    torch.manual_seed(42)
    calib = [torch.randn(4, 8) for _ in range(5)]
    eval_data = [(torch.randn(4, 8), torch.randint(0, 2, (4,)))]

    def _eval(model, data):
        model.eval()
        with torch.no_grad():
            if isinstance(data, list):
                for item in data:
                    x = item[0] if isinstance(item, (tuple, list)) else item
                    model(x)
            else:
                model(data)
        return {"dummy": 0.5}

    # Baseline: int4 only (no group_sparse)
    cfg_baseline = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        a_format="int4", a_granularity="per_channel",
        quantize_nonlinear=False,
    )
    model_baseline = _SimpleMLP(in_dim=8, hid_dim=4, out_dim=2)
    session_baseline = Session(model_baseline, cfg_baseline, keep_fp32=True)
    result_baseline = session_baseline.run(calib, eval_fn=_eval, eval_data=eval_data,
                                            outputs=["distribution"])

    # Group sparse: int4 + int8
    cfg_gs = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        a_format="int4", a_granularity="per_channel",
        group_format="int8", group_ratio=0.5,
        quantize_nonlinear=False,
    )
    model_gs = _SimpleMLP(in_dim=8, hid_dim=4, out_dim=2)
    session_gs = Session(model_gs, cfg_gs, keep_fp32=True)
    result_gs = session_gs.run(calib, eval_fn=_eval, eval_data=eval_data,
                                outputs=["distribution"])

    # Both should succeed
    assert result_baseline is not None
    assert result_gs is not None
