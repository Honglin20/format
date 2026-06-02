"""
TDD tests: GranularitySpec outlier_ratio — per-bank outlier split quantization.

outlier_ratio > 0 triggers per-bank outlier/normal split within
PER_BLOCK granularity, using the same format but separate scales
for each group.  outlier_ratio == 0 is a no-op and must not
change existing PER_BLOCK behavior.

Mathematical derivation
------------------------
Given a bank B of n = block_size elements, k = max(1, floor(n * r))
with outlier ratio r in (0, 1]:

  1.  Select top-k by magnitude:  o_idx = argsort(|B|, descending)[:k]
  2.  Partition:  B_o = B[o_idx],  B_n = B[~o_idx]
  3.  Compute shared exponent per group (same MX algorithm):
        e_g = floor(log2(max(|B_g|) + ε))
      Scale: s_g = 2^{e_g - emax}
  4.  Normalize:  B̂_g = B_g / 2^{e_g}
  5.  Quantize elemwise:  Q_g = quantize_elemwise(B̂_g)
  6.  Rescale:     B̃_g = Q_g * 2^{e_g}
  7.  Recompose:   result[o_idx] = B̃_o,  result[~o_idx] = B̃_n

When r == 0 (k == 0): degenerate to standard PER_BLOCK (single group).
When r == 1 (k == n): all elements in one group, same as PER_BLOCK.

Per-bank advantage: if B = [0.1, 0.2, ..., 20.0], standard PER_BLOCK
uses e = floor(log2(20.0)) = 4, crushing 0.1/16 ≈ 0.  With outlier
split, e_n ≈ floor(log2(0.5)) = -1, preserving 8× more resolution
for the normal group.
"""
import pytest
import torch

from src.formats.base import FormatBase
from src.scheme.granularity import GranularitySpec, GranularityMode


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

def test_outlier_ratio_defaults_to_zero():
    """outlier_ratio defaults to 0.0 — no behavior change."""
    spec = GranularitySpec.per_block(32)
    assert spec.outlier_ratio == 0.0


def test_outlier_ratio_rejects_negative():
    """outlier_ratio must be in [0, 1]."""
    with pytest.raises(ValueError):
        GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=32,
                        outlier_ratio=-0.1)


def test_outlier_ratio_rejects_above_one():
    """outlier_ratio must be in [0, 1]."""
    with pytest.raises(ValueError):
        GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=32,
                        outlier_ratio=1.01)


def test_outlier_ratio_valid_all_modes():
    """outlier_ratio > 0 is valid with all granularity modes (not just PER_BLOCK)."""
    # PER_TENSOR + outlier_ratio → constructs without error
    s1 = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1)
    assert s1.outlier_ratio == 0.1
    assert s1.mode == GranularityMode.PER_TENSOR

    # PER_CHANNEL + outlier_ratio → constructs without error
    s2 = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                          outlier_ratio=0.05)
    assert s2.outlier_ratio == 0.05
    assert s2.mode == GranularityMode.PER_CHANNEL


def test_outlier_ratio_requires_positive_block_size():
    """outlier_ratio > 0 with block_size=0 is invalid."""
    with pytest.raises(ValueError):
        GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=0,
                        outlier_ratio=0.1)


def test_outlier_ratio_valid_values():
    """Valid outlier_ratio configurations should construct without error."""
    # ratio 0 with PER_BLOCK (default)
    s1 = GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=64,
                         outlier_ratio=0.0)
    assert s1.outlier_ratio == 0.0

    # ratio 0.125 with PER_BLOCK
    s2 = GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=32,
                         outlier_ratio=0.125)
    assert s2.outlier_ratio == 0.125

    # ratio 1.0 with PER_BLOCK
    s3 = GranularitySpec(mode=GranularityMode.PER_BLOCK, block_size=32,
                         outlier_ratio=1.0)
    assert s3.outlier_ratio == 1.0

    # ratio 0 with non-PER_BLOCK mode (happens to use default)
    s4 = GranularitySpec.per_channel(axis=0)
    assert s4.outlier_ratio == 0.0

    # ratio 0.1 with PER_TENSOR (now valid per ADR-011)
    s5 = GranularitySpec(mode=GranularityMode.PER_TENSOR, outlier_ratio=0.1)
    assert s5.outlier_ratio == 0.1

    # ratio 0.05 with PER_CHANNEL (now valid per ADR-011)
    s6 = GranularitySpec(mode=GranularityMode.PER_CHANNEL, channel_axis=0,
                          outlier_ratio=0.05)
    assert s6.outlier_ratio == 0.05


# ---------------------------------------------------------------------------
# Degeneracy: outlier_ratio=0 is identical to standard PER_BLOCK
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt_name", ["int8", "fp8_e4m3", "fp4_e2m1"])
def test_outlier_ratio_zero_equals_per_block(fmt_name):
    """When outlier_ratio=0, result must equal standard PER_BLOCK path."""
    torch.manual_seed(42)
    x = torch.randn(4, 64)

    fmt = FormatBase.from_str(fmt_name)

    gran_standard = GranularitySpec.per_block(32)
    gran_outlier0 = GranularitySpec(
        mode=GranularityMode.PER_BLOCK, block_size=32, outlier_ratio=0.0,
    )

    result_standard = fmt.quantize(x, gran_standard, "nearest")
    result_outlier0 = fmt.quantize(x, gran_outlier0, "nearest")

    assert torch.allclose(result_standard, result_outlier0, atol=1e-7), \
        f"{fmt_name}: outlier_ratio=0 differs from per_block. " \
        f"max diff = {(result_standard - result_outlier0).abs().max()}"


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt_name", ["int8", "fp8_e4m3", "int4", "fp4_e2m1"])
def test_outlier_bank_preserves_shape(fmt_name):
    """Outlier bank quantization must not change tensor shape."""
    torch.manual_seed(42)
    for shape in [(4, 64), (2, 3, 32), (8, 128)]:
        x = torch.randn(*shape)
        fmt = FormatBase.from_str(fmt_name)
        gran = GranularitySpec(
            mode=GranularityMode.PER_BLOCK, block_size=32,
            outlier_ratio=0.125,
        )
        result = fmt.quantize(x, gran, "nearest")
        assert result.shape == x.shape, \
            f"{fmt_name} shape {shape}: got {result.shape}"


@pytest.mark.parametrize("fmt_name", ["int8", "fp8_e4m3", "int4"])
def test_outlier_bank_produces_finite_output(fmt_name):
    """All outputs must be finite (no NaN, no Inf)."""
    torch.manual_seed(42)
    x = torch.randn(8, 64)

    fmt = FormatBase.from_str(fmt_name)
    gran = GranularitySpec(
        mode=GranularityMode.PER_BLOCK, block_size=32,
        outlier_ratio=0.125,
    )
    result = fmt.quantize(x, gran, "nearest")
    assert result.isfinite().all(), \
        f"{fmt_name}: outlier_bank produced non-finite values"


# ---------------------------------------------------------------------------
# Correctness: outlier_bank improves outlier-heavy blocks
# ---------------------------------------------------------------------------

def test_outlier_bank_preserves_small_values_better():
    """In a block with one large outlier, small normals keep more resolution.

    Proof: Block B = [v_small × 31, v_large × 1], v_large ≫ v_small.
    Standard PER_BLOCK: e = floor(log2(v_large)), crushing v_small.
    Outlier bank: e_n = floor(log2(v_small)), preserving v_small at full scale.

    We measure this by checking that small input values survive quantization
    (output ≠ 0) with outlier_bank but can be crushed with standard PER_BLOCK.
    """
    torch.manual_seed(42)
    fmt = FormatBase.from_str("int4")

    # Build a 2D tensor: 2 blocks, each (32,) along last dim.
    x = torch.ones(2, 32) * 0.1

    # Insert one large outlier per block
    x[0, 0] = 20.0
    x[1, 16] = 20.0

    gran_standard = GranularitySpec.per_block(32)
    gran_outlier = GranularitySpec(
        mode=GranularityMode.PER_BLOCK, block_size=32,
        outlier_ratio=0.125,  # k=4 outliers per block
    )

    result_standard = fmt.quantize(x, gran_standard, "nearest")
    result_outlier = fmt.quantize(x, gran_outlier, "nearest")

    # After standard PER_BLOCK, small normal values near 0.1 may be crushed
    # to 0 or the smallest representable value because the shared exponent is
    # dominated by the 20.0 outlier.
    # With outlier_bank, the normal group gets its own scale → better resolution.

    # Metric: mean absolute error for the normal positions (all except the
    # outlier at index 0 of first block, index 16 of second block).
    normal_mask = torch.ones(2, 32, dtype=torch.bool)
    normal_mask[0, 0] = False
    normal_mask[1, 16] = False

    mae_standard = (result_standard[normal_mask] - x[normal_mask]).abs().mean()
    mae_outlier = (result_outlier[normal_mask] - x[normal_mask]).abs().mean()

    # Outlier bank should have strictly lower error on normals.
    assert mae_outlier < mae_standard, \
        f"Expected outlier_bank MAE ({mae_outlier:.6f}) < standard MAE ({mae_standard:.6f})"


def test_outlier_bank_each_block_independent():
    """Each bank selects outliers independently based on local magnitudes.

    Block 0: large values at positions [0, 1], small elsewhere.
    Block 1: large values at positions [16, 17], small elsewhere.

    With k=2, each bank should pick its own local outliers, not
    a global set.
    """
    torch.manual_seed(42)
    fmt = FormatBase.from_str("int8")

    x = torch.ones(2, 32) * 0.5  # 2 blocks along last dim
    x[0, 0] = 10.0
    x[0, 1] = 9.0
    x[1, 16] = 10.0
    x[1, 17] = 9.0

    gran = GranularitySpec(
        mode=GranularityMode.PER_BLOCK, block_size=32,
        outlier_ratio=0.0625,  # k=2 outliers per block
    )

    result = fmt.quantize(x, gran, "nearest")

    # If each block correctly identifies its own outliers:
    # Block 0: positions 0,1 get outlier scale → preserved near 10.0, 9.0
    # Block 1: positions 16,17 get outlier scale → preserved near 10.0, 9.0

    # The outliers should be close to their original values.
    assert (result[0, 0] - 10.0).abs() < 1.0, \
        f"Block 0 outlier at [0,0] should be preserved, got {result[0, 0]}"
    assert (result[1, 16] - 10.0).abs() < 1.0, \
        f"Block 1 outlier at [1,16] should be preserved, got {result[1, 16]}"


def test_outlier_bank_same_format_both_groups():
    """Both outlier and normal groups use the SAME format (not dual format).

    Verify by checking that for a uniform tensor (all values equal),
    outliers and normals get similar quantization error.
    """
    torch.manual_seed(42)
    fmt = FormatBase.from_str("int4")

    # Uniform values — outliers and normals should have same scale
    x = torch.ones(4, 32) * 0.5

    gran = GranularitySpec(
        mode=GranularityMode.PER_BLOCK, block_size=32,
        outlier_ratio=0.125,  # k=4 per block
    )

    result = fmt.quantize(x, gran, "nearest")

    # All values are the same, so all quantization errors should be similar.
    errors = (result - x).abs()
    # The max error should be reasonable (< 20% relative error for 0.5)
    assert errors.max() < 0.2, \
        f"Uniform tensor: max error {errors.max():.4f} too large for same-format quantization"


# ---------------------------------------------------------------------------
# Extreme ratios
# ---------------------------------------------------------------------------

def test_outlier_ratio_one_equals_single_group():
    """outlier_ratio=1.0: all elements are outliers → single group → same as PER_BLOCK."""
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    fmt = FormatBase.from_str("int8")

    gran_standard = GranularitySpec.per_block(32)
    gran_all_outlier = GranularitySpec(
        mode=GranularityMode.PER_BLOCK, block_size=32,
        outlier_ratio=1.0,
    )

    result_standard = fmt.quantize(x, gran_standard, "nearest")
    result_all_outlier = fmt.quantize(x, gran_all_outlier, "nearest")

    # When all elements are in one group, it's equivalent to PER_BLOCK.
    assert torch.allclose(result_standard, result_all_outlier, atol=1e-7), \
        f"outlier_ratio=1.0 should equal standard PER_BLOCK. " \
        f"max diff = {(result_standard - result_all_outlier).abs().max()}"


def test_outlier_bank_small_ratio():
    """Very small outlier_ratio (k=1 per bank) still works correctly."""
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    fmt = FormatBase.from_str("int8")

    gran = GranularitySpec(
        mode=GranularityMode.PER_BLOCK, block_size=32,
        outlier_ratio=0.01,  # k = max(1, floor(32*0.01)) = 1
    )
    result = fmt.quantize(x, gran, "nearest")
    assert result.shape == x.shape
    assert result.isfinite().all()


# ---------------------------------------------------------------------------
# Different block_axis
# ---------------------------------------------------------------------------

def test_outlier_bank_block_axis_0():
    """outlier_bank respects block_axis parameter."""
    torch.manual_seed(42)
    # Shape (64, 8): block along axis 0
    x = torch.randn(64, 8)
    # Put outliers in specific block
    x[0, :] = 20.0

    fmt = FormatBase.from_str("int8")
    gran = GranularitySpec(
        mode=GranularityMode.PER_BLOCK, block_size=32,
        block_axis=0, outlier_ratio=0.125,
    )
    result = fmt.quantize(x, gran, "nearest")
    assert result.shape == x.shape
    assert result.isfinite().all()
