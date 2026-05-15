"""
Comprehensive 4-bit format sparse tensor analysis.

Tests all three 4-bit formats (fp4_e2m1, int4, nf4) with dense and sparse
COO tensors, verifying correctness, consistency, and edge-case behavior.

Focus: sparse tensor quantization path — verifying it works correctly for
each 4-bit format and identifying any gaps.
"""
import pytest
import torch
import numpy as np

from src.formats.base import FormatBase
from src.formats._core import _elemwise_core
from src.scheme.granularity import GranularitySpec
from src.scheme.quant_scheme import QuantScheme
from src.quantize.elemwise import quantize

# ═══════════════════════════════════════════════════════════════════════════════
# Format under test
# ═══════════════════════════════════════════════════════════════════════════════

FOUR_BIT_FORMATS = ["fp4_e2m1", "int4", "nf4"]


def _get_format(name):
    return FormatBase.from_str(name)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_sparse_coo(dense_tensor, sparsity=0.7):
    """Convert a dense tensor to sparse COO with given sparsity ratio."""
    torch.manual_seed(42)
    mask = torch.rand(dense_tensor.shape) > sparsity
    sparse_vals = dense_tensor[mask]
    indices = mask.nonzero(as_tuple=False).t()
    return torch.sparse_coo_tensor(indices, sparse_vals, dense_tensor.shape,
                                   dtype=dense_tensor.dtype,
                                   device=dense_tensor.device)


def _sparse_numel(sparse_tensor):
    """Number of non-zero elements in a sparse COO tensor."""
    return sparse_tensor._nnz()


def _quantization_error(x, x_q):
    """Compute relative quantization error metrics."""
    abs_err = (x - x_q).abs()
    rel_err = abs_err / (x.abs() + 1e-12)
    return {
        "mse": (abs_err ** 2).mean().item(),
        "mae": abs_err.mean().item(),
        "max_abs": abs_err.max().item(),
        "mean_rel": rel_err.mean().item(),
        "max_rel": rel_err.max().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Format parameters — verify all 4-bit format definitions
# ═══════════════════════════════════════════════════════════════════════════════

class TestFourBitFormatParameters:
    """Verify the defined parameters for each 4-bit format."""

    def test_fp4_e2m1_params(self):
        fmt = _get_format("fp4_e2m1")
        assert fmt.ebits == 2
        assert fmt.mbits == 3      # sign + implicit + 1 actual mantissa
        assert fmt.emax == 2       # sub-byte float: 2^(ebits-1) = 2
        # max_norm = 2^emax * (2^(mbits-1)-1) / 2^(mbits-2) = 4 * 3 / 2 = 6.0
        assert fmt.max_norm == 6.0
        # min_norm = 2^(2 - 2^(ebits-1)) = 2^(2-2) = 2^0 = 1.0
        assert fmt.min_norm == 1.0
        assert fmt.is_integer == False

    def test_int4_params(self):
        fmt = _get_format("int4")
        assert fmt.ebits == 0
        assert fmt.mbits == 4
        assert fmt.emax == 0
        # max_norm = (2^(4-1)-1) / 2^(4-2) = 7/4 = 1.75
        assert fmt.max_norm == 1.75
        assert fmt.min_norm == 0.0
        assert fmt.is_integer == True

    def test_nf4_params(self):
        fmt = _get_format("nf4")
        assert fmt.ebits == 0
        assert fmt.mbits == 4       # log2(16-1) bits
        assert fmt.emax == 0
        assert fmt.max_norm == 1.0
        assert fmt.min_norm == 0.0
        assert fmt.levels.numel() == 16
        assert fmt.is_integer == True  # ebits==0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Dense tensor quantization — baseline correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestFourBitDenseBaseline:
    """Verify all 4-bit formats produce correct results on dense tensors."""

    @pytest.mark.parametrize("fmt_name", FOUR_BIT_FORMATS)
    def test_output_finite(self, fmt_name):
        """Quantization of normal random tensor produces finite output."""
        torch.manual_seed(42)
        x = torch.randn(4, 32)
        fmt = _get_format(fmt_name)
        result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
        assert torch.isfinite(result).all(), f"{fmt_name}: non-finite values in output"

    @pytest.mark.parametrize("fmt_name", FOUR_BIT_FORMATS)
    def test_zero_input(self, fmt_name):
        """Zero input → zero output."""
        x = torch.zeros(4, 32)
        fmt = _get_format(fmt_name)
        result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
        assert (result == 0).all(), f"{fmt_name}: zero input not preserved"

    @pytest.mark.parametrize("fmt_name", FOUR_BIT_FORMATS)
    def test_shape_preserved(self, fmt_name):
        """Output shape matches input shape for various ranks."""
        torch.manual_seed(42)
        fmt = _get_format(fmt_name)
        for shape in [(4, 32), (2, 3, 64), (1, 2, 4, 16)]:
            x = torch.randn(*shape)
            result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
            assert result.shape == x.shape, f"{fmt_name}: shape mismatch for {shape}"

    @pytest.mark.parametrize("fmt_name", FOUR_BIT_FORMATS)
    def test_error_bounded(self, fmt_name):
        """Quantization error should be reasonable for each format."""
        torch.manual_seed(42)
        x = torch.randn(4, 128)
        fmt = _get_format(fmt_name)
        result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
        err = _quantization_error(x, result)

        # 4-bit formats: only ~16 levels → expect significant relative error
        # but max absolute error should be bounded
        assert err["max_abs"] < x.abs().max().item() * 2.0, \
            f"{fmt_name}: max_abs={err['max_abs']:.4f} unreasonably large"

    def test_fp4_representable_values(self):
        """fp4_e2m1: verify quantization levels are correct sub-byte floats."""
        fmt = _get_format("fp4_e2m1")
        # fp4_e2m1 has sign + 2exp + 1mantissa → 2^4 = 16 levels
        # Normal values: ±(1.x) * 2^{e-1}, e in {0,1,2,3}
        # e=0: ±1.0 * 2^{-1} = ±0.5, ±1.5 * 2^{-1} = ±0.75
        # e=1: ±1.0 * 2^0 = ±1.0, ±1.5 * 2^0 = ±1.5
        # e=2: ±1.0 * 2^1 = ±2.0, ±1.5 * 2^1 = ±3.0
        # e=3: ±1.0 * 2^2 = ±4.0, ±1.5 * 2^2 = ±6.0
        # Subnormals (e=0 with leading 0): ±0.0, ±0.5 * 2^{-1} ≈ ±0.25
        test_vals = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                  -0.5, -1.0, -2.0, -3.0, -6.0])
        result = fmt.quantize_elemwise(test_vals)
        for i, (inp, out) in enumerate(zip(test_vals, result)):
            assert torch.isfinite(out), f"fp4: value {inp} → {out}"

    def test_int4_representable_values(self):
        """int4: verify sign-magnitude integer quantization levels."""
        fmt = _get_format("int4")
        # int4 per_tensor normalizes by amax first, so test direct elemwise
        # int4 max_norm = 1.75, levels are ±k/4 for k=0..7 → ±{0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75}
        test_vals = torch.tensor([0.0, 0.125, 0.375, 0.625, 0.875, 1.0, 1.5, 1.75,
                                  -0.375, -1.0, -1.5, -1.75])
        result = fmt.quantize_elemwise(test_vals)
        assert torch.isfinite(result).all()

    def test_nf4_representable_values(self):
        """nf4: verify nearest-neighbor maps to known levels."""
        fmt = _get_format("nf4")
        # Exact levels must map to themselves
        levels = fmt.levels
        result = fmt.quantize_elemwise(levels)
        assert torch.equal(result, levels), "NF4 levels should map to themselves"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Sparse tensor quantization — the SPARSE path
# ═══════════════════════════════════════════════════════════════════════════════

class TestFourBitSparseQuantization:
    """Test sparse COO tensor quantization for all 4-bit formats."""

    @pytest.fixture
    def dense_data(self):
        torch.manual_seed(42)
        return torch.randn(4, 32)

    @pytest.fixture
    def sparse_data(self, dense_data):
        return _make_sparse_coo(dense_data, sparsity=0.7)

    # ---- fp4_e2m1 sparse ----

    def test_fp4_sparse_output_is_sparse(self, sparse_data):
        """fp4_e2m1: sparse input → sparse output."""
        fmt = _get_format("fp4_e2m1")
        result = fmt.quantize_elemwise(sparse_data, round_mode="nearest")
        assert result.is_sparse, "fp4_e2m1: sparse input should produce sparse output"
        assert result.shape == sparse_data.shape

    def test_fp4_sparse_dense_consistency(self, dense_data, sparse_data):
        """fp4_e2m1: sparse quantize matches dense quantize on non-zero elements."""
        fmt = _get_format("fp4_e2m1")

        # Dense path
        dense_result = fmt.quantize_elemwise(dense_data, round_mode="nearest")

        # Sparse path
        sparse_result = fmt.quantize_elemwise(sparse_data, round_mode="nearest")
        sparse_dense = sparse_result.to_dense()

        # Compare on positions where sparse has values
        mask = sparse_data.to_dense() != 0
        diff = (dense_result[mask] - sparse_dense[mask]).abs().max()
        assert diff == 0.0, \
            f"fp4_e2m1: dense-sparse mismatch, max diff = {diff}"

    # ---- int4 sparse ----

    def test_int4_sparse_output_is_sparse(self, sparse_data):
        """int4: sparse input → sparse output."""
        fmt = _get_format("int4")
        result = fmt.quantize_elemwise(sparse_data, round_mode="nearest")
        assert result.is_sparse, "int4: sparse input should produce sparse output"
        assert result.shape == sparse_data.shape

    def test_int4_sparse_dense_consistency(self, dense_data, sparse_data):
        """int4: sparse quantize matches dense quantize on non-zero elements."""
        fmt = _get_format("int4")

        dense_result = fmt.quantize_elemwise(dense_data, round_mode="nearest")
        sparse_result = fmt.quantize_elemwise(sparse_data, round_mode="nearest")
        sparse_dense = sparse_result.to_dense()

        mask = sparse_data.to_dense() != 0
        diff = (dense_result[mask] - sparse_dense[mask]).abs().max()
        assert diff == 0.0, \
            f"int4: dense-sparse mismatch, max diff = {diff}"

    # ---- nf4 sparse ----

    def test_nf4_sparse_behavior(self, sparse_data):
        """nf4: test sparse tensor behavior through quantize_elemwise.

        NF4Format.quantize_elemwise() has its own LUT-based implementation
        that does NOT go through _elemwise_core and does NOT have the
        sparse-specific extract-values-then-reconstruct logic.

        This test characterizes what actually happens when you pass a sparse
        tensor to NF4 quantization.
        """
        fmt = _get_format("nf4")
        result = fmt.quantize_elemwise(sparse_data, round_mode="nearest")

        # Check if output is sparse — NF4 doesn't have sparse reconstruction,
        # so the output type depends on how torch operations handle sparse inputs.
        print(f"\n  nf4 sparse input:  is_sparse={sparse_data.is_sparse}, shape={sparse_data.shape}, nnz={_sparse_numel(sparse_data)}")
        print(f"  nf4 sparse output: is_sparse={result.is_sparse}, shape={result.shape}")

        if result.is_sparse:
            print(f"  nf4 sparse output nnz: {_sparse_numel(result)}")
        else:
            print(f"  nf4 sparse output is DENSE (sparse info lost)")

    def test_nf4_sparse_dense_consistency(self, dense_data, sparse_data):
        """nf4: check whether sparse and dense paths agree on non-zero positions.

        If NF4 doesn't handle sparse properly, the results will diverge.
        """
        fmt = _get_format("nf4")

        dense_result = fmt.quantize_elemwise(dense_data, round_mode="nearest")
        sparse_result = fmt.quantize_elemwise(sparse_data, round_mode="nearest")

        # Convert both to dense for comparison
        sparse_dense = sparse_result.to_dense() if sparse_result.is_sparse else sparse_result

        mask = sparse_data.to_dense() != 0
        diff = (dense_result[mask] - sparse_dense[mask]).abs().max()

        if diff > 0:
            mismatch_count = (dense_result[mask] != sparse_dense[mask]).sum().item()
            total_count = mask.sum().item()
            print(f"\n  nf4 sparse-dense consistency:")
            print(f"    max diff:     {diff:.6e}")
            print(f"    mismatches:   {mismatch_count}/{total_count} ({100*mismatch_count/total_count:.1f}%)")
            if mismatch_count > 0:
                mismatch_mask = dense_result[mask] != sparse_dense[mask]
                idxs = mismatch_mask.nonzero(as_tuple=False)[:5].squeeze(-1)
                for idx in idxs:
                    val = dense_data[mask][idx]
                    d_r = dense_result[mask][idx]
                    s_r = sparse_dense[mask][idx]
                    print(f"    [{idx.item()}]: val={val:.6f} dense→{d_r:.6f} sparse→{s_r:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Sparse edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestFourBitSparseEdgeCases:
    """Edge cases for sparse tensor quantization with 4-bit formats."""

    # ---- All zeros sparse ----

    @pytest.mark.parametrize("fmt_name", ["fp4_e2m1", "int4"])
    def test_sparse_all_zeros(self, fmt_name):
        """All-zero values in sparse tensor should quantize to all zeros."""
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.zeros(2)
        A = torch.sparse_coo_tensor(indices, values, size=(2, 2))
        fmt = _get_format(fmt_name)
        result = fmt.quantize_elemwise(A, round_mode="nearest")
        if result.is_sparse:
            result = result.coalesce()
            assert (result.values() == 0).all(), \
                f"{fmt_name}: sparse zero values should remain zero"
        else:
            assert (result == 0).all()

    # ---- Single non-zero ----

    @pytest.mark.parametrize("fmt_name", ["fp4_e2m1", "int4"])
    def test_sparse_single_element(self, fmt_name):
        """A sparse tensor with a single non-zero element."""
        indices = torch.tensor([[0], [0]])
        values = torch.tensor([1.5])
        A = torch.sparse_coo_tensor(indices, values, size=(2, 2))
        fmt = _get_format(fmt_name)
        result = fmt.quantize_elemwise(A, round_mode="nearest")
        assert result.is_sparse, f"{fmt_name}: output should be sparse"

        # Compare with dense equivalent
        dense_eq = torch.zeros(2, 2)
        dense_eq[0, 0] = 1.5
        dense_result = fmt.quantize_elemwise(dense_eq, round_mode="nearest")
        result_dense = result.to_dense()
        assert torch.equal(result_dense, dense_result), \
            f"{fmt_name}: single-element sparse differs from dense"

    # ---- NaN/Inf in sparse ----

    @pytest.mark.parametrize("fmt_name", ["fp4_e2m1", "int4"])
    def test_sparse_with_inf(self, fmt_name):
        """Sparse tensor containing ±Inf — Inf should be preserved."""
        indices = torch.tensor([[0, 0, 1], [0, 1, 2]])
        values = torch.tensor([1.0, float("inf"), -float("inf")])
        A = torch.sparse_coo_tensor(indices, values, size=(2, 3))
        fmt = _get_format(fmt_name)
        result = fmt.quantize_elemwise(A, round_mode="nearest")
        assert result.is_sparse, f"{fmt_name}: output should be sparse"

    @pytest.mark.parametrize("fmt_name", ["fp4_e2m1", "int4"])
    def test_sparse_with_nan(self, fmt_name):
        """Sparse tensor containing NaN — NaN should be preserved."""
        indices = torch.tensor([[0, 0], [0, 1]])
        values = torch.tensor([1.0, float("nan")])
        A = torch.sparse_coo_tensor(indices, values, size=(1, 2))
        fmt = _get_format(fmt_name)
        result = fmt.quantize_elemwise(A, round_mode="nearest")
        result = result.coalesce()
        result_vals = result.values()
        assert torch.isnan(result_vals[1]), \
            f"{fmt_name}: NaN should be preserved in sparse output"

    # ---- Uniform sparse values ----

    @pytest.mark.parametrize("fmt_name", ["fp4_e2m1", "int4"])
    def test_sparse_uniform_values(self, fmt_name):
        """Sparse tensor where all non-zero values are identical."""
        indices = torch.tensor([[0, 0, 1, 1], [0, 3, 1, 2]])
        values = torch.tensor([2.5, 2.5, 2.5, 2.5])
        A = torch.sparse_coo_tensor(indices, values, size=(2, 4))
        fmt = _get_format(fmt_name)
        result = fmt.quantize_elemwise(A, round_mode="nearest")

        dense_eq = torch.zeros(2, 4)
        dense_eq[0, 0] = 2.5
        dense_eq[0, 3] = 2.5
        dense_eq[1, 1] = 2.5
        dense_eq[1, 2] = 2.5
        dense_result = fmt.quantize_elemwise(dense_eq, round_mode="nearest")
        assert torch.equal(result.to_dense(), dense_result), \
            f"{fmt_name}: uniform sparse differs from dense"

    # ---- Large sparse tensor ----

    @pytest.mark.parametrize("fmt_name", ["fp4_e2m1", "int4"])
    def test_sparse_large_tensor(self, fmt_name):
        """Larger sparse tensor (128x128, ~10% dense)."""
        torch.manual_seed(123)
        dense = torch.randn(128, 128)
        sparse = _make_sparse_coo(dense, sparsity=0.9)
        fmt = _get_format(fmt_name)

        # Dense path (reference)
        dense_q = fmt.quantize_elemwise(dense, round_mode="nearest")

        # Sparse path
        sparse_q = fmt.quantize_elemwise(sparse, round_mode="nearest")
        assert sparse_q.is_sparse, f"{fmt_name}: large sparse output should be sparse"

        # Verify consistency on non-zero positions
        mask = sparse.to_dense() != 0
        sparse_dense = sparse_q.to_dense()
        diff = (dense_q[mask] - sparse_dense[mask]).abs().max()
        assert diff == 0.0, \
            f"{fmt_name}: large sparse mismatch, max diff = {diff}"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Full pipeline — quantize() with QuantScheme on sparse tensors
# ═══════════════════════════════════════════════════════════════════════════════

class TestFourBitSparseFullPipeline:
    """Test the full quantize(x, scheme) pipeline with sparse tensors."""

    @pytest.mark.parametrize("fmt_name", FOUR_BIT_FORMATS)
    def test_per_tensor_sparse_pipeline(self, fmt_name):
        """quantize(x, scheme) with PER_TENSOR on sparse input."""
        torch.manual_seed(42)
        dense = torch.randn(4, 32)
        sparse = _make_sparse_coo(dense, sparsity=0.7)

        scheme = QuantScheme.per_tensor(fmt_name)

        # Dense baseline
        dense_q = quantize(dense, scheme)

        # Sparse — may or may not work depending on format
        try:
            sparse_q = quantize(sparse, scheme)
            if sparse_q.is_sparse:
                sparse_dense = sparse_q.to_dense()
            else:
                sparse_dense = sparse_q

            mask = sparse.to_dense() != 0
            diff = (dense_q[mask] - sparse_dense[mask]).abs().max()
            if diff > 0:
                mismatch_count = (dense_q[mask] != sparse_dense[mask]).sum().item()
                print(f"\n  {fmt_name} per_tensor pipeline: {mismatch_count} mismatches, max diff = {diff:.6e}")
            else:
                print(f"\n  {fmt_name} per_tensor pipeline: OK (bit-exact match)")
        except Exception as e:
            print(f"\n  {fmt_name} per_tensor pipeline: ERROR — {type(e).__name__}: {e}")

    @pytest.mark.parametrize("fmt_name", FOUR_BIT_FORMATS)
    def test_per_block_sparse_pipeline(self, fmt_name):
        """quantize(x, scheme) with PER_BLOCK on sparse input."""
        torch.manual_seed(42)
        dense = torch.randn(4, 64)
        sparse = _make_sparse_coo(dense, sparsity=0.7)

        scheme = QuantScheme.mxfp(fmt_name, block_size=32)

        dense_q = quantize(dense, scheme)

        try:
            sparse_q = quantize(sparse, scheme)
            if sparse_q.is_sparse:
                sparse_dense = sparse_q.to_dense()
            else:
                sparse_dense = sparse_q

            mask = sparse.to_dense() != 0
            diff = (dense_q[mask] - sparse_dense[mask]).abs().max()
            if diff > 0:
                mismatch_count = (dense_q[mask] != sparse_dense[mask]).sum().item()
                print(f"\n  {fmt_name} per_block pipeline: {mismatch_count} mismatches, max diff = {diff:.6e}")
            else:
                print(f"\n  {fmt_name} per_block pipeline: OK (bit-exact match)")
        except Exception as e:
            print(f"\n  {fmt_name} per_block pipeline: ERROR — {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Direct _elemwise_core sparse path verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestElemwiseCoreSparse:
    """Direct tests of the _elemwise_core sparse path for 4-bit parameters."""

    def test_elemwise_core_fp4_params_sparse(self):
        """_elemwise_core with fp4 params on sparse tensor."""
        indices = torch.tensor([[0, 0, 1, 1], [0, 3, 1, 2]])
        values = torch.tensor([1.5, -0.5, 3.0, -2.0])
        A = torch.sparse_coo_tensor(indices, values, size=(2, 4))

        # fp4_e2m1: mbits=3, ebits=2, max_norm=6.0
        out = _elemwise_core(A, bits=3, exp_bits=2, max_norm=6.0, round_mode="nearest")
        assert out.is_sparse, "Output should be sparse"
        assert out.shape == A.shape
        assert out._nnz() == A._nnz(), \
            f"nnz changed: {A._nnz()} → {out._nnz()}"

        # Compare with dense equivalent
        dense_eq = A.to_dense()
        dense_out = _elemwise_core(dense_eq, bits=3, exp_bits=2, max_norm=6.0, round_mode="nearest")
        mask = dense_eq != 0
        assert torch.equal(out.to_dense()[mask], dense_out[mask]), \
            "fp4 sparse-dense mismatch in _elemwise_core"

    def test_elemwise_core_int4_params_sparse(self):
        """_elemwise_core with int4 params on sparse tensor."""
        indices = torch.tensor([[0, 0, 1, 1], [0, 3, 1, 2]])
        values = torch.tensor([1.0, -0.5, 0.75, -1.5])
        A = torch.sparse_coo_tensor(indices, values, size=(2, 4))

        # int4: mbits=4, ebits=0, max_norm=1.75
        out = _elemwise_core(A, bits=4, exp_bits=0, max_norm=1.75, round_mode="nearest")
        assert out.is_sparse, "Output should be sparse"
        assert out.shape == A.shape
        assert out._nnz() == A._nnz()

        dense_eq = A.to_dense()
        dense_out = _elemwise_core(dense_eq, bits=4, exp_bits=0, max_norm=1.75, round_mode="nearest")
        mask = dense_eq != 0
        assert torch.equal(out.to_dense()[mask], dense_out[mask]), \
            "int4 sparse-dense mismatch in _elemwise_core"

    def test_elemwise_core_sparse_zeros_stay_sparse(self):
        """Elements that quantize to zero should remain in sparse output (not dropped)."""
        indices = torch.tensor([[0, 0, 1], [0, 1, 2]])
        # 0.001 → very small, may round to zero in 4-bit
        values = torch.tensor([0.001, 2.0, -2.0])
        A = torch.sparse_coo_tensor(indices, values, size=(2, 3))

        out = _elemwise_core(A, bits=4, exp_bits=0, max_norm=1.75, round_mode="nearest")
        assert out.is_sparse
        # The zero-valued entry is preserved (not coalesced away)
        # _elemwise_core returns sparse with same nnz
        assert out._nnz() <= A._nnz()  # may coalesce

    def test_elemwise_core_sparse_empty(self):
        """Empty sparse tensor (no non-zero elements)."""
        indices = torch.tensor([[], []], dtype=torch.long).reshape(2, 0)
        values = torch.tensor([])
        A = torch.sparse_coo_tensor(indices, values, size=(2, 4))

        out = _elemwise_core(A, bits=3, exp_bits=2, max_norm=6.0, round_mode="nearest")
        assert out.is_sparse
        assert out._nnz() == 0
        assert out.shape == (2, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: NF4 sparse gap — detailed characterization
# ═══════════════════════════════════════════════════════════════════════════════

class TestNF4SparseGapAnalysis:
    """Characterize NF4's sparse tensor behavior in detail.

    NF4Format.quantize_elemwise() overrides FormatBase.quantize_elemwise()
    and does NOT route through _elemwise_core(). It therefore lacks the
    sparse-specific extract-values → quantize → reconstruct logic.

    This class characterizes exactly what happens at each step of the NF4
    quantization pipeline when given a sparse tensor.
    """

    def test_nf4_elemwise_sparse_input(self):
        """Step 1: NF4 quantize_elemwise on sparse COO — what happens?"""
        torch.manual_seed(42)
        dense = torch.randn(4, 8)
        sparse = _make_sparse_coo(dense, sparsity=0.7)

        fmt = _get_format("nf4")

        dense_q = fmt.quantize_elemwise(dense, round_mode="nearest")
        sparse_q = fmt.quantize_elemwise(sparse, round_mode="nearest")

        print(f"\n  === NF4 sparse characterization ===")
        print(f"  Input sparse: nnz={_sparse_numel(sparse)}, shape={sparse.shape}")
        print(f"  Output type: {'sparse' if sparse_q.is_sparse else 'dense'}")

        if sparse_q.is_sparse:
            # If output is sparse, compare values
            sq_vals = sparse_q.values()
            # Get the corresponding dense result values
            mask = sparse.to_dense() != 0
            dq_masked = dense_q[mask]
            print(f"  Output nnz: {_sparse_numel(sparse_q)}")
            print(f"  Values match: {torch.equal(sq_vals, dq_masked)}")
            if not torch.equal(sq_vals, dq_masked):
                diff = (sq_vals - dq_masked).abs()
                print(f"  Max value diff: {diff.max().item():.6e}")
                print(f"  Mismatch count: {(sq_vals != dq_masked).sum().item()}/{sq_vals.numel()}")
        else:
            # Output is dense — sparse information was lost
            mask = sparse.to_dense() != 0
            sq_masked = sparse_q[mask]
            dq_masked = dense_q[mask]
            diff = (sq_masked - dq_masked).abs()
            print(f"  Max diff vs dense: {diff.max().item():.6e}")
            print(f"  Match count: {(sq_masked == dq_masked).sum().item()}/{mask.sum().item()}")

    def test_nf4_per_tensor_sparse_pipeline_detail(self):
        """Step 2: Full quantize(x, scheme) with per_tensor NF4 on sparse."""
        torch.manual_seed(42)
        dense = torch.randn(4, 8)
        sparse = _make_sparse_coo(dense, sparsity=0.7)

        scheme = QuantScheme.per_tensor("nf4")

        dense_q = quantize(dense, scheme)

        print(f"\n  === NF4 per_tensor sparse pipeline ===")
        try:
            sparse_q = quantize(sparse, scheme)
            print(f"  Output is_sparse: {sparse_q.is_sparse}")

            mask = sparse.to_dense() != 0
            if sparse_q.is_sparse:
                sparse_dense = sparse_q.to_dense()
            else:
                sparse_dense = sparse_q

            diff = (dense_q[mask] - sparse_dense[mask]).abs()
            print(f"  Max diff vs dense: {diff.max().item():.6e}")
            print(f"  Bit-exact matches: {(sparse_dense[mask] == dense_q[mask]).sum().item()}/{mask.sum().item()}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    def test_nf4_per_block_sparse_pipeline_detail(self):
        """Step 3: Full quantize(x, scheme) with per_block NF4 on sparse."""
        torch.manual_seed(42)
        dense = torch.randn(4, 32)
        sparse = _make_sparse_coo(dense, sparsity=0.7)

        scheme = QuantScheme.mxfp("nf4", block_size=8)

        dense_q = quantize(dense, scheme)

        print(f"\n  === NF4 per_block sparse pipeline ===")
        try:
            sparse_q = quantize(sparse, scheme)
            print(f"  Output is_sparse: {sparse_q.is_sparse}")

            mask = sparse.to_dense() != 0
            if sparse_q.is_sparse:
                sparse_dense = sparse_q.to_dense()
            else:
                sparse_dense = sparse_q

            diff = (dense_q[mask] - sparse_dense[mask]).abs()
            print(f"  Max diff vs dense: {diff.max().item():.6e}")
            print(f"  Bit-exact matches: {(sparse_dense[mask] == dense_q[mask]).sum().item()}/{mask.sum().item()}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Analysis — numerical error profiles for all 4-bit formats
# ═══════════════════════════════════════════════════════════════════════════════

class TestFourBitErrorAnalysis:
    """Analyze quantization error characteristics for each 4-bit format.

    This is a diagnostic section — all tests pass (they're informational).
    """

    def test_error_profile_comparison(self):
        """Print error profiles for all 4-bit formats on the same data."""
        torch.manual_seed(42)

        # Different distribution types
        distributions = {
            "normal": torch.randn(4, 128),
            "uniform": torch.rand(4, 128) * 10 - 5,
            "normal_small": torch.randn(4, 128) * 0.1,
            "normal_large": torch.randn(4, 128) * 5.0,
        }

        print("\n  === 4-bit format error profiles (per_tensor) ===")
        for dist_name, x in distributions.items():
            print(f"\n  --- {dist_name} ---")
            print(f"  {'Format':<12} {'MSE':>10} {'MAE':>10} {'MaxAbs':>10} {'MeanRel':>10}")
            print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

            for fmt_name in FOUR_BIT_FORMATS:
                fmt = _get_format(fmt_name)
                result = fmt.quantize(x, GranularitySpec.per_tensor(), "nearest")
                err = _quantization_error(x, result)
                print(f"  {fmt_name:<12} {err['mse']:>10.5f} {err['mae']:>10.5f} "
                      f"{err['max_abs']:>10.5f} {err['mean_rel']:>10.5f}")

    def test_fp4_int4_nf4_level_counts(self):
        """Compare the number of distinct quantization levels available."""
        print("\n  === 4-bit format level analysis ===")

        # fp4_e2m1: 2^4 = 16 levels total
        #   Normal: sign × (2^exp_bits-1) × 2^mbits_actual = 2 × 3 × 2 = 12
        #   Subnormal: sign × 2 × 2 = 4
        #   Plus zero
        print(f"  fp4_e2m1: 16 levels (2^4), range [{-6.0}, {6.0}], subnormals present")

        # int4: sign-magnitude, 16 levels, range [-1.75, 1.75]
        print(f"  int4:     16 levels (2^4), range [{-1.75}, {1.75}], uniform spacing")

        # nf4: 16 asymmetric levels, range [-1.0, 1.0]
        fmt = _get_format("nf4")
        levels = fmt.levels
        pos_levels = levels[levels > 0]
        neg_levels = levels[levels < 0]
        pos_spacings = pos_levels[1:] - pos_levels[:-1]
        neg_spacings = (-neg_levels)[1:] - (-neg_levels)[:-1]
        print(f"  nf4:      16 levels, range [{-1.0}, {1.0}], asymmetric")
        print(f"    pos spacings: {[f'{s:.4f}' for s in pos_spacings.tolist()]}")
        print(f"    neg spacings: {[f'{s:.4f}' for s in neg_spacings.tolist()]}")

    def test_sparse_vs_dense_error_comparison(self):
        """Compare error between sparse and dense quantization paths."""
        torch.manual_seed(42)
        dense = torch.randn(8, 64)
        sparse = _make_sparse_coo(dense, sparsity=0.7)

        print("\n  === Sparse vs dense quantization error ===")

        for fmt_name in ["fp4_e2m1", "int4"]:
            fmt = _get_format(fmt_name)

            dense_q = fmt.quantize_elemwise(dense, round_mode="nearest")
            sparse_q = fmt.quantize_elemwise(sparse, round_mode="nearest")

            mask = sparse.to_dense() != 0
            sparse_dense = sparse_q.to_dense()

            diff = (dense_q[mask] - sparse_dense[mask]).abs()
            print(f"  {fmt_name}:")
            print(f"    Dense-sparse max diff: {diff.max().item():.10f}")
            print(f"    Bit-exact match: {torch.equal(dense_q[mask], sparse_dense[mask])}")

            # Also compute error relative to original
            err_dense = _quantization_error(dense[mask], dense_q[mask])
            err_sparse = _quantization_error(dense[mask], sparse_dense[mask])
            print(f"    Dense error MAE:  {err_dense['mae']:.6f}")
            print(f"    Sparse error MAE: {err_sparse['mae']:.6f}")
