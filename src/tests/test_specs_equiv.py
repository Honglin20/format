"""
MxSpecs equivalence tests: verify new src/specs/ produces identical
output to old mx/specs/ for every function and configuration.
"""
import pytest
from mx.specs import (
    MxSpecs as OldMxSpecs,
    get_default_mx_specs as old_get_default,
    apply_mx_specs as old_apply,
    finalize_mx_specs as old_finalize,
    get_backwards_mx_specs as old_get_bw,
)

ALL_SPEC_KEYS = [
    "scale_bits", "w_elem_format", "a_elem_format", "w_elem_format_bp",
    "a_elem_format_bp", "a_elem_format_bp_ex", "a_elem_format_bp_os",
    "mx_flush_fp32_subnorms", "shared_exp_method", "block_size",
    "bfloat", "fp", "bfloat_subnorms", "quantize_backprop",
    "round", "round_m", "round_weight", "round_output",
    "round_grad_weight", "round_grad_input", "round_mx_output",
    "round_mx_input_grad_input", "round_mx_weight_grad_input",
    "round_mx_grad_output_grad_input", "round_mx_input_grad_weight",
    "round_mx_grad_output_grad_weight",
    "softmax_exp2", "vec_use_exp2", "vec_use_recip", "custom_cuda",
]


# ---------------------------------------------------------------------------
# 1. MxSpecs defaults
# ---------------------------------------------------------------------------

def test_mx_specs_defaults():
    from src.specs.specs import MxSpecs, get_default_mx_specs
    old = old_get_default()
    new = get_default_mx_specs()
    for key in ALL_SPEC_KEYS:
        assert key in new, f"Key {key!r} missing from new MxSpecs"
        assert new[key] == old[key], f"Default mismatch for {key!r}: {new[key]} != {old[key]}"
    assert len(new) == len(old), f"Key count mismatch: {len(new)} != {len(old)}"


# ---------------------------------------------------------------------------
# 2. apply_mx_specs
# ---------------------------------------------------------------------------

def test_apply_mx_specs():
    from src.specs.specs import apply_mx_specs

    test_cases = [
        {"bfloat": 16},
        {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3", "block_size": 32, "bfloat": 16},
        {"bfloat": 16, "round": "floor"},
        {"fp": 8, "quantize_backprop": False},
    ]
    for user_specs in test_cases:
        old_result = old_apply(user_specs.copy())
        new_result = apply_mx_specs(user_specs.copy())
        for key in ALL_SPEC_KEYS:
            assert new_result[key] == old_result[key], \
                f"apply_mx_specs mismatch for {key!r} with input {user_specs}: {new_result[key]} != {old_result[key]}"


# ---------------------------------------------------------------------------
# 3. finalize_mx_specs
# ---------------------------------------------------------------------------

def test_finalize_mx_specs():
    from src.specs.specs import finalize_mx_specs

    test_cases = [
        {"bfloat": 16},
        {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3", "block_size": 32, "bfloat": 16},
        {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp4_e2m1", "block_size": 32, "bfloat": 16},
        {"w_elem_format": "int8", "a_elem_format": "int8", "block_size": 32, "bfloat": 16},
        {"bfloat": 16, "quantize_backprop": False},
        {"bfloat": 16, "round": "floor"},
        {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3", "block_size": 32, "bfloat": 16,
         "w_elem_format_bp": "fp4_e2m1"},
        {},  # Should return None (no quantization)
    ]
    for user_specs in test_cases:
        old_result = old_finalize(user_specs.copy())
        new_result = finalize_mx_specs(user_specs.copy())
        if old_result is None:
            assert new_result is None, f"Expected None for {user_specs}"
        else:
            for key in ALL_SPEC_KEYS:
                assert new_result[key] == old_result[key], \
                    f"finalize mismatch for {key!r} with {user_specs}: {new_result[key]} != {old_result[key]}"


# ---------------------------------------------------------------------------
# 4. get_backwards_mx_specs
# ---------------------------------------------------------------------------

def test_get_backwards_mx_specs():
    from src.specs.specs import get_backwards_mx_specs, finalize_mx_specs

    specs_with_bp = finalize_mx_specs({"bfloat": 16, "quantize_backprop": True})
    specs_no_bp = finalize_mx_specs({"bfloat": 16, "quantize_backprop": False})

    old_bw_bp = old_get_bw(old_finalize({"bfloat": 16, "quantize_backprop": True}))
    old_bw_no = old_get_bw(old_finalize({"bfloat": 16, "quantize_backprop": False}))
    new_bw_bp = get_backwards_mx_specs(specs_with_bp)
    new_bw_no = get_backwards_mx_specs(specs_no_bp)

    for key in ALL_SPEC_KEYS:
        assert new_bw_bp[key] == old_bw_bp[key], f"backwards_bp mismatch: {key}"
        assert new_bw_no[key] == old_bw_no[key], f"backwards_no_bp mismatch: {key}"


# ---------------------------------------------------------------------------
# 5. mx_assert_test
# ---------------------------------------------------------------------------

def test_mx_assert_test():
    from src.specs.specs import mx_assert_test
    # With valid specs, should not raise
    specs = old_finalize({"bfloat": 16})
    mx_assert_test(specs)  # no exception


# ---------------------------------------------------------------------------
# 6. add_mx_args / get_mx_specs (argparse bridge)
# ---------------------------------------------------------------------------

def test_add_mx_args_and_get_mx_specs():
    import argparse
    from mx.specs import add_mx_args as old_add, get_mx_specs as old_get
    from src.specs.specs import add_mx_args, get_mx_specs

    # Build old and new parsers
    old_parser = argparse.ArgumentParser()
    old_parser = old_add(old_parser)
    new_parser = argparse.ArgumentParser()
    new_parser = add_mx_args(new_parser)

    # Test with no args (should return None — no quantization)
    old_args = old_parser.parse_args([])
    new_args = new_parser.parse_args([])
    old_result = old_get(old_args)
    new_result = get_mx_specs(new_args)
    assert old_result is None and new_result is None, "No-args should return None"

    # Test with bfloat=16
    old_parser2 = argparse.ArgumentParser()
    old_parser2 = old_add(old_parser2)
    new_parser2 = argparse.ArgumentParser()
    new_parser2 = add_mx_args(new_parser2)

    old_args2 = old_parser2.parse_args([])
    old_args2.bfloat = 16
    new_args2 = new_parser2.parse_args([])
    new_args2.bfloat = 16

    old_result2 = old_get(old_args2)
    new_result2 = get_mx_specs(new_args2)

    for key in ALL_SPEC_KEYS:
        assert new_result2[key] == old_result2[key], \
            f"argparse bridge mismatch for {key!r}: {new_result2[key]} != {old_result2[key]}"
