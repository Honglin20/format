"""
MxSpecs: quantization configuration system.

Faithful migration from mx/specs.py. MxSpecs is a 39-key dict
controlling all quantization behavior (formats, rounding, block sizes, etc.).
"""
import os
import json
import traceback
import collections


_ASSERT_MODE = os.environ.get('MX_ASSERT', 'False')


class MxSpecs(collections.UserDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        defaults = {
            "scale_bits": 0,

            "w_elem_format": None,
            "a_elem_format": None,
            "w_elem_format_bp": None,
            "a_elem_format_bp": None,
            "a_elem_format_bp_ex": None,
            "a_elem_format_bp_os": None,
            "mx_flush_fp32_subnorms": False,

            "shared_exp_method": "max",
            "block_size": 0,

            "bfloat": 0,
            "fp": 0,

            "bfloat_subnorms": True,

            "quantize_backprop": True,

            "round": "nearest",
            "round_m": "nearest",
            "round_weight": "nearest",
            "round_output": "nearest",
            "round_grad_weight": "nearest",
            "round_grad_input": "nearest",
            "round_mx_output": "nearest",
            "round_mx_input_grad_input": "nearest",
            "round_mx_weight_grad_input": "nearest",
            "round_mx_grad_output_grad_input": "nearest",
            "round_mx_input_grad_weight": "nearest",
            "round_mx_grad_output_grad_weight": "nearest",

            "softmax_exp2": False,
            "vec_use_exp2": False,
            "vec_use_recip": False,

            "custom_cuda": False,
        }

        self.help_strings = {
            "scale_bits": "Bits (sign + magnitude) to use for shared exponent/scale",
            "w_elem_format": "Weight MX elem format",
            "a_elem_format": "Activation MX elem format. See w_elem_format",
            "w_elem_format_bp": "Backpass weight MX elem format. See w_elem_format",
            "a_elem_format_bp": "Backpass stashed activation MX elem format. See w_elem_format",
            "a_elem_format_bp_ex": "Backpass act (grad) MX elem format. See w_elem_format",
            "a_elem_format_bp_os": "Backpass act (grad) MX elem format. See w_elem_format",
            "mx_flush_fp32_subnorms": "MX quantization flushes blocks with "
                                      "subnormal shared scale to zero",
            "shared_exp_method": "Shared exponent calculation method. Options: max, none",
            "block_size": "mx shared exponent block size",
            "bfloat": "BfloatX format (8exp + sign + mantissa). Only one of bfloat or fp can be used",
            "fp": "fpX format (5exp + sign + mantissa). Only one of bfloat or fp can be used",
            "bfloat_subnorms": "Bfloat/FP supports subnorms",
            "quantize_backprop": "Enable mx/bfloat quantization on backward pass",
            "round": "Global rounding mode. Choices: nearest, floor",
            "round_m": "ADAM optimizer m and v rounding mode",
            "round_weight": "Weight bfloat rounding mode (W in WAGE)",
            "round_output": "Activation bfloat rounding mode (A in WAGE)",
            "round_grad_weight": "Weight update rounding mode (G in WAGE)",
            "round_grad_input": "Error gradient rounding mode (E in WAGE)",
            "round_mx_output": "Forward pass mx rounding mode",
            "round_mx_input_grad_input": "",
            "round_mx_weight_grad_input": "",
            "round_mx_grad_output_grad_input": "",
            "round_mx_input_grad_weight": "",
            "round_mx_grad_output_grad_weight": "",
            "softmax_exp2": "Softmax uses 2^x instead of e^x",
            "vec_use_exp2": "Use 2^x to compute e^x",
            "vec_use_recip": "Use 1/x to compute division",
            "custom_cuda": "Enable custom CUDA kernels for quantization",
        }

        for k in defaults:
            if k not in self.data.keys():
                self.data[k] = defaults[k]

        for k in self.data.keys():
            assert k in self.help_strings.keys()

    def safe_json(self, indent=None):
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        return json.dumps(self.data, indent=indent, default=default)

    def __str__(self):
        return self.safe_json(indent=4)


def get_default_mx_specs():
    return MxSpecs()


def get_backwards_mx_specs(specs):
    bspecs = specs.copy()

    if bspecs["quantize_backprop"] == False:
        bspecs["w_elem_format"] = None
        bspecs["a_elem_format"] = None
        bspecs["w_elem_format_bp"] = None
        bspecs["a_elem_format_bp"] = None
        bspecs["a_elem_format_bp_os"] = None
        bspecs["a_elem_format_bp_ex"] = None
        bspecs["block_size"] = 0
        bspecs["bfloat"] = 0
        bspecs["fp"] = 0

    return bspecs


def apply_mx_specs(mx_specs, default_mx_specs=None):
    if not default_mx_specs:
        default_mx_specs = get_default_mx_specs()

    if not mx_specs:
        return default_mx_specs

    for k in mx_specs:
        if mx_specs[k] != None:
            if k not in default_mx_specs:
                raise KeyError(f"Unknown key '{k}' passed to mx specs")
            default_mx_specs[k] = mx_specs[k]

    return default_mx_specs


def finalize_mx_specs(specs, early_exit=True):
    # Early exit, works for 0 and None
    if (
        not specs.get("w_elem_format", 0)
        and not specs.get("a_elem_format", 0)
        and not specs.get("w_elem_format_bp", 0)
        and not specs.get("a_elem_format_bp", 0)
        and not specs.get("a_elem_format_bp_os", 0)
        and not specs.get("a_elem_format_bp_ex", 0)
        and not specs.get("bfloat", 0)
        and not specs.get("fp", 0)
        and early_exit
    ):
        return None

    if specs.get('custom_cuda'):
        import torch
        assert torch.cuda.is_available(), "'custom_cuda' is only supported on CUDA devices."

    def assign_if_none(f1, f2):
        if (f1 not in specs or specs[f1] is None) and f2 in specs:
            specs[f1] = specs[f2]

    assign_if_none("w_elem_format_bp", "w_elem_format")
    assign_if_none("a_elem_format_bp", "a_elem_format")
    assign_if_none("a_elem_format_bp_os", "a_elem_format")
    assign_if_none("a_elem_format_bp_ex", "a_elem_format")

    assign_if_none("round_m", "round")
    assign_if_none("round_output", "round")
    assign_if_none("round_grad_weight", "round")
    assign_if_none("round_grad_input", "round")
    assign_if_none("round_weight", "round")
    assign_if_none("round_mx_output", "round")

    assign_if_none("round_mx_input_grad_input", "round_grad_input")
    assign_if_none("round_mx_weight_grad_input", "round_grad_input")
    assign_if_none("round_mx_grad_output_grad_input", "round_grad_input")
    assign_if_none("round_mx_input_grad_weight", "round_grad_input")
    assign_if_none("round_mx_grad_output_grad_weight", "round_grad_input")
    assign_if_none("round_mx_grad_output_grad_input", "round_grad_input")

    specs = apply_mx_specs(specs, get_default_mx_specs())
    return specs


def mx_assert_test(mx_specs):
    if _ASSERT_MODE == "True" and mx_specs is None:
        stack = traceback.extract_stack()
        f1 = stack[-2]
        f2 = stack[-3]
        msg = (
            "MX assert test failed!\n"
            + f"mx_specs is None in function {f1.name}\n"
            + f"Called from {f2.filename}, line {f2.lineno}\n"
            + f"  {f2.line}"
        )
        raise ValueError(msg)
