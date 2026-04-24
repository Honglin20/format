"""
Golden reference capture script.

Runs every operator with representative mx_specs configurations
and saves (input, mx_specs, output, grad_input, grad_weight) as .pt files.
Used for regression testing after refactoring.
"""
import os
import sys
import torch
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mx.specs import finalize_mx_specs

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
os.makedirs(GOLDEN_DIR, exist_ok=True)

torch.manual_seed(42)
DEVICE = "cpu"


def save_golden(name, **tensors):
    path = os.path.join(GOLDEN_DIR, f"{name}.pt")
    torch.save(tensors, path)
    print(f"  Saved {name}")


# ---------------------------------------------------------------------------
# MX specs configs to capture
# ---------------------------------------------------------------------------

CONFIGS = {
    "mxfp8_e5m2": finalize_mx_specs({
        "w_elem_format": "fp8_e5m2",
        "a_elem_format": "fp8_e5m2",
        "block_size": 32,
        "bfloat": 16,
    }),
    "mxfp8_e4m3": finalize_mx_specs({
        "w_elem_format": "fp8_e4m3",
        "a_elem_format": "fp8_e4m3",
        "block_size": 32,
        "bfloat": 16,
    }),
    "mxfp6_e3m2": finalize_mx_specs({
        "w_elem_format": "fp6_e3m2",
        "a_elem_format": "fp6_e3m2",
        "block_size": 32,
        "bfloat": 16,
    }),
    "mxfp6_e2m3": finalize_mx_specs({
        "w_elem_format": "fp6_e2m3",
        "a_elem_format": "fp6_e2m3",
        "block_size": 32,
        "bfloat": 16,
    }),
    "mxfp4_e2m1": finalize_mx_specs({
        "w_elem_format": "fp4_e2m1",
        "a_elem_format": "fp4_e2m1",
        "block_size": 32,
        "bfloat": 16,
    }),
    "mxint8": finalize_mx_specs({
        "w_elem_format": "int8",
        "a_elem_format": "int8",
        "block_size": 32,
        "bfloat": 16,
    }),
    "mxint4": finalize_mx_specs({
        "w_elem_format": "int4",
        "a_elem_format": "int4",
        "block_size": 32,
        "bfloat": 16,
    }),
    "mxint2": finalize_mx_specs({
        "w_elem_format": "int2",
        "a_elem_format": "int2",
        "block_size": 32,
        "bfloat": 16,
    }),
    "bfloat16": finalize_mx_specs({"bfloat": 16}),
}


# ---------------------------------------------------------------------------
# Capture functions per operator
# ---------------------------------------------------------------------------

def capture_linear():
    from mx.linear import Linear

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        layer = Linear(64, 32, mx_specs=mx_specs).to(DEVICE)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        save_golden(
            f"linear_{cfg_name}",
            input=x.detach(),
            weight=layer.weight.detach(),
            bias=layer.bias.detach() if layer.bias is not None else None,
            output=out.detach(),
            grad_input=x.grad.detach(),
            grad_weight=layer.weight.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_matmul():
    from mx.matmul import matmul

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        a = torch.randn(2, 8, 32, device=DEVICE, requires_grad=True)
        b = torch.randn(2, 32, 16, device=DEVICE, requires_grad=True)
        out = matmul(a, b, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"matmul_{cfg_name}",
            input_a=a.detach(),
            input_b=b.detach(),
            output=out.detach(),
            grad_a=a.grad.detach(),
            grad_b=b.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_bmm():
    from mx.bmm import bmm

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        a = torch.randn(2, 8, 32, device=DEVICE, requires_grad=True)
        b = torch.randn(2, 32, 16, device=DEVICE, requires_grad=True)
        out = bmm(a, b, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"bmm_{cfg_name}",
            input_a=a.detach(),
            input_b=b.detach(),
            output=out.detach(),
            grad_a=a.grad.detach(),
            grad_b=b.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_layernorm():
    from mx.layernorm import LayerNorm

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        layer = LayerNorm(64, mx_specs=mx_specs).to(DEVICE)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        save_golden(
            f"layernorm_{cfg_name}",
            input=x.detach(),
            weight=layer.weight.detach(),
            bias=layer.bias.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            grad_weight=layer.weight.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_batchnorm():
    from mx.batchnorm import BatchNorm1d, BatchNorm2d

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        layer = BatchNorm1d(16, mx_specs=mx_specs).to(DEVICE)
        x = torch.randn(2, 16, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        save_golden(
            f"batchnorm1d_{cfg_name}",
            input=x.detach(),
            weight=layer.weight.detach(),
            bias=layer.bias.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            grad_weight=layer.weight.grad.detach(),
            mx_specs=dict(mx_specs),
        )

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        layer = BatchNorm2d(16, mx_specs=mx_specs).to(DEVICE)
        x = torch.randn(2, 16, 4, 4, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        save_golden(
            f"batchnorm2d_{cfg_name}",
            input=x.detach(),
            weight=layer.weight.detach(),
            bias=layer.bias.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            grad_weight=layer.weight.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_groupnorm():
    from mx.groupnorm import GroupNorm

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        layer = GroupNorm(4, 16, mx_specs=mx_specs).to(DEVICE)
        x = torch.randn(2, 16, 8, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        save_golden(
            f"groupnorm_{cfg_name}",
            input=x.detach(),
            weight=layer.weight.detach(),
            bias=layer.bias.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            grad_weight=layer.weight.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_relu():
    from mx.activations import relu

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = relu(x, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"relu_{cfg_name}",
            input=x.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_relu6():
    from mx.activations import relu6

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = relu6(x, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"relu6_{cfg_name}",
            input=x.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_leaky_relu():
    from mx.activations import leaky_relu

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = leaky_relu(x, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"leaky_relu_{cfg_name}",
            input=x.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_gelu():
    from mx.activations import gelu

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = gelu(x, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"gelu_{cfg_name}",
            input=x.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_silu():
    from mx.activations import silu

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = silu(x, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"silu_{cfg_name}",
            input=x.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_sigmoid():
    from mx.activations import sigmoid

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = sigmoid(x, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"sigmoid_{cfg_name}",
            input=x.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_tanh():
    from mx.activations import tanh

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = tanh(x, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"tanh_{cfg_name}",
            input=x.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_softmax():
    from mx.softmax import softmax

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = softmax(x, dim=-1, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"softmax_{cfg_name}",
            input=x.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_conv2d():
    from mx.convolution import Conv2d

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        layer = Conv2d(3, 16, 3, padding=1, mx_specs=mx_specs).to(DEVICE)
        x = torch.randn(1, 3, 8, 8, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        save_golden(
            f"conv2d_{cfg_name}",
            input=x.detach(),
            weight=layer.weight.detach(),
            bias=layer.bias.detach() if layer.bias is not None else None,
            output=out.detach(),
            grad_input=x.grad.detach(),
            grad_weight=layer.weight.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_convtranspose2d():
    from mx.transpose_convolution import ConvTranspose2d

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        layer = ConvTranspose2d(16, 3, 3, padding=1, mx_specs=mx_specs).to(DEVICE)
        x = torch.randn(1, 16, 8, 8, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        save_golden(
            f"convtranspose2d_{cfg_name}",
            input=x.detach(),
            weight=layer.weight.detach(),
            bias=layer.bias.detach() if layer.bias is not None else None,
            output=out.detach(),
            grad_input=x.grad.detach(),
            grad_weight=layer.weight.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_adaptive_avg_pool2d():
    from mx.adaptive_avg_pooling import adaptive_avg_pool2d

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(1, 16, 8, 8, device=DEVICE, requires_grad=True)
        out = adaptive_avg_pool2d(x, output_size=4, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"adaptive_avg_pool2d_{cfg_name}",
            input=x.detach(),
            output=out.detach(),
            grad_input=x.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_lstm():
    from mx.rnn import LSTM

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        layer = LSTM(32, 16, num_layers=1, mx_specs=mx_specs).to(DEVICE)
        x = torch.randn(4, 2, 32, device=DEVICE, requires_grad=True)
        output, (h_n, c_n) = layer(x)
        output.sum().backward()

        save_golden(
            f"lstm_{cfg_name}",
            input=x.detach(),
            output=output.detach(),
            h_n=h_n.detach(),
            c_n=c_n.detach(),
            grad_input=x.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_simd_add():
    from mx.simd_ops import simd_add

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        a = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        b = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = simd_add(a, b, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"simd_add_{cfg_name}",
            input_a=a.detach(),
            input_b=b.detach(),
            output=out.detach(),
            grad_a=a.grad.detach(),
            grad_b=b.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_simd_sub():
    from mx.simd_ops import simd_sub

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        a = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        b = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = simd_sub(a, b, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"simd_sub_{cfg_name}",
            input_a=a.detach(),
            input_b=b.detach(),
            output=out.detach(),
            grad_a=a.grad.detach(),
            grad_b=b.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_simd_mul():
    from mx.simd_ops import simd_mul

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        a = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        b = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        out = simd_mul(a, b, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"simd_mul_{cfg_name}",
            input_a=a.detach(),
            input_b=b.detach(),
            output=out.detach(),
            grad_a=a.grad.detach(),
            grad_b=b.grad.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_simd_div():
    from mx.simd_ops import simd_div

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        a = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        # Ensure b is positive to avoid division by zero, but keep in autograd graph
        b_raw = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        b = torch.clamp(b_raw, min=0.1)
        out = simd_div(a, b, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"simd_div_{cfg_name}",
            input_a=a.detach(),
            input_b=b_raw.detach(),
            output=out.detach(),
            grad_a=a.grad.detach(),
            grad_b=b_raw.grad.detach() if b_raw.grad is not None else None,
            mx_specs=dict(mx_specs),
        )


def capture_simd_unary_ops():
    """Capture simd_split, simd_sqrt, simd_square, simd_exp, simd_log."""
    from mx.simd_ops import simd_split, simd_sqrt, simd_square, simd_exp, simd_log

    unary_ops = [
        ("simd_square", simd_square),
        ("simd_exp", simd_exp),
    ]

    for op_name, op_fn in unary_ops:
        for cfg_name, mx_specs in CONFIGS.items():
            torch.manual_seed(42)
            x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
            out = op_fn(x, mx_specs=mx_specs)
            out.sum().backward()

            save_golden(
                f"{op_name}_{cfg_name}",
                input=x.detach(),
                output=out.detach(),
                grad_input=x.grad.detach(),
                mx_specs=dict(mx_specs),
            )

    # simd_sqrt requires non-negative input — create positive tensor
    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x_raw = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        x_pos = torch.clamp(x_raw, min=0.0)
        out = simd_sqrt(x_pos, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"simd_sqrt_{cfg_name}",
            input=x_raw.detach(),
            output=out.detach(),
            grad_input=x_raw.grad.detach() if x_raw.grad is not None else None,
            mx_specs=dict(mx_specs),
        )

    # simd_log requires positive input
    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x_raw = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        x_pos = torch.clamp(x_raw, min=0.1)
        out = simd_log(x_pos, mx_specs=mx_specs)
        out.sum().backward()

        save_golden(
            f"simd_log_{cfg_name}",
            input=x_raw.detach(),
            output=out.detach(),
            grad_input=x_raw.grad.detach() if x_raw.grad is not None else None,
            mx_specs=dict(mx_specs),
        )

    # simd_split returns tuple
    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)
        x1, x2 = simd_split(x, mx_specs=mx_specs)

        save_golden(
            f"simd_split_{cfg_name}",
            input=x.detach(),
            output_x1=x1.detach(),
            output_x2=x2.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_simd_reduce_ops():
    """Capture simd_reduce_sum, simd_reduce_mean, simd_norm."""
    from mx.simd_ops import simd_reduce_sum, simd_reduce_mean, simd_norm

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(2, 8, 64, device=DEVICE, requires_grad=True)

        out_sum = simd_reduce_sum(x, dim=-1, mx_specs=mx_specs)
        out_mean = simd_reduce_mean(x, dim=-1, mx_specs=mx_specs)
        out_norm = simd_norm(x, mx_specs=mx_specs)

        save_golden(
            f"simd_reduce_sum_{cfg_name}",
            input=x.detach(),
            output=out_sum.detach(),
            mx_specs=dict(mx_specs),
        )
        save_golden(
            f"simd_reduce_mean_{cfg_name}",
            input=x.detach(),
            output=out_mean.detach(),
            mx_specs=dict(mx_specs),
        )
        save_golden(
            f"simd_norm_{cfg_name}",
            input=x.detach(),
            output=out_norm.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_elemwise_quantize():
    from mx.elemwise_ops import quantize_elemwise_op

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(4, 32, device=DEVICE)
        out = quantize_elemwise_op(x, mx_specs=mx_specs)

        save_golden(
            f"elemwise_quantize_{cfg_name}",
            input=x.detach(),
            output=out.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_quantize_bfloat():
    from mx.quantize import quantize_bfloat

    for cfg_name, mx_specs in CONFIGS.items():
        torch.manual_seed(42)
        x = torch.randn(4, 32, device=DEVICE)
        out = quantize_bfloat(x, mx_specs=mx_specs)

        save_golden(
            f"quantize_bfloat_{cfg_name}",
            input=x.detach(),
            output=out.detach(),
            mx_specs=dict(mx_specs),
        )


def capture_mx_quantize():
    from mx.mx_ops import quantize_mx_op

    formats = ["fp8_e5m2", "fp8_e4m3", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1", "int8", "int4", "int2"]
    block_sizes = [32]

    for fmt in formats:
        for bs in block_sizes:
            mx_specs = finalize_mx_specs({
                "w_elem_format": fmt,
                "a_elem_format": fmt,
                "block_size": bs,
                "bfloat": 16,
            })
            torch.manual_seed(42)
            x = torch.randn(4, 64, device=DEVICE)
            out = quantize_mx_op(x, mx_specs, elem_format=fmt, axes=[-1])

            save_golden(
                f"mx_quantize_{fmt}_bs{bs}",
                input=x.detach(),
                output=out.detach(),
                mx_specs=dict(mx_specs),
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Capturing golden references...")
    print()
    print("=== Core quantization ===")
    capture_elemwise_quantize()
    capture_quantize_bfloat()
    capture_mx_quantize()
    print()
    print("=== Linear operators ===")
    capture_linear()
    capture_matmul()
    capture_bmm()
    print()
    print("=== Normalization ===")
    capture_layernorm()
    capture_batchnorm()
    capture_groupnorm()
    print()
    print("=== Activations ===")
    capture_relu()
    capture_relu6()
    capture_leaky_relu()
    capture_gelu()
    capture_silu()
    capture_sigmoid()
    capture_tanh()
    print()
    print("=== Softmax ===")
    capture_softmax()
    print()
    print("=== Convolutions ===")
    capture_conv2d()
    capture_convtranspose2d()
    print()
    print("=== Pooling ===")
    capture_adaptive_avg_pool2d()
    print()
    print("=== RNN ===")
    capture_lstm()
    print()
    print("=== SIMD ops ===")
    capture_simd_add()
    capture_simd_sub()
    capture_simd_mul()
    capture_simd_div()
    capture_simd_unary_ops()
    capture_simd_reduce_ops()
    print()
    print("Done! Golden files saved to:", GOLDEN_DIR)
