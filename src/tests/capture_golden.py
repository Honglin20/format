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


def capture_mx_quantize():
    from mx.mx_ops import quantize_mx_op

    formats = ["fp8_e4m3", "fp6_e3m2", "fp4_e2m1", "int8"]
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
    capture_mx_quantize()
    print()
    print("=== Operators ===")
    capture_linear()
    capture_matmul()
    capture_layernorm()
    capture_relu()
    capture_gelu()
    capture_sigmoid()
    capture_softmax()
    capture_conv2d()
    capture_simd_add()
    print()
    print("Done! Golden files saved to:", GOLDEN_DIR)
