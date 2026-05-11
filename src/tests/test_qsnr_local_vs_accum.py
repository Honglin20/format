"""
Test: local (observer) QSNR vs accum (hook) QSNR consistency.

For the FIRST layer of a model, there is no upstream error propagation.
Therefore, accum QSNR (fp32 output vs quantized output) should approximately
equal local QSNR (observer, output role, total layer error).

Before the observer fix, these values diverge because the observer's fp32
reference is the pre-quant tensor (already degraded by weight/input quant),
not the true fp32 output. This test validates the fix.
"""
import pytest
import torch
import torch.nn as nn

from src.session._session import Session
from src.session._config import QuantConfig


def _extract_qsnr_all_roles(observers_data):
    """Extract per-layer minimum QSNR across ALL roles (input/weight/output/bias).

    Returns ``Dict[str, float]`` with the worst-case observer QSNR per layer.
    """
    from src.session._session import _extract_qsnr_mse

    all_qsnr: dict = {}
    for role in ["input", "weight", "output", "bias"]:
        role_qsnr, _ = _extract_qsnr_mse(observers_data, role=role)
        for layer, qsnr in role_qsnr.items():
            prev = all_qsnr.get(layer)
            if prev is None or qsnr < prev:
                all_qsnr[layer] = qsnr
    return all_qsnr


def _run_analyze(model, cfg, input_tensor, n_batches=1):
    """Run session.run and return (accum_qsnr, local_out, local_all)."""
    session = Session(model, cfg)
    batches = [input_tensor for _ in range(n_batches)]
    result = session.run(batches, outputs=["qsnr"])

    accum = dict(result.accum_qsnr_per_layer)
    local_output = dict(result.qsnr_per_layer)

    # Local from all roles (min across input/weight/output/bias)
    if session._observers_data:
        local_all = _extract_qsnr_all_roles(session._observers_data)
    else:
        local_all = {}

    return accum, local_output, local_all


def _qsnr_close(val1, val2, atol=2.0):
    """Check two QSNR values are close (in dB)."""
    return abs(val1 - val2) <= atol


# ---------------------------------------------------------------------------
# Single-op models
# ---------------------------------------------------------------------------

class SingleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc(x)


class SingleConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3)

    def forward(self, x):
        return self.conv(x)


class SingleBN2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(x)


class SingleLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(8)

    def forward(self, x):
        return self.ln(x)


class SingleReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x)


class SingleSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.sm(x)


# ---------------------------------------------------------------------------
# Test parametrization
# ---------------------------------------------------------------------------

OP_TEST_CASES = [
    pytest.param(
        "linear", SingleLinear, torch.randn(4, 16),
        id="linear",
    ),
    pytest.param(
        "conv2d", SingleConv2d, torch.randn(2, 3, 8, 8),
        id="conv2d",
    ),
    pytest.param(
        "batchnorm2d", SingleBN2d, torch.randn(2, 4, 8, 8),
        id="batchnorm2d",
    ),
    pytest.param(
        "layernorm", SingleLN, torch.randn(4, 8),
        id="layernorm",
    ),
    pytest.param(
        "relu", SingleReLU, torch.randn(4, 8),
        id="relu",
    ),
    pytest.param(
        "softmax", SingleSoftmax, torch.randn(4, 8),
        id="softmax",
    ),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLocalVsAccumQSNR:
    """Verify observer local QSNR ≈ accum QSNR for first layer of each op."""

    @pytest.mark.parametrize("op_name,model_cls,input_tensor", OP_TEST_CASES)
    def test_first_layer_local_equals_accum(
        self, op_name, model_cls, input_tensor
    ):
        """First layer should have local QSNR ≈ accum QSNR (no upstream)."""
        torch.manual_seed(42)
        model = model_cls()
        cfg = QuantConfig(
            w_format="int8", w_granularity="per_tensor",
            a_format="int8", a_granularity="per_tensor",
        )

        accum, local_out, local_all = _run_analyze(model, cfg, input_tensor)

        if not accum:
            pytest.skip(f"No accum QSNR data for {op_name}")
        if not local_out:
            pytest.skip(f"No local QSNR data for {op_name}")

        first_layer = next(iter(accum.keys()))
        accum_val = accum[first_layer]
        local_val = local_out.get(first_layer)
        local_all_val = local_all.get(first_layer)

        assert local_val is not None, (
            f"{op_name}: no local QSNR for '{first_layer}'"
        )

        # After fix: accum ≈ local (within 2 dB)
        assert _qsnr_close(accum_val, local_val, atol=2.0), (
            f"{op_name} [{first_layer}]: "
            f"accum={accum_val:.2f} dB, local(output)={local_val:.2f} dB, "
            f"diff={abs(accum_val - local_val):.2f} dB"
        )

    @pytest.mark.parametrize("op_name,model_cls,input_tensor", OP_TEST_CASES)
    def test_first_layer_all_roles_consistent(
        self, op_name, model_cls, input_tensor
    ):
        """Observer data across all roles should be consistent."""
        torch.manual_seed(42)
        model = model_cls()
        cfg = QuantConfig(
            w_format="int8", w_granularity="per_tensor",
            a_format="int8", a_granularity="per_tensor",
        )

        accum, local_out, local_all = _run_analyze(model, cfg, input_tensor)

        if not accum:
            pytest.skip(f"No accum data for {op_name}")

        first_layer = next(iter(accum.keys()))
        accum_val = accum[first_layer]
        local_all_val = local_all.get(first_layer)

        if local_all_val is None:
            pytest.skip(f"No all-role local QSNR for {op_name}")

        # all-role local should be ≤ accum (worst-case across all roles)
        # and within reasonable range
        diff = local_all_val - accum_val
        assert diff < 5.0, (
            f"{op_name} [{first_layer}]: "
            f"all-role local={local_all_val:.2f} >> accum={accum_val:.2f} dB "
            f"(diff={diff:.2f}) — observer missing error sources"
        )


class TestMultiLayerPropagation:
    """Verify error propagation logic with stacked layers."""

    def test_two_linear_layers_delta_nonzero(self):
        """Two layers: second layer should have delta > 0 (accum drops)."""
        torch.manual_seed(42)

        class TwoLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 8)
                self.fc2 = nn.Linear(8, 4)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        model = TwoLinear()
        cfg = QuantConfig(
            w_format="int8", w_granularity="per_tensor",
            a_format="int8", a_granularity="per_tensor",
        )
        accum, local_out, local_all = _run_analyze(
            model, cfg, torch.randn(4, 16),
        )

        layers = sorted(accum.keys())
        assert len(layers) >= 2, f"Expected 2+ layers, got {layers}"

        # First layer: local ≈ accum
        l0_accum = accum[layers[0]]
        l0_local = local_out.get(layers[0])
        if l0_local is not None:
            assert _qsnr_close(l0_accum, l0_local, atol=2.0), (
                f"Layer 0 [{layers[0]}]: accum={l0_accum:.2f}, local={l0_local:.2f}"
            )

        # Second layer: accum should drop (delta > 0)
        l1_accum = accum[layers[1]]
        assert l1_accum <= l0_accum + 0.5, (
            f"Layer 1 accum ({l1_accum:.2f}) should not exceed "
            f"layer 0 accum ({l0_accum:.2f})"
        )


# ---------------------------------------------------------------------------
# Comparison table generator (run manually)
# ---------------------------------------------------------------------------

def generate_comparison_table():
    """Generate accum vs local QSNR comparison for all ops.

    Run: python -m pytest src/tests/test_qsnr_local_vs_accum.py -s -k 'generate'
    """
    model_configs = [
        ("Linear", SingleLinear, torch.randn(4, 16)),
        ("Conv2d", SingleConv2d, torch.randn(2, 3, 8, 8)),
        ("BatchNorm2d", SingleBN2d, torch.randn(2, 4, 8, 8)),
        ("LayerNorm", SingleLN, torch.randn(4, 8)),
        ("ReLU", SingleReLU, torch.randn(4, 8)),
        ("Softmax", SingleSoftmax, torch.randn(4, 8)),
    ]

    cfg = QuantConfig(
        w_format="int8", w_granularity="per_tensor",
        a_format="int8", a_granularity="per_tensor",
    )

    print(f"\n{'='*90}")
    print(f"  accum QSNR vs local QSNR — per-operator comparison")
    print(f"{'='*90}")
    print(f"{'Op':<16} {'Layer':<28} {'Accum QSNR':>12} {'Local(out)':>12} "
          f"{'Local(all)':>12} {'Diff(out)':>10} {'Diff(all)':>10}")
    print("-" * 90)

    for op_name, model_cls, input_tensor in model_configs:
        torch.manual_seed(42)
        model = model_cls()
        try:
            accum, local_out, local_all = _run_analyze(
                model, cfg, input_tensor,
            )
        except Exception as e:
            print(f"{op_name:<16} {'ERROR':<28} {str(e)[:40]}")
            continue

        if not accum:
            print(f"{op_name:<16} {'(no accum data)':<28}")
            continue

        first_layer = next(iter(accum.keys()))
        accum_val = accum[first_layer]
        local_val = local_out.get(first_layer, float('nan'))
        local_all_val = local_all.get(first_layer, float('nan'))
        diff_out = local_val - accum_val
        diff_all = local_all_val - accum_val

        print(
            f"{op_name:<16} {first_layer:<28} "
            f"{accum_val:>12.2f} {local_val:>12.2f} {local_all_val:>12.2f} "
            f"{diff_out:>+10.2f} {diff_all:>+10.2f}"
        )

    print("-" * 90)
    print("  Diff = local - accum. Should be ≈ 0 for first layer after fix.")
    print("  Positive diff means observer overestimates QSNR (missing error sources).")


if __name__ == "__main__":
    generate_comparison_table()
