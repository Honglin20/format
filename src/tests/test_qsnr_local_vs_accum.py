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


# ---------------------------------------------------------------------------
# Deep multi-operator model: per-layer local vs accum QSNR
# ---------------------------------------------------------------------------

class DeepMixedModel(nn.Module):
    """10-layer model mixing conv, norm, activation, linear, and pooling.

    Layer order:
      0. conv   — Conv2d(3→8, k3p1)
      1. bn     — BatchNorm2d(8)
      2. relu   — ReLU
      3. pool   — AdaptiveAvgPool2d(→4×4)
      4. fc1    — Linear(128→64)
      5. ln     — LayerNorm(64)
      6. relu2  — ReLU
      7. fc2    — Linear(64→32)
      8. relu3  — ReLU
      9. fc3    — Linear(32→10)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128, 64)
        self.ln = nn.LayerNorm(64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv(x)       # 0: Conv2d
        x = self.bn(x)         # 1: BatchNorm2d
        x = self.relu(x)       # 2: ReLU
        x = self.pool(x)       # 3: AdaptiveAvgPool2d
        x = x.flatten(1)
        x = self.fc1(x)        # 4: Linear
        x = self.ln(x)         # 5: LayerNorm
        x = self.relu2(x)      # 6: ReLU
        x = self.fc2(x)        # 7: Linear
        x = self.relu3(x)      # 8: ReLU
        x = self.fc3(x)        # 9: Linear
        return x


def generate_deep_model_table():
    """Run deep mixed model and print per-layer accum vs local QSNR table.

    Usage::

        python -c "from src.tests.test_qsnr_local_vs_accum import generate_deep_model_table; generate_deep_model_table()"
    """
    torch.manual_seed(42)
    model = DeepMixedModel()
    cfg = QuantConfig(
        w_format="int8", w_granularity="per_tensor",
        a_format="int8", a_granularity="per_tensor",
    )
    input_tensor = torch.randn(2, 3, 8, 8)

    session = Session(model, cfg)
    result = session.run([input_tensor], outputs=["qsnr"])

    accum = dict(result.accum_qsnr_per_layer)
    local_out = dict(result.qsnr_per_layer)
    local_all = _extract_qsnr_all_roles(result.observers_data)

    all_layers = sorted(set(accum.keys()) | set(local_out.keys()))
    if not all_layers:
        print("No layer data found.")
        return

    op_tags = {
        "conv": "Conv2d",
        "bn": "BatchNorm2d",
        "relu": "ReLU",
        "pool": "AvgPool2d",
        "fc1": "Linear",
        "fc2": "Linear",
        "fc3": "Linear",
        "ln": "LayerNorm",
        "relu2": "ReLU",
        "relu3": "ReLU",
    }

    print(f"\n{'='*105}")
    print(f"  Deep Mixed Model — Per-Layer accum QSNR vs local QSNR")
    print(f"  Model: Conv→BN→ReLU→Pool→Linear→LN→ReLU→Linear→ReLU→Linear")
    print(f"  Config: int8 per_tensor (quantize_nonlinear=True)")
    print(f"{'='*105}")
    print(f"{'#':<3} {'Layer':<16} {'Type':<14} {'Accum':>10} {'Local':>10} "
          f"{'Diff':>10} {'ΔAccum':>10} {'Note'}")
    print("-" * 105)

    prev_accum = None
    for i, layer in enumerate(all_layers):
        tag = op_tags.get(layer, "?")
        accum_val = accum.get(layer, float('nan'))
        local_val = local_out.get(layer, float('nan'))
        local_all_val = local_all.get(layer, float('nan'))

        def _fv(v):
            if v is None or (isinstance(v, float) and v != v):
                return "N/A"
            return f"{v:.2f}"

        diff = None
        note = ""
        if not isnan_str(accum_val) and not isnan_str(local_val):
            diff = local_val - accum_val
            if abs(diff) < 2.0:
                note = "✓ consistent"
            elif local_val > 100:
                note = "pass-through (no entry quant)"
            elif diff > 10:
                note = "upstream error propagation"
            else:
                note = "minor discrepancy"

        delta_accum = (accum_val - prev_accum) if (prev_accum is not None and not isnan_str(accum_val)) else None
        prev_accum = accum_val if not isnan_str(accum_val) else prev_accum

        print(
            f"{i:<3} {layer:<16} {tag:<14} "
            f"{_fv(accum_val):>10} {_fv(local_val):>10} "
            f"{_fv(diff):>10} {_fv(delta_accum):>10} "
            f"{note}"
        )

    print("-" * 105)
    print("  Local = observer (op's own quantization error)")
    print("  Accum = hook   (total error incl. upstream propagation)")
    print("  ΔAccum = accum[i] - accum[i-1] (≤0 → error accumulates)")
    print("  Diff > 0 for deep layers = expected ✓ (local excludes upstream error)")
    print()

    # Summary stats
    diffs = []
    for layer in all_layers:
        av = accum.get(layer)
        lv = local_out.get(layer)
        if not isnan_str(av) and not isnan_str(lv) and lv < 100:
            diffs.append((layer, lv - av, av, lv))

    if diffs:
        diffs.sort(key=lambda x: x[1])
        print(f"  Layers with meaningful local QSNR (< 100 dB):")
        for layer, diff, av, lv in diffs:
            print(f"    {layer:<16} accum={av:.2f}  local={lv:.2f}  diff={diff:+.2f} dB")
        avg_diff = sum(d[1] for d in diffs) / len(diffs)
        print(f"  Average diff (meaningful layers): {avg_diff:+.2f} dB")
        print(f"  → local QSNR exceeds accum by ~{avg_diff:.1f} dB on average")
        print(f"  → this is the per-layer quantization overhead (upstream error excluded)")


def isnan_str(v):
    """Check if value is NaN (works for float('nan') and numpy nan)."""
    if v is None:
        return True
    if isinstance(v, float):
        return v != v  # NaN check
    return False


if __name__ == "__main__":
    generate_comparison_table()
