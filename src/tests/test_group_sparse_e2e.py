"""
End-to-end integration tests for ADR-013 Group Sparse via Session / Study APIs.

Covers all granularity modes, edge-case shapes, format combos, transforms,
weight-only mode, and scale-path safety (the topk-out-of-range bug fix).

All tests use high-level public APIs only: Session, Study, QuantConfig.
"""

import pytest
import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._compat import Session
from src.session._study import Study


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_calib(batch=4, features=16):
    return torch.randn(batch, features)


def _dummy_eval_fn(model, data):
    """eval_fn compatible with Session.run — returns a scalar metric dict.

    During calibration/analysis, *data* is a list of tensors.
    During evaluation, *data* is a list of (x, y) tuples or a DataLoader.
    """
    model.eval()
    with torch.no_grad():
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    x, _y = item
                    model(x)
                elif isinstance(item, torch.Tensor):
                    model(item)
                else:
                    model(item)
        elif isinstance(data, torch.Tensor):
            model(data)
        else:
            # DataLoader or other iterable
            for batch in data:
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    x, _y = batch
                    model(x)
                else:
                    model(batch)
    return {"dummy": 0.5}


def _dummy_eval_data(batch=4, features=16, num_classes=4):
    return [(torch.randn(batch, features), torch.randint(0, num_classes, (batch,)))]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Granularity mode coverage — every mode works with group_sparse
# ═══════════════════════════════════════════════════════════════════════════

class _MLP(nn.Module):
    def __init__(self, in_dim=16, hid_dim=8, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@pytest.mark.parametrize("granularity,block_size", [
    ("per_tensor", None),
    ("per_channel", None),
    ("per_block", 4),
    ("bank", 4),
])
def test_group_sparse_all_granularities(granularity, block_size):
    """Every granularity mode works end-to-end with group_sparse via Session."""
    cfg = QuantConfig(
        name=f"gs-{granularity}",
        w_format="int4", w_granularity=granularity, w_block_size=block_size,
        group_format="int8", group_ratio=0.3,
    )
    model = _MLP()
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(_make_calib())
    assert result.quant_metrics is None  # no eval → no metrics
    # qmodel ran without error — success


# ═══════════════════════════════════════════════════════════════════════════
# 2. group_ratio edge cases — boundary values
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("ratio", [0.01, 0.1, 0.3, 0.5, 0.99, 1.0])
def test_group_sparse_ratio_boundaries(ratio):
    """group_ratio across the full (0, 1] range — no topk out-of-range."""
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        group_format="int8", group_ratio=ratio,
    )
    model = _MLP(in_dim=8, hid_dim=4, out_dim=2)
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(_make_calib(features=8))
    # No crash = pass (k = max(1, int(C * ratio)) must stay ≤ C)
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# 3. Minimal channel count — triggers the old topk-out-of-range bug
# ═══════════════════════════════════════════════════════════════════════════

def test_group_sparse_minimal_channels():
    """2 output channels + group_ratio=0.3 → k=1, with scores.numel()==2.

    Before the fix, k was computed from C=x.shape[axis] but scores came from
    amax.flatten(). If scale had a mismatched shape, topk would fail.
    This test exercises both the dynamic path (no calibration) and the
    calibrated path to ensure both are safe.
    """
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        a_format="int4", a_granularity="per_channel",
        group_format="int8", group_ratio=0.3,
    )
    # C=2 along axis=0 (weight: out_features=2 → (2,4))
    model = nn.Sequential(nn.Linear(4, 2))
    session = Session(model, cfg, keep_fp32=True)
    # Calibration runs with scale → exercises the scale path
    result = session.run(
        _make_calib(batch=2, features=4),
        eval_fn=_dummy_eval_fn,
        eval_data=_dummy_eval_data(batch=2, features=4, num_classes=2),
        outputs=["qsnr"],
    )
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# 4. Shape diversity — 1D / 2D / 4D weight tensors
# ═══════════════════════════════════════════════════════════════════════════

def test_group_sparse_1d_bias_tensor():
    """Bias tensors are 1D — group_sparse must handle single-dim tensors."""
    model = nn.Sequential(nn.Linear(8, 4, bias=True))
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        group_format="int8", group_ratio=0.3,
    )
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(_make_calib(features=8))
    assert result is not None


def test_group_sparse_conv2d():
    """Conv2d weights are 4D — per_channel axis covers output channels."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 4),
    )
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        group_format="int8", group_ratio=0.3,
    )
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(torch.randn(2, 3, 8, 8))
    assert result is not None


def test_group_sparse_embedding():
    """Embedding weight is 2D (vocab_size, emb_dim) — per_channel covers vocab dimension."""
    model = nn.Sequential(
        nn.Embedding(10, 8),
        nn.Flatten(),
        nn.Linear(40, 4),  # 5 tokens * 8 dims = 40
    )
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        group_format="int8", group_ratio=0.3,
    )
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(torch.randint(0, 10, (4, 5)))  # (batch=4, seq=5)
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# 5. Format combinations
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("w_fmt,group_fmt", [
    ("int4", "int8"),
    ("int8", "fp8_e4m3"),
    ("int4", "fp8_e5m2"),
    ("fp8_e4m3", "int8"),
])
def test_group_sparse_format_combos(w_fmt, group_fmt):
    """Various (format, group_format) pairs — especially float + int combos."""
    cfg = QuantConfig(
        w_format=w_fmt, w_granularity="per_channel",
        group_format=group_fmt, group_ratio=0.3,
    )
    model = _MLP()
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(_make_calib())
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# 6. Weight-only mode
# ═══════════════════════════════════════════════════════════════════════════

def test_group_sparse_weight_only():
    """weight_only=True — activations are NOT quantized, only weights."""
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        group_format="int8", group_ratio=0.3,
        weight_only=True,
    )
    model = _MLP()
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(_make_calib())
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# 7. Group sparse with transforms
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("transform", ["hadamard", "smoothquant", "prescale"])
def test_group_sparse_with_transform(transform):
    """group_sparse + transform — transform applied before group-aware quantize."""
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        a_format="int4", a_granularity="per_channel",
        group_format="int8", group_ratio=0.3,
        transform=transform,
        quantize_nonlinear=False,
    )
    model = _MLP()
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(_make_calib())
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# 8. LayerNorm model — exercises non-linear path with group_sparse
# ═══════════════════════════════════════════════════════════════════════════

def test_group_sparse_layernorm_model():
    """LayerNorm weight (1D) + group_sparse — per_channel on norm weight."""
    model = nn.Sequential(
        nn.Linear(8, 8),
        nn.LayerNorm(8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        group_format="int8", group_ratio=0.3,
    )
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(_make_calib(features=8))
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# 9. Full Session pipeline — eval + analysis outputs
# ═══════════════════════════════════════════════════════════════════════════

def test_group_sparse_full_session_pipeline():
    """Complete Session.run() with eval_fn, eval_data, and all observer outputs."""
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        a_format="int4", a_granularity="per_channel",
        group_format="int8", group_ratio=0.3,
        quantize_nonlinear=False,
    )
    model = _MLP()
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(
        _make_calib(),
        eval_fn=_dummy_eval_fn,
        eval_data=_dummy_eval_data(),
        outputs=["distribution", "qsnr", "histogram"],
    )
    assert result is not None
    # SessionResult should have observer data populated
    assert hasattr(result, "qsnr_per_layer")


# ═══════════════════════════════════════════════════════════════════════════
# 10. Study API — multi-config comparison
# ═══════════════════════════════════════════════════════════════════════════

def test_group_sparse_study_api():
    """Study API comparing group_sparse vs baseline configs."""
    configs = [
        QuantConfig(
            name="gs-baseline",
            w_format="int4", w_granularity="per_channel",
            a_format="int4", a_granularity="per_channel",
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="gs-int8-0.3",
            w_format="int4", w_granularity="per_channel",
            a_format="int4", a_granularity="per_channel",
            group_format="int8", group_ratio=0.3,
            quantize_nonlinear=False,
        ),
        QuantConfig(
            name="gs-int8-1.0",
            w_format="int4", w_granularity="per_channel",
            a_format="int4", a_granularity="per_channel",
            group_format="int8", group_ratio=1.0,
            quantize_nonlinear=False,
        ),
    ]
    model = _MLP()
    study = Study(configs, model=model)
    report = study.run(
        [_make_calib() for _ in range(4)],
        eval_fn=_dummy_eval_fn,
        eval_data=_dummy_eval_data(),
        outputs="all",
    )
    assert report is not None
    # All three configs should have results (in the "default" part)
    assert report.total_experiments == 3


# ═══════════════════════════════════════════════════════════════════════════
# 11. Activation-only group_format (a_group_format override)
# ═══════════════════════════════════════════════════════════════════════════

def test_group_sparse_activation_override():
    """a_group_format overrides group_format for activations."""
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        a_format="int4", a_granularity="per_channel",
        group_format="int8", group_ratio=0.3,
        a_group_format="fp8_e4m3", a_group_ratio=0.5,
        quantize_nonlinear=False,
    )
    model = _MLP()
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(_make_calib())
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# 12. quantize_nonlinear=True path
# ═══════════════════════════════════════════════════════════════════════════

def test_group_sparse_quantize_nonlinear():
    """group_sparse works when quantize_nonlinear=True (norm/activation/pool ops too)."""
    cfg = QuantConfig(
        w_format="int4", w_granularity="per_channel",
        a_format="int4", a_granularity="per_channel",
        group_format="int8", group_ratio=0.3,
        quantize_nonlinear=True,
    )
    model = _MLP()
    session = Session(model, cfg, keep_fp32=True)
    result = session.run(_make_calib())
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# 13. Direct quantize() safety net — scale shape mismatch (the original bug)
# ═══════════════════════════════════════════════════════════════════════════

def test_group_sparse_direct_scale_mismatch_safety():
    """Direct quantize() with a scalar scale for per_channel — must not crash.

    This is the exact bug pattern: when scale is passed with a different
    shape than the expected per-channel amax, k (computed from scores.numel())
    must not exceed the actual number of elements in scores.
    """
    from src.formats.base import FormatBase
    from src.scheme.granularity import GranularitySpec

    fmt_int4 = FormatBase.from_str("int4")
    fmt_int8 = FormatBase.from_str("int8")

    # Per-channel with 4 channels, but pass a scalar scale
    x = torch.randn(4, 8)
    g = GranularitySpec.per_channel(axis=0)

    out = fmt_int4.quantize(x, g,
                            group_format=fmt_int8, group_ratio=0.3,
                            scale=torch.tensor(2.0))  # scalar, not (4,) shape
    assert out.shape == x.shape


def test_group_sparse_direct_scale_correct_shape():
    """Direct quantize() with correctly-shaped per-channel scale — no degradation."""
    from src.formats.base import FormatBase
    from src.scheme.granularity import GranularitySpec

    fmt_int4 = FormatBase.from_str("int4")
    fmt_int8 = FormatBase.from_str("int8")

    x = torch.randn(4, 8)
    g = GranularitySpec.per_channel(axis=0)
    scale = torch.tensor([[2.0], [3.0], [1.0], [4.0]])  # (4, 1) — correct shape

    out = fmt_int4.quantize(x, g,
                            group_format=fmt_int8, group_ratio=0.3,
                            scale=scale)
    assert out.shape == x.shape


# ═══════════════════════════════════════════════════════════════════════════
# 14. Bank-specific edge cases
# ═══════════════════════════════════════════════════════════════════════════

def test_group_sparse_bank_small():
    """BANK granularity with minimal banks — k computation must be safe."""
    from src.formats.base import FormatBase
    from src.scheme.granularity import GranularityMode, GranularitySpec

    fmt_int4 = FormatBase.from_str("int4")
    fmt_int8 = FormatBase.from_str("int8")

    # (2, 8), bank_size=4 → 2 banks, k=max(1,int(2*0.3))=1
    x = torch.randn(2, 8)
    g = GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1)

    out = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.3)
    assert out.shape == x.shape

    # With correctly-shaped scale: (num_banks,) = (2,)
    # BANK reshaping: (2, 8) → (2, 2, 4), amax reduces dims [0,1] → (1, 1, 2)
    out2 = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.3,
                             scale=torch.tensor([2.0, 3.0]).view(1, 1, 2))
    assert out2.shape == x.shape


def test_group_sparse_bank_minimal():
    """BANK with just 1 bank — degenerate but must not crash."""
    from src.formats.base import FormatBase
    from src.scheme.granularity import GranularityMode, GranularitySpec

    fmt_int4 = FormatBase.from_str("int4")
    fmt_int8 = FormatBase.from_str("int8")

    x = torch.randn(4, 4)  # bank_size=4 → 1 bank
    g = GranularitySpec(mode=GranularityMode.BANK, bank_size=4, bank_axis=-1)

    out = fmt_int4.quantize(x, g, group_format=fmt_int8, group_ratio=0.5)
    assert out.shape == x.shape


# ═══════════════════════════════════════════════════════════════════════════
# 15. Interaction with outlier_format — mutual exclusivity enforced
# ═══════════════════════════════════════════════════════════════════════════

def test_group_sparse_and_outlier_mutually_exclusive_config():
    """QuantConfig rejects group_format + outlier_format together."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        QuantConfig(
            w_format="int4", w_granularity="per_channel",
            group_format="int8", group_ratio=0.3,
            outlier_format="fp8_e4m3", outlier_ratio=0.1,
        )
