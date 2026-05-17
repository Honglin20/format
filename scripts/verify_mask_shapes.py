#!/usr/bin/env python3
"""Comprehensive E2E mask shape verification — exhaustive coverage.

Exercises every path where static sparse masks are computed and consumed.
Key additions over the basic gate:
  - Variable batch sizes *during* calibration (every round different)
  - Many model sizes: tiny (4-dim) to wide (512-dim)
  - Conv1d/Conv2d with varied C, H, W, kernel, stride, bank_axis
  - All granularity modes × all op types × outlier_format combos
  - Degenerate cases, divisibility boundaries, single-bank, near-100% ratio
  - Mixed Linear+Conv models
  - Determinism: different batch schedules → same mask
  - Multi-eval: same model inferenced with 3+ different eval batch sizes
"""

import torch
import sys
from dataclasses import replace
sys.path.insert(0, ".")

from src.scheme.op_config import OpQuantConfig
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularityMode, GranularitySpec
from src.formats.base import FormatBase
from src.session._model import quantize_model
from src.calibration.pipeline import CalibrationSession
from src.calibration.strategies import MaxScaleStrategy

PASS = 0
FAIL = 0


def check(condition, label):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  OK  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}")


def check_eq(actual, expected, label):
    ok = actual == expected
    if not ok:
        print(f"    expected: {expected}, got: {actual}")
    check(ok, label)


def check_shape(tensor, expected_shape, label):
    ok = tuple(tensor.shape) == tuple(expected_shape)
    if not ok:
        print(f"    expected shape: {expected_shape}, got: {tuple(tensor.shape)}")
    check(ok, f"{label}: shape {tuple(tensor.shape)}")


def check_no_nan_inf(tensor, label):
    ok = torch.isfinite(tensor).all()
    if not ok:
        n_nan = torch.isnan(tensor).sum().item()
        n_inf = torch.isinf(tensor).sum().item()
        print(f"    NaN={n_nan}, Inf={n_inf}")
    check(ok, f"{label}: no NaN/Inf")


def check_mask_dim0(mask, label):
    """Verify mask batch dim is 1 and mask is boolean."""
    check(mask.dtype == torch.bool, f"{label}: dtype bool")
    check(mask.shape[0] == 1, f"{label}: dim0==1 (shape {tuple(mask.shape)})")


def make_scheme(fmt_str, gran_mode, outlier_ratio=0.0, bank_size=16,
                bank_axis=-1, channel_axis=-1):
    fmt = FormatBase.from_str(fmt_str)
    if gran_mode == "per_tensor":
        g = GranularitySpec(mode=GranularityMode.PER_TENSOR,
                            outlier_ratio=outlier_ratio)
    elif gran_mode == "per_channel":
        g = GranularitySpec(mode=GranularityMode.PER_CHANNEL,
                            outlier_ratio=outlier_ratio,
                            channel_axis=channel_axis)
    elif gran_mode == "bank":
        g = GranularitySpec(mode=GranularityMode.BANK,
                            outlier_ratio=outlier_ratio,
                            bank_size=bank_size,
                            bank_axis=bank_axis)
    else:
        raise ValueError(f"Unknown mode: {gran_mode}")
    return QuantScheme(format=fmt, granularity=g, scale_storage="fp32")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def calib_run_fixed(model, calib_data, batch_size, track_input=True):
    """Calibrate with fixed batch size."""
    with CalibrationSession(model, MaxScaleStrategy(),
                            track_input=track_input, sparse=True):
        for i in range(0, len(calib_data), batch_size):
            batch = calib_data[i:i + batch_size]
            xb = torch.stack(batch, dim=0)
            with torch.no_grad():
                model(xb)


def calib_run_variable(model, calib_data, batch_sizes, track_input=True):
    """Calibrate with *different* batch size each round.

    batch_sizes = [3, 7, 2, 5, ...] — one per round.
    Repeats if more data than batch_sizes.
    """
    with CalibrationSession(model, MaxScaleStrategy(),
                            track_input=track_input, sparse=True):
        idx = 0
        round_idx = 0
        while idx < len(calib_data):
            bs = batch_sizes[round_idx % len(batch_sizes)]
            batch = calib_data[idx:idx + bs]
            if not batch:
                break
            xb = torch.stack(batch, dim=0)
            with torch.no_grad():
                model(xb)
            idx += bs
            round_idx += 1


def verify_buffers(qmodel, layer_name, spatial_shape, gran_mode, expected_scale_numel=None):
    """Verify mask, scale_n, scale_o shapes on a module."""
    layer = qmodel
    for part in layer_name.split("."):
        layer = getattr(layer, part)

    mask = layer.get_buffer("_input_mask")
    expected_mask_shape = (1,) + tuple(spatial_shape)
    check_shape(mask, expected_mask_shape, f"{layer_name} input_mask")
    check_mask_dim0(mask, f"{layer_name} input_mask")

    scale = layer.get_buffer("_input_scale")
    check_no_nan_inf(scale, f"{layer_name} input_scale")
    if expected_scale_numel is not None:
        check_eq(scale.numel(), expected_scale_numel,
                 f"{layer_name} input_scale numel={expected_scale_numel}")

    scale_o = layer.get_buffer("_input_scale_o")
    check_no_nan_inf(scale_o, f"{layer_name} input_scale_o")
    if expected_scale_numel is not None:
        check_eq(scale_o.numel(), expected_scale_numel,
                 f"{layer_name} input_scale_o numel={expected_scale_numel}")


def verify_forward(qmodel, eval_shapes, tag):
    """Forward with multiple eval batch sizes. All must produce finite output."""
    for b, *spatial in eval_shapes:
        x = torch.randn(b, *spatial)
        try:
            with torch.no_grad():
                out = qmodel(x)
            check(torch.isfinite(out).all(),
                  f"{tag}: forward(B={b}) finite")
        except Exception as e:
            check(False, f"{tag}: forward(B={b}) crashed: {e}")


# ═══════════════════════════════════════════════════════════════════
# Convenience model builders
# ═══════════════════════════════════════════════════════════════════

class TinyLinear(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = torch.nn.Linear(in_f, out_f)
    def forward(self, x):
        return self.linear(x)


class TinyConv1d(torch.nn.Module):
    def __init__(self, in_c, out_c, kernel=3, padding=1, stride=1, length=10):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_c, out_c, kernel_size=kernel,
                                    padding=padding, stride=stride)
        self._L = length
    def forward(self, x):
        return self.conv(x)


class TinyConv2d(torch.nn.Module):
    def __init__(self, in_c, out_c, kernel=3, padding=1, stride=1, H=8, W=8):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_c, out_c, kernel_size=kernel,
                                    padding=padding, stride=stride)
        self._H, self._W = H, W
    def forward(self, x):
        return self.conv(x)


def conv1d_out_len(L, kernel, padding, stride):
    return (L + 2 * padding - kernel) // stride + 1


def conv2d_out_len(HW, kernel, padding, stride):
    return (HW + 2 * padding - kernel) // stride + 1


# ═══════════════════════════════════════════════════════════════════
# §1  VARIABLE BATCH SIZE during calibration
# ═══════════════════════════════════════════════════════════════════

print("=" * 60)
print("§1  Variable batch size during calibration")
print("=" * 60)

# 1.1a — Linear BANK, variable batch [3,7,2,5,...]
print("\n1.1a Linear BANK: variable calib batches [3,7,2,5] → eval B=11,31,64")
scheme = make_scheme("int4", "bank", outlier_ratio=0.25, bank_size=4)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(8, 8)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(8) for _ in range(50)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 7, 2, 5, 1, 8])
verify_buffers(qmodel, "linear", [8], "bank", expected_scale_numel=2)
verify_forward(qmodel, [(11, 8), (31, 8), (64, 8)], "1.1a")

# 1.1b — Same as 1.1a but with fixed batch=4: mask should match
torch.manual_seed(42)
calib_data_b = [torch.randn(8) for _ in range(50)]
model_b = TinyLinear(8, 8)
qmodel_b = quantize_model(model_b, cfg=cfg)
calib_run_fixed(qmodel_b, calib_data_b, batch_size=4)
# Different batch schedules with different data → masks will differ.
# Just verify shapes are correct, not content match.
check_mask_dim0(qmodel_b.linear.get_buffer("_input_mask"), "1.1b fixed-batch mask")

# 1.2 — Conv1d BANK, variable calib batches
print("\n1.2 Conv1d BANK: variable calib batches [2,1,5,3] → eval B=7,19,43")
scheme_in = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
scheme_w = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=0)
scheme_out = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
model = TinyConv1d(8, 4, kernel=3, padding=1, length=12)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(8, 12) for _ in range(30)]
calib_run_variable(qmodel, calib_data, batch_sizes=[2, 1, 5, 3])
verify_buffers(qmodel, "conv", [8, 12], "bank", expected_scale_numel=2)
verify_forward(qmodel, [(7, 8, 12), (19, 8, 12), (43, 8, 12)], "1.2")

# 1.3 — Conv2d PER_CHANNEL, variable calib batches
print("\n1.3 Conv2d PER_CHANNEL: variable calib batches [1,3,2] → eval B=5,17,37")
scheme_in = make_scheme("int4", "per_channel", outlier_ratio=0.15, channel_axis=1)
scheme_w = make_scheme("int4", "per_channel", outlier_ratio=0.15, channel_axis=0)
scheme_out = make_scheme("int4", "per_channel", outlier_ratio=0.15, channel_axis=1)
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
model = TinyConv2d(6, 10, kernel=3, padding=1, H=8, W=8)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(6, 8, 8) for _ in range(12)]
calib_run_variable(qmodel, calib_data, batch_sizes=[1, 3, 2])
verify_buffers(qmodel, "conv", [6, 8, 8], "per_channel", expected_scale_numel=6)
verify_forward(qmodel, [(5, 6, 8, 8), (17, 6, 8, 8), (37, 6, 8, 8)], "1.3")

# 1.4 — Linear, all batch_size=1 during calib (extreme: every sample alone)
print("\n1.4 Linear PER_TENSOR: all calib batches size=1 → eval B=100")
scheme = make_scheme("int4", "per_tensor", outlier_ratio=0.3)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(16, 16)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(16) for _ in range(20)]
calib_run_variable(qmodel, calib_data, batch_sizes=[1])  # every round bs=1
verify_buffers(qmodel, "linear", [16], "per_tensor", expected_scale_numel=1)
verify_forward(qmodel, [(100, 16)], "1.4")

# ═══════════════════════════════════════════════════════════════════
# §2  MANY MODEL SIZES — Linear
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§2  Many Linear model sizes")
print("=" * 60)

linear_sizes = [
    # (in_f, out_f, bank_size, num_banks_in, num_banks_out, label)
    (4, 4, 4, 1, 1, "tiny 4→4 bs=4"),
    (16, 8, 4, 4, 2, "small 16→8 bs=4"),
    (32, 32, 8, 4, 4, "medium 32→32 bs=8"),
    (64, 16, 4, 16, 4, "wide-in 64→16 bs=4"),
    (8, 64, 4, 2, 16, "wide-out 8→64 bs=4"),
    (128, 256, 16, 8, 16, "large 128→256 bs=16"),
    (256, 128, 8, 32, 16, "large 256→128 bs=8"),
    (512, 512, 32, 16, 16, "xlarge 512→512 bs=32"),
]

for in_f, out_f, bs, nb_in, nb_out, label in linear_sizes:
    print(f"\n2.{linear_sizes.index((in_f, out_f, bs, nb_in, nb_out, label))} Linear {label}")
    scheme = make_scheme("int8", "bank", outlier_ratio=0.2, bank_size=bs)
    cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
    model = TinyLinear(in_f, out_f)
    qmodel = quantize_model(model, cfg=cfg)
    calib_data = [torch.randn(in_f) for _ in range(max(30, nb_in * 4))]
    calib_run_variable(qmodel, calib_data,
                       batch_sizes=[3, 7, 5] if len(calib_data) > 20 else [2, 3])

    verify_buffers(qmodel, "linear", [in_f], "bank", expected_scale_numel=nb_in)
    # Eval with several batch sizes
    verify_forward(qmodel, [(1, in_f), (17, in_f), (59, in_f)], label)

    # Output mask
    out_mask = qmodel.linear.get_buffer("_output_mask")
    check_mask_dim0(out_mask, f"output_mask {label}")
    expected_out_spatial = [out_f]
    check_shape(out_mask, (1,) + tuple(expected_out_spatial),
                f"output_mask {label}")

# ═══════════════════════════════════════════════════════════════════
# §3  MANY MODEL SIZES — Conv1d
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§3  Many Conv1d sizes")
print("=" * 60)

conv1d_configs = [
    # (in_c, out_c, L, kernel, pad, stride, bs, nb, label)
    (4, 8, 12, 3, 1, 1, 4, 1, "C=4→8 L=12 k3 bs=4"),
    (8, 8, 20, 3, 1, 1, 4, 2, "C=8→8 L=20 k3 bs=4"),
    (12, 4, 16, 5, 2, 1, 4, 3, "C=12→4 L=16 k5 bs=4"),
    (16, 16, 32, 3, 1, 1, 8, 2, "C=16→16 L=32 k3 bs=8"),
    (6, 24, 8, 3, 1, 1, 2, 3, "C=6→24 L=8 k3 bs=2"),
    (24, 6, 24, 3, 1, 2, 6, 4, "C=24→6 L=24 k3 s=2 bs=6"),
    (32, 32, 48, 3, 1, 1, 16, 2, "C=32→32 L=48 k3 bs=16"),
]

for in_c, out_c, L, kernel, pad, stride, bs, nb, label in conv1d_configs:
    print(f"\n3.{conv1d_configs.index((in_c, out_c, L, kernel, pad, stride, bs, nb, label))} Conv1d {label}")
    scheme_in = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=bs, bank_axis=1)
    scheme_w = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=bs, bank_axis=0)
    scheme_out = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=bs, bank_axis=1)
    cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
    model = TinyConv1d(in_c, out_c, kernel=kernel, padding=pad, stride=stride, length=L)
    qmodel = quantize_model(model, cfg=cfg)
    calib_data = [torch.randn(in_c, L) for _ in range(max(20, nb * 3))]
    calib_run_variable(qmodel, calib_data, batch_sizes=[3, 5, 2])

    verify_buffers(qmodel, "conv", [in_c, L], "bank", expected_scale_numel=nb)
    out_L = conv1d_out_len(L, kernel, pad, stride)
    out_mask = qmodel.conv.get_buffer("_output_mask")
    check_mask_dim0(out_mask, f"output_mask {label}")
    verify_forward(qmodel, [(7, in_c, L), (31, in_c, L), (99, in_c, L)], label)

# ═══════════════════════════════════════════════════════════════════
# §4  MANY MODEL SIZES — Conv2d
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§4  Many Conv2d sizes")
print("=" * 60)

conv2d_configs = [
    # (in_c, out_c, H, W, kernel, pad, stride, bs, nb, label)
    (4, 8, 8, 8, 3, 1, 1, 4, 1, "C=4→8 8x8 k3 bs=4"),
    (8, 4, 12, 12, 3, 1, 1, 4, 2, "C=8→4 12x12 k3 bs=4"),
    (16, 16, 6, 6, 3, 1, 1, 8, 2, "C=16→16 6x6 k3 bs=8"),
    (6, 18, 10, 10, 5, 2, 1, 2, 3, "C=6→18 10x10 k5 bs=2"),
    (12, 24, 16, 8, 3, 1, 2, 4, 3, "C=12→24 16x8 k3 s=2 bs=4"),
    (24, 12, 14, 14, 3, 1, 1, 6, 4, "C=24→12 14x14 k3 bs=6"),
    (10, 10, 20, 20, 3, 1, 1, 2, 5, "C=10→10 20x20 k3 bs=2"),
]

for in_c, out_c, H, W, kernel, pad, stride, bs, nb, label in conv2d_configs:
    print(f"\n4.{conv2d_configs.index((in_c, out_c, H, W, kernel, pad, stride, bs, nb, label))} Conv2d {label}")
    scheme_in = make_scheme("int4", "bank", outlier_ratio=0.12, bank_size=bs, bank_axis=1)
    scheme_w = make_scheme("int4", "bank", outlier_ratio=0.12, bank_size=bs, bank_axis=0)
    scheme_out = make_scheme("int4", "bank", outlier_ratio=0.12, bank_size=bs, bank_axis=1)
    cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
    model = TinyConv2d(in_c, out_c, kernel=kernel, padding=pad, stride=stride, H=H, W=W)
    qmodel = quantize_model(model, cfg=cfg)
    calib_data = [torch.randn(in_c, H, W) for _ in range(max(20, nb * 3))]
    calib_run_variable(qmodel, calib_data, batch_sizes=[3, 1, 4, 2])

    verify_buffers(qmodel, "conv", [in_c, H, W], "bank", expected_scale_numel=nb)
    out_H = conv2d_out_len(H, kernel, pad, stride)
    out_W = conv2d_out_len(W, kernel, pad, stride)
    out_mask = qmodel.conv.get_buffer("_output_mask")
    check_mask_dim0(out_mask, f"output_mask {label}")
    verify_forward(qmodel, [(5, in_c, H, W), (23, in_c, H, W), (47, in_c, H, W)], label)

# ═══════════════════════════════════════════════════════════════════
# §5  DIFFERENT BANK_AXIS values for Conv
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§5  Different bank_axis values for Conv")
print("=" * 60)

# 5.1 — Conv1d: bank on spatial dim (axis=2 or -1)
print("\n5.1 Conv1d: bank on spatial (axis=-1, L=16, bs=4)")

class Conv1dSpatial(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1)
    def forward(self, x):
        return self.conv(x)

scheme_in = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=-1)
scheme_w = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=0)
scheme_out = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=-1)
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
model = Conv1dSpatial()
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(4, 16) for _ in range(20)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 5, 2, 7])
# 16/4=4 banks on spatial dim
verify_buffers(qmodel, "conv", [4, 16], "bank", expected_scale_numel=4)
verify_forward(qmodel, [(7, 4, 16), (31, 4, 16), (55, 4, 16)], "5.1")

# 5.2 — Conv2d: bank on H dim (axis=2)
print("\n5.2 Conv2d: bank on H (axis=2, H=12, bs=4)")

class Conv2dBankH(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1)
    def forward(self, x):
        return self.conv(x)

scheme_in = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=2)
scheme_w = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=0)
scheme_out = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=2)
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
model = Conv2dBankH()
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(4, 12, 8) for _ in range(20)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 4, 2])
# H=12, bs=4 → 3 banks on H
verify_buffers(qmodel, "conv", [4, 12, 8], "bank", expected_scale_numel=3)
verify_forward(qmodel, [(11, 4, 12, 8), (37, 4, 12, 8), (87, 4, 12, 8)], "5.2")

# 5.3 — Conv2d: bank on W dim (axis=-1)
print("\n5.3 Conv2d: bank on W (axis=-1, W=16, bs=8)")

class Conv2dBankW(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, kernel_size=3, padding=1)
    def forward(self, x):
        return self.conv(x)

scheme_in = make_scheme("int8", "bank", outlier_ratio=0.1, bank_size=8, bank_axis=-1)
scheme_w = make_scheme("int8", "bank", outlier_ratio=0.1, bank_size=8, bank_axis=0)
scheme_out = make_scheme("int8", "bank", outlier_ratio=0.1, bank_size=8, bank_axis=-1)
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
model = Conv2dBankW()
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(8, 8, 16) for _ in range(24)]
calib_run_variable(qmodel, calib_data, batch_sizes=[4, 1, 6, 3])
# W=16, bs=8 → 2 banks on W; weight axis=0 dim=8, 8%8=0 → 1 bank
verify_buffers(qmodel, "conv", [8, 8, 16], "bank", expected_scale_numel=2)
verify_forward(qmodel, [(9, 8, 8, 16), (41, 8, 8, 16), (73, 8, 8, 16)], "5.3")

# ═══════════════════════════════════════════════════════════════════
# §6  OUTLIER_FORMAT combinations
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§6  outlier_format combinations")
print("=" * 60)

# 6.1 — int4 base + int8 outlier
print("\n6.1 int4 base + int8 outlier_format (Linear, BANK)")
scheme = make_scheme("int4", "bank", outlier_ratio=0.25, bank_size=4)
scheme = replace(scheme, outlier_format=FormatBase.from_str("int8"))
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(16, 8)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(16) for _ in range(24)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 5, 2, 7, 1])
verify_buffers(qmodel, "linear", [16], "bank", expected_scale_numel=4)
verify_forward(qmodel, [(3, 16), (29, 16), (97, 16)], "6.1")

# 6.2 — int8 base + int4 outlier (inverse)
print("\n6.2 int8 base + int4 outlier_format (Linear, PER_TENSOR)")
scheme = make_scheme("int8", "per_tensor", outlier_ratio=0.15)
scheme = replace(scheme, outlier_format=FormatBase.from_str("int4"))
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(32, 32)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(32) for _ in range(30)]
calib_run_variable(qmodel, calib_data, batch_sizes=[5, 3, 7, 2])
verify_buffers(qmodel, "linear", [32], "per_tensor", expected_scale_numel=1)
verify_forward(qmodel, [(13, 32), (51, 32), (99, 32)], "6.2")

# 6.3 — int4 base + int8 outlier (Conv2d, BANK)
print("\n6.3 int4 base + int8 outlier_format (Conv2d, BANK, axis=1)")
scheme_in = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
scheme_in = replace(scheme_in, outlier_format=FormatBase.from_str("int8"))
scheme_w = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=0)
scheme_out = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
scheme_out = replace(scheme_out, outlier_format=FormatBase.from_str("int8"))
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
model = TinyConv2d(8, 12, kernel=3, padding=1, H=10, W=10)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(8, 10, 10) for _ in range(16)]
calib_run_variable(qmodel, calib_data, batch_sizes=[2, 4, 1, 3])
verify_buffers(qmodel, "conv", [8, 10, 10], "bank", expected_scale_numel=2)
verify_forward(qmodel, [(7, 8, 10, 10), (33, 8, 10, 10), (61, 8, 10, 10)], "6.3")

# 6.4 — int4 base + bfloat16 outlier (float format outlier)
print("\n6.4 int4 base + bfloat16 outlier_format (Linear, PER_CHANNEL)")
scheme = make_scheme("int4", "per_channel", outlier_ratio=0.1, channel_axis=-1)
scheme = replace(scheme, outlier_format=FormatBase.from_str("bfloat16"))
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(16, 8)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(16) for _ in range(20)]
calib_run_variable(qmodel, calib_data, batch_sizes=[4, 2, 6])
verify_buffers(qmodel, "linear", [16], "per_channel", expected_scale_numel=16)
verify_forward(qmodel, [(9, 16), (45, 16), (103, 16)], "6.4")

# ═══════════════════════════════════════════════════════════════════
# §7  DIFFERENT FORMAT TYPES (int8, fp16, bfloat16)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§7  Different format types")
print("=" * 60)

# 7.1 — int8 base (more levels, larger range)
print("\n7.1 int8 base BANK, ratio=0.3")
scheme = make_scheme("int8", "bank", outlier_ratio=0.3, bank_size=8)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(32, 16)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(32) for _ in range(24)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 6, 2, 5])
verify_buffers(qmodel, "linear", [32], "bank", expected_scale_numel=4)
verify_forward(qmodel, [(11, 32), (49, 32), (81, 32)], "7.1")

# 7.2 — int8 base PER_CHANNEL
print("\n7.2 int8 base PER_CHANNEL, ratio=0.18")
scheme = make_scheme("int8", "per_channel", outlier_ratio=0.18, channel_axis=-1)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(24, 12)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(24) for _ in range(20)]
calib_run_variable(qmodel, calib_data, batch_sizes=[2, 5, 3, 1])
verify_buffers(qmodel, "linear", [24], "per_channel", expected_scale_numel=24)
verify_forward(qmodel, [(7, 24), (53, 24), (119, 24)], "7.2")

# 7.3 — int8 base Conv2d BANK
print("\n7.3 int8 base Conv2d BANK, ratio=0.15")
scheme_in = make_scheme("int8", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=1)
scheme_w = make_scheme("int8", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=0)
scheme_out = make_scheme("int8", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=1)
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
model = TinyConv2d(12, 8, kernel=3, padding=1, H=12, W=12)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(12, 12, 12) for _ in range(20)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 4, 2, 1])
verify_buffers(qmodel, "conv", [12, 12, 12], "bank", expected_scale_numel=3)
verify_forward(qmodel, [(7, 12, 12, 12), (39, 12, 12, 12), (71, 12, 12, 12)], "7.3")

# ═══════════════════════════════════════════════════════════════════
# §8  DEGENERATE / EDGE CASES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§8  Degenerate and edge cases")
print("=" * 60)

# 8.1 — Single bank (bank_size == dim)
print("\n8.1 BANK: bank_size == dim (single bank), multiple sizes")
for dim, bs_val, label in [(4, 4, "dim=4,bs=4"), (8, 8, "dim=8,bs=8"),
                             (16, 16, "dim=16,bs=16"), (32, 32, "dim=32,bs=32")]:
    scheme = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=bs_val)
    cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
    model = TinyLinear(dim, dim)
    qmodel = quantize_model(model, cfg=cfg)
    calib_data = [torch.randn(dim) for _ in range(16)]
    calib_run_variable(qmodel, calib_data, batch_sizes=[3, 2, 5, 1])
    verify_buffers(qmodel, "linear", [dim], "bank", expected_scale_numel=1)
    # Forward: 1 bank means bank_axis dim must be divisible by bank_size. dim/bs=1 ✓
    verify_forward(qmodel, [(11, dim), (47, dim)], f"8.1 {label}")

# 8.2 — ratio=0.01 (minimum: 1 outlier per group)
print("\n8.2 ratio=0.01 (minimum), BANK with different bank_sizes")
for bs_val, dim, nb in [(4, 8, 2), (8, 16, 2), (16, 32, 2)]:
    scheme = make_scheme("int4", "bank", outlier_ratio=0.01, bank_size=bs_val)
    cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
    model = TinyLinear(dim, dim)
    qmodel = quantize_model(model, cfg=cfg)
    calib_data = [torch.randn(dim) for _ in range(8)]
    calib_run_variable(qmodel, calib_data, batch_sizes=[2, 3, 1])
    mask = qmodel.linear.get_buffer("_input_mask")
    # At least 1 True per bank
    check(mask.sum().item() >= 1, f"8.2 ratio=0.01 bs={bs_val}: ≥1 outlier total")
    verify_forward(qmodel, [(5, dim), (33, dim)], f"8.2 bs={bs_val}")

# 8.3 — ratio=0.99 (near full: almost all are outliers)
print("\n8.3 ratio=0.99 (near full), PER_TENSOR")
scheme = make_scheme("int4", "per_tensor", outlier_ratio=0.99)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(16, 16)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(16) for _ in range(10)]
calib_run_variable(qmodel, calib_data, batch_sizes=[2, 3, 1, 4])
mask = qmodel.linear.get_buffer("_input_mask")
check(mask.sum().item() >= 15, f"8.3 ratio=0.99: ≥15/16 outliers (got {mask.sum().item()})")
verify_forward(qmodel, [(7, 16), (51, 16)], "8.3")

# 8.4 — Small spatial: 1 element in a dimension
print("\n8.4 Small spatial: Linear 2→2 (2 elements total, bank=2)")
scheme = make_scheme("int4", "bank", outlier_ratio=0.5, bank_size=2)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(2, 2)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(2) for _ in range(20)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 7, 1, 5, 2])
verify_buffers(qmodel, "linear", [2], "bank", expected_scale_numel=1)
verify_forward(qmodel, [(13, 2), (61, 2), (127, 2)], "8.4")

# 8.5 — Conv2d with single-bank channel (C=bank_size)
print("\n8.5 Conv2d: C=4=bank_size (single channel bank)")
scheme_in = make_scheme("int4", "bank", outlier_ratio=0.25, bank_size=4, bank_axis=1)
scheme_w = make_scheme("int4", "bank", outlier_ratio=0.25, bank_size=4, bank_axis=0)
scheme_out = make_scheme("int4", "bank", outlier_ratio=0.25, bank_size=4, bank_axis=1)
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
model = TinyConv2d(4, 4, kernel=3, padding=1, H=8, W=8)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(4, 8, 8) for _ in range(12)]
calib_run_variable(qmodel, calib_data, batch_sizes=[2, 1, 3, 4])
verify_buffers(qmodel, "conv", [4, 8, 8], "bank", expected_scale_numel=1)
verify_forward(qmodel, [(9, 4, 8, 8), (43, 4, 8, 8), (79, 4, 8, 8)], "8.5")

# 8.6 — Conv1d with kernel=1 (no spatial reduction)
print("\n8.6 Conv1d: kernel=1, pad=0 (no spatial reduction)")
scheme_in = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
scheme_w = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=0)
scheme_out = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
model = TinyConv1d(8, 8, kernel=1, padding=0, stride=1, length=16)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(8, 16) for _ in range(20)]
calib_run_variable(qmodel, calib_data, batch_sizes=[4, 2, 3, 5])
verify_buffers(qmodel, "conv", [8, 16], "bank", expected_scale_numel=2)
verify_forward(qmodel, [(5, 8, 16), (37, 8, 16), (83, 8, 16)], "8.6")

# ═══════════════════════════════════════════════════════════════════
# §9  MIXED Linear + Conv models
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§9  Mixed Linear + Conv models")
print("=" * 60)

# 9.1 — Conv1d → Linear
print("\n9.1 Conv1d(4,8) → Flatten → Linear(80,16)  [BANK on channels]")

class ConvThenLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.linear = torch.nn.Linear(80, 16)  # 8 * 10 = 80
    def forward(self, x):
        x = self.conv(x)               # (B, 8, 10)
        x = x.flatten(1)               # (B, 80)
        return self.linear(x)

scheme_conv_in = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
scheme_conv_w = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=0)
scheme_conv_out = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
scheme_lin = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4)
per_layer = {
    "conv": OpQuantConfig(input=scheme_conv_in, weight=scheme_conv_w, output=scheme_conv_out),
    "linear": OpQuantConfig(input=scheme_lin, weight=scheme_lin, output=scheme_lin),
}
model = ConvThenLinear()
qmodel = quantize_model(model, cfg=per_layer)
calib_data = [torch.randn(4, 10) for _ in range(24)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 4, 2, 1, 5])

verify_buffers(qmodel, "conv", [4, 10], "bank", expected_scale_numel=1)
verify_buffers(qmodel, "linear", [80], "bank", expected_scale_numel=20)
verify_forward(qmodel, [(7, 4, 10), (29, 4, 10), (67, 4, 10)], "9.1")

# 9.2 — Conv2d → Conv2d (cascade)
print("\n9.2 Conv2d(4,8) → ReLU → Conv2d(8,4)  [BANK on channels]")

class CascadeConv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=3, padding=1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        return self.conv2(x)

scheme_c1_in = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=1)
scheme_c1_w = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=0)
scheme_c1_out = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=1)
scheme_c2_in = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=1)
scheme_c2_w = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=0)
scheme_c2_out = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=4, bank_axis=1)
per_layer = {
    "conv1": OpQuantConfig(input=scheme_c1_in, weight=scheme_c1_w, output=scheme_c1_out),
    "conv2": OpQuantConfig(input=scheme_c2_in, weight=scheme_c2_w, output=scheme_c2_out),
}
model = CascadeConv2d()
qmodel = quantize_model(model, cfg=per_layer)
calib_data = [torch.randn(4, 12, 12) for _ in range(24)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 5, 2, 1, 4])

verify_buffers(qmodel, "conv1", [4, 12, 12], "bank", expected_scale_numel=1)
verify_buffers(qmodel, "conv2", [8, 12, 12], "bank", expected_scale_numel=2)
verify_forward(qmodel, [(11, 4, 12, 12), (31, 4, 12, 12), (59, 4, 12, 12)], "9.2")

# 9.3 — Linear → Conv1d (reshaping)
print("\n9.3 Linear(64, 32) → Reshape → Conv1d(4, 8, k=3)  [mixed granularity]")

class LinearThenConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
        self.conv = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.linear(x)           # (B, 32)
        x = x.view(-1, 4, 8)         # (B, 4, 8)
        return self.conv(x)

scheme_lin = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4)
scheme_c_in = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
scheme_c_w = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=0)
scheme_c_out = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
per_layer = {
    "linear": OpQuantConfig(input=scheme_lin, weight=scheme_lin, output=scheme_lin),
    "conv": OpQuantConfig(input=scheme_c_in, weight=scheme_c_w, output=scheme_c_out),
}
model = LinearThenConv()
qmodel = quantize_model(model, cfg=per_layer)
calib_data = [torch.randn(64) for _ in range(30)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 7, 2, 5, 1])

verify_buffers(qmodel, "linear", [64], "bank", expected_scale_numel=16)
verify_buffers(qmodel, "conv", [4, 8], "bank", expected_scale_numel=1)
verify_forward(qmodel, [(7, 64), (31, 64), (73, 64)], "9.3")

# ═══════════════════════════════════════════════════════════════════
# §10  DETERMINISM: different batch schedules → same mask
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§10  Determinism: different batch schedules → same mask")
print("=" * 60)

# 10.1 — Same total data, different batch splits → identical mask
print("\n10.1 Same data, 3 different batch schedules → identical masks (Linear BANK)")
torch.manual_seed(1234)
all_data = [torch.randn(12) for _ in range(60)]

scheme = make_scheme("int4", "bank", outlier_ratio=0.25, bank_size=4)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)

masks = []
for schedule_name, batch_sizes in [
    ("fixed=4", [4]),
    ("fixed=3", [3]),
    ("variable [3,7,2]", [3, 7, 2]),
]:
    torch.manual_seed(9999)  # reset model init
    model = TinyLinear(12, 8)
    qmodel = quantize_model(model, cfg=cfg)
    # Deep copy the data to avoid any mutation side effects
    data_copy = [t.clone() for t in all_data]
    if len(batch_sizes) == 1:
        calib_run_fixed(qmodel, data_copy, batch_sizes[0])
    else:
        calib_run_variable(qmodel, data_copy, batch_sizes)
    masks.append(qmodel.linear.get_buffer("_input_mask"))

# All masks must be identical (same data, only batching differs)
for i in range(1, len(masks)):
    check(torch.equal(masks[0], masks[i]),
          f"10.1 mask fixed=4 == {['fixed=3','variable'][i-1]}")
    check_mask_dim0(masks[i], f"10.1 mask {i}")

# 10.2 — Same for Conv2d PER_CHANNEL
print("\n10.2 Same data, 2 different batch schedules → identical masks (Conv2d PER_CHANNEL)")
torch.manual_seed(5678)
all_data_2d = [torch.randn(6, 10, 10) for _ in range(36)]

scheme_in = make_scheme("int4", "per_channel", outlier_ratio=0.15, channel_axis=1)
scheme_w = make_scheme("int4", "per_channel", outlier_ratio=0.15, channel_axis=0)
scheme_out = make_scheme("int4", "per_channel", outlier_ratio=0.15, channel_axis=1)
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)

masks_2d = []
for schedule_name, batch_sizes in [
    ("fixed=6", [6]),
    ("variable [2,5,3,1,7]", [2, 5, 3, 1, 7]),
]:
    torch.manual_seed(8888)
    model = TinyConv2d(6, 8, kernel=3, padding=1, H=10, W=10)
    qmodel = quantize_model(model, cfg=cfg)
    data_copy = [t.clone() for t in all_data_2d]
    if len(batch_sizes) == 1:
        calib_run_fixed(qmodel, data_copy, batch_sizes[0])
    else:
        calib_run_variable(qmodel, data_copy, batch_sizes)
    masks_2d.append(qmodel.conv.get_buffer("_input_mask"))

check(torch.equal(masks_2d[0], masks_2d[1]),
      "10.2 mask fixed=6 == variable [2,5,3,1,7]")
for i, m in enumerate(masks_2d):
    check_mask_dim0(m, f"10.2 mask {i}")

# 10.3 — Determinism for Conv1d BANK
print("\n10.3 Same data, 3 different batch schedules → identical masks (Conv1d BANK)")
torch.manual_seed(4321)
all_data_1d = [torch.randn(8, 12) for _ in range(40)]

scheme_in = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
scheme_w = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=0)
scheme_out = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)

masks_1d = []
for batch_sizes in [[2], [5], [1, 3, 7, 2]]:
    torch.manual_seed(7777)
    model = TinyConv1d(8, 4, kernel=3, padding=1, length=12)
    qmodel = quantize_model(model, cfg=cfg)
    data_copy = [t.clone() for t in all_data_1d]
    if len(batch_sizes) == 1:
        calib_run_fixed(qmodel, data_copy, batch_sizes[0])
    else:
        calib_run_variable(qmodel, data_copy, batch_sizes)
    masks_1d.append(qmodel.conv.get_buffer("_input_mask"))

for i in range(1, len(masks_1d)):
    check(torch.equal(masks_1d[0], masks_1d[i]),
          f"10.3 mask bs={[2,5,[1,3,7,2]][0]} == bs={[2,5,[1,3,7,2]][i]}")
    check_mask_dim0(masks_1d[i], f"10.3 mask {i}")

# ═══════════════════════════════════════════════════════════════════
# §11  MULTI-EVAL: same model, many different eval batch sizes
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§11  Multi-eval: many different eval batch sizes")
print("=" * 60)

# 11.1 — Linear BANK: calibrate once, eval with 10 different batch sizes
print("\n11.1 Linear BANK: 10 eval batch sizes (1 to 200)")
scheme = make_scheme("int4", "bank", outlier_ratio=0.25, bank_size=4)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(16, 8)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(16) for _ in range(32)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 5, 2, 7, 1])

eval_batches = [1, 2, 3, 7, 13, 29, 47, 73, 131, 200]
for b in eval_batches:
    x = torch.randn(b, 16)
    with torch.no_grad():
        out = qmodel(x)
    check_shape(out, (b, 8), f"11.1 eval B={b}")
    check_no_nan_inf(out, f"11.1 eval B={b}")

# 11.2 — Conv2d BANK: 8 eval batch sizes
print("\n11.2 Conv2d BANK: 8 eval batch sizes (1 to 150)")
scheme_in = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
scheme_w = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=0)
scheme_out = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4, bank_axis=1)
cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
model = TinyConv2d(8, 4, kernel=3, padding=1, H=10, W=10)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(8, 10, 10) for _ in range(24)]
calib_run_variable(qmodel, calib_data, batch_sizes=[2, 4, 1, 3])

for b in [1, 3, 7, 19, 41, 67, 103, 150]:
    x = torch.randn(b, 8, 10, 10)
    with torch.no_grad():
        out = qmodel(x)
    check_shape(out, (b, 4, 10, 10), f"11.2 eval B={b}")
    check_no_nan_inf(out, f"11.2 eval B={b}")

# ═══════════════════════════════════════════════════════════════════
# §12  ALL GRANULARITY MODES × ALL OP TYPES (combinatorial)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§12  Combinatorial: all granularity modes × op types")
print("=" * 60)

op_configs = [
    ("Linear", TinyLinear, {"in_f": 16, "out_f": 12}, (16,), (16,)),
    ("Conv1d", TinyConv1d, {"in_c": 8, "out_c": 8, "kernel": 3, "padding": 1, "length": 12},
     (8, 12), (8, 12)),
    ("Conv2d", TinyConv2d, {"in_c": 8, "out_c": 8, "kernel": 3, "padding": 1, "H": 10, "W": 10},
     (8, 10, 10), (8, 10, 10)),
]

gran_configs = [
    ("per_tensor", {"gran_mode": "per_tensor", "outlier_ratio": 0.25},
     None),  # no expected numel (varies)
    ("per_channel", {"gran_mode": "per_channel", "outlier_ratio": 0.2, "channel_axis": 1},
     None),
    ("bank", {"gran_mode": "bank", "outlier_ratio": 0.18, "bank_size": 4}, None),
]

for op_name, op_cls, op_kwargs, input_spatial, full_spatial in op_configs:
    for gran_name, gran_kwargs, _ in gran_configs:
        print(f"\n12. {op_name} × {gran_name}")

        # Build scheme differently for Conv (separate input/weight/output axes)
        if op_name == "Linear":
            scheme = make_scheme("int4", **gran_kwargs)
            cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
            layer_attr = "linear"
        else:
            # Conv: input/output bank/channel on spatial dim 1 (C),
            # weight bank/channel on dim 0 (out_channels)
            mode = gran_kwargs.get("gran_mode", "")
            is_per_ch = "channel_axis" in gran_kwargs
            is_bank = "bank_axis" in gran_kwargs or mode == "bank"
            extra_in, extra_out, extra_w = {}, {}, {}
            if is_per_ch:
                extra_in["channel_axis"] = 1
                extra_w["channel_axis"] = 0
                extra_out["channel_axis"] = 1
            if is_bank:
                extra_in["bank_axis"] = 1
                extra_w["bank_axis"] = 0
                extra_out["bank_axis"] = 1
            scheme_in = make_scheme("int4", **{**gran_kwargs, **extra_in})
            scheme_w = make_scheme("int4", **{**gran_kwargs, **extra_w})
            scheme_out = make_scheme("int4", **{**gran_kwargs, **extra_out})
            cfg = OpQuantConfig(input=scheme_in, weight=scheme_w, output=scheme_out)
            layer_attr = "conv"

        model = op_cls(**op_kwargs)
        qmodel = quantize_model(model, cfg=cfg)
        calib_data = [torch.randn(*input_spatial) for _ in range(24)]
        calib_run_variable(qmodel, calib_data, batch_sizes=[3, 4, 2, 5, 1])

        layer = qmodel
        for part in layer_attr.split("."):
            layer = getattr(layer, part)
        mask = layer.get_buffer("_input_mask")
        check_mask_dim0(mask, f"12 {op_name}×{gran_name} mask")

        eval_spatial = (13,) + full_spatial
        x = torch.randn(*eval_spatial)
        with torch.no_grad():
            out = qmodel(x)
        check_no_nan_inf(out, f"12 {op_name}×{gran_name} forward")

# ═══════════════════════════════════════════════════════════════════
# §13  TRANSFORM interaction — Hadamard + sparse
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§13  Transform + sparse interaction")
print("=" * 60)

# 13.1 — HadamardTransform + BANK sparse on Linear
print("\n13.1 HadamardTransform + BANK sparse (Linear)")
from src.transform.hadamard import HadamardTransform

scheme = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=4)
scheme = replace(scheme, transform=HadamardTransform())
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(16, 16)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(16) for _ in range(20)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 5, 2])
# Hadamard operates on the pre-quantization tensor; mask shape should still be valid
mask = qmodel.linear.get_buffer("_input_mask")
check_mask_dim0(mask, "13.1 hadamard input_mask")
verify_forward(qmodel, [(7, 16), (31, 16), (59, 16)], "13.1")

# ═══════════════════════════════════════════════════════════════════
# §14  STRESS: deep wide model, every layer different shape
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§14  Stress: deep wide model, varied layer shapes")
print("=" * 60)


class VarShapeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(64, 128)
        self.l1 = torch.nn.Linear(128, 96)
        self.l2 = torch.nn.Linear(96, 48)
        self.l3 = torch.nn.Linear(48, 32)
        self.l4 = torch.nn.Linear(32, 24)
        self.l5 = torch.nn.Linear(24, 16)
        self.l6 = torch.nn.Linear(16, 8)

    def forward(self, x):
        for layer in [self.l0, self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            x = torch.relu(layer(x))
        return x


print("\n14.1 7-layer varied MLP, BANK, variable calib batches")
scheme = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=4)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = VarShapeMLP()
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(64) for _ in range(60)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 5, 2, 7, 1, 4, 6])

expected_nbs = [16, 32, 24, 12, 8, 6, 4, 2]
for i, (name, nb) in enumerate(zip(
    ["l0", "l1", "l2", "l3", "l4", "l5", "l6"],
    [16, 32, 24, 12, 8, 6, 4]  # in_features//4
)):
    layer = getattr(qmodel, name)
    mask = layer.get_buffer("_input_mask")
    check_mask_dim0(mask, f"14.1 {name} mask (shape {tuple(mask.shape)})")
    scale = layer.get_buffer("_input_scale")
    check_eq(scale.numel(), nb, f"14.1 {name} scale numel={nb}")

eval_x = torch.randn(41, 64)
with torch.no_grad():
    out = qmodel(eval_x)
check_shape(out, (41, 8), "14.1 output shape")
check_no_nan_inf(out, "14.1 output")

# ═══════════════════════════════════════════════════════════════════
# §15  BANK SIZE DIVISIBILITY boundaries
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("§15  Bank size divisibility boundaries")
print("=" * 60)

# 15.1 — bank_size=2 (smallest practical)
print("\n15.1 bank_size=2, dim=10 (10%2=0 ✓)")
scheme = make_scheme("int4", "bank", outlier_ratio=0.3, bank_size=2)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(10, 6)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(10) for _ in range(20)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 7, 2])
verify_buffers(qmodel, "linear", [10], "bank", expected_scale_numel=5)
verify_forward(qmodel, [(7, 10), (43, 10)], "15.1")

# 15.2 — bank_size=dim (single bank) with outlier_ratio=0.5
print("\n15.2 bank_size=dim=12, single bank, ratio=0.5")
scheme = make_scheme("int4", "bank", outlier_ratio=0.5, bank_size=12)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(12, 12)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(12) for _ in range(24)]
calib_run_variable(qmodel, calib_data, batch_sizes=[4, 2, 3, 5, 1])
verify_buffers(qmodel, "linear", [12], "bank", expected_scale_numel=1)
verify_forward(qmodel, [(3, 12), (77, 12)], "15.2")

# 15.3 — bank_size=3, dim=15 (odd bank size)
print("\n15.3 bank_size=3, dim=15 (15%3=0, 5 banks)")
scheme = make_scheme("int4", "bank", outlier_ratio=0.2, bank_size=3)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(15, 9)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(15) for _ in range(18)]
calib_run_variable(qmodel, calib_data, batch_sizes=[3, 2, 4])
verify_buffers(qmodel, "linear", [15], "bank", expected_scale_numel=5)
verify_forward(qmodel, [(11, 15), (53, 15)], "15.3")

# 15.4 — bank_size=5, dim=25 (prime-related)
print("\n15.4 bank_size=5, dim=25 (5 banks of size 5)")
scheme = make_scheme("int4", "bank", outlier_ratio=0.15, bank_size=5)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(25, 15)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(25) for _ in range(20)]
calib_run_variable(qmodel, calib_data, batch_sizes=[2, 3, 5, 1])
verify_buffers(qmodel, "linear", [25], "bank", expected_scale_numel=5)
verify_forward(qmodel, [(9, 25), (49, 25)], "15.4")

# 15.5 — bank_size=7, dim=49
print("\n15.5 bank_size=7, dim=49 (7 banks of size 7)")
scheme = make_scheme("int4", "bank", outlier_ratio=0.1, bank_size=7)
cfg = OpQuantConfig(input=scheme, weight=scheme, output=scheme)
model = TinyLinear(49, 21)
qmodel = quantize_model(model, cfg=cfg)
calib_data = [torch.randn(49) for _ in range(14)]
calib_run_variable(qmodel, calib_data, batch_sizes=[2, 3, 1, 4])
verify_buffers(qmodel, "linear", [49], "bank", expected_scale_numel=7)
verify_forward(qmodel, [(5, 49), (37, 49)], "15.5")

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("ALL CHECKS PASSED")
else:
    print(f"{FAIL} CHECKS FAILED")
    sys.exit(1)
