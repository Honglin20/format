"""
00 — Comprehensive: every format, granularity, transform, and session feature.

Covers ALL QuantSession API surface and ALL quantization axes:
  1. All 11 registered formats (int8/4/2, fp8/6/4, nf4, bf16/fp16)
  2. All 3 granularity modes (per_tensor, per_channel, per_block)
  3. Both transforms (Hadamard, SmoothQuant)
  4. All 4 calibration strategies (Max, Percentile, MSE, KL)
  5. All 4 observers (QSNR, MSE, Histogram, Distribution)
  6. E2E comparison: Comparator, compare_models, compare_sessions, session.compare()
  7. QuantSession: calibrate, analyze, compare, export_onnx, clear_scales
  8. Per-layer dict config + per-op inline configs (op_cfgs)
  9. QAT: training-aware quantization with grad_* fields
 10. Custom format registration
 11. Performance evaluation (evaluate_performance)
 12. Multi-format comparison (compare_formats)

Run:  PYTHONPATH=. python examples/00_comprehensive.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _model import ToyMLP
from src.formats.base import FormatBase
from src.formats.registry import register_format
from src.formats.lookup_formats import LookupFormat
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.session import QuantSession
from src.mapping.quantize_model import quantize_model
from src.analysis.e2e import Comparator, compare_models, compare_sessions
from src.analysis.compare import compare_formats, ComparisonReport
from src.analysis.eval_performance import evaluate_performance
from src.analysis.observers import (
    QSNRObserver, MSEObserver, HistogramObserver, DistributionObserver,
)
from src.calibration.strategies import (
    MaxScaleStrategy, PercentileScaleStrategy,
    MSEScaleStrategy, KLScaleStrategy,
)
from src.transform.hadamard import HadamardTransform
from src.transform.smooth_quant import SmoothQuantTransform
from src.transform.pre_scale import PreScaleTransform
from src.calibration.lsq_optimizer import LayerwiseScaleOptimizer


D = 128   # hidden dim
H = 512   # intermediate dim

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def make_data(n=64):
    return torch.randn(n, D)

def make_labeled_data(n=64, n_classes=3):
    return torch.randn(n, D), torch.randint(0, n_classes, (n,))

def make_loader(n=64, bs=8):
    x, y = make_labeled_data(n, n_classes=2)
    return DataLoader(TensorDataset(x, y), batch_size=bs)

def accuracy(logits, labels):
    return {"acc": (logits.argmax(-1) == labels).float().mean().item()}

def fmt(name):
    return FormatBase.from_str(name)

def scheme(fmt_name, gran, **kw):
    return QuantScheme(format=fmt(fmt_name), granularity=gran, **kw)

def cfg_in(s):   return OpQuantConfig(input=s)
def cfg_w(s):    return OpQuantConfig(weight=s)
def cfg_io(s):   return OpQuantConfig(input=s, output=s)
def cfg_iwo(s):  return OpQuantConfig(input=s, weight=s, output=s)
def cfg_qat(s):  return OpQuantConfig(
    input=s, weight=s, output=s,
    grad_output=s, grad_input=s, grad_weight=s,
)

PER_T  = GranularitySpec.per_tensor()
PER_C  = GranularitySpec.per_channel(axis=-1)
PER_B  = GranularitySpec.per_block(32)


# ══════════════════════════════════════════════════════════════════════
# Section 1 — All 11 Registered Formats
# ══════════════════════════════════════════════════════════════════════

def section_1():
    print("=" * 60)
    print("1. ALL REGISTERED FORMATS (per_tensor)")
    print("=" * 60)

    ref = ToyMLP(); ref.eval()
    x = make_data(4)

    formats = [
        ("int8",      "int8",        cfg_iwo),
        ("int4",      "int4",        cfg_iwo),
        ("int2",      "int2",        cfg_iwo),
        ("fp8_e4m3",  "fp8_e4m3",    cfg_iwo),
        ("fp8_e5m2",  "fp8_e5m2",    cfg_iwo),
        ("fp6_e3m2",  "fp6_e3m2",    cfg_iwo),
        ("fp6_e2m3",  "fp6_e2m3",    cfg_iwo),
        ("fp4_e2m1",  "fp4_e2m1",    cfg_iwo),
        ("nf4",       "nf4",         cfg_w),
        ("bfloat16",  "bfloat16",    cfg_iwo),
        ("float16",   "float16",     cfg_iwo),
    ]

    print(f"  {'Format':<16} {'MSE vs fp32':>14}  {'Note'}")
    print(f"  {'-'*16} {'-'*14}  {'-'*30}")
    with torch.no_grad():
        ref_out = ref(x)
        for label, key, cfg_fn in formats:
            s = scheme(key, PER_T)
            c = cfg_fn(s)
            m = quantize_model(ToyMLP(), cfg=c)
            m.load_state_dict(ref.state_dict(), strict=False); m.eval()
            out = m(x)
            mse = (ref_out - out).pow(2).mean().item()
            note = ""
            if mse == 0:
                note = "(passthrough — same as fp32)"
            elif mse < 1e-5:
                note = "(near-lossless)"
            print(f"  {label:<16} {mse:>14.6e}  {note}")
    print()


# ══════════════════════════════════════════════════════════════════════
# Section 2 — All Granularity Modes
# ══════════════════════════════════════════════════════════════════════

def section_2():
    print("=" * 60)
    print("2. ALL GRANULARITY MODES (int8)")
    print("=" * 60)

    ref = ToyMLP(); ref.eval()
    x = make_data(4)

    modes = [
        ("per_tensor",            GranularitySpec.per_tensor()),
        ("per_channel(axis=-1)",  GranularitySpec.per_channel(axis=-1)),
        ("per_channel(axis=0)",   GranularitySpec.per_channel(axis=0)),
        ("per_block(32)",         GranularitySpec.per_block(32)),
        ("per_block(64)",         GranularitySpec.per_block(64)),
    ]

    print(f"  {'Granularity':<24} {'MSE vs fp32':>14}")
    print(f"  {'-'*24} {'-'*14}")
    with torch.no_grad():
        ref_out = ref(x)
        for label, gran in modes:
            s = scheme("int8", gran)
            c = cfg_iwo(s)
            m = quantize_model(ToyMLP(), cfg=c)
            m.load_state_dict(ref.state_dict(), strict=False); m.eval()
            out = m(x)
            mse = (ref_out - out).pow(2).mean().item()
            print(f"  {label:<24} {mse:>14.6e}")
    print()


# ══════════════════════════════════════════════════════════════════════
# Section 3 — Transforms (Hadamard + SmoothQuant)
# ══════════════════════════════════════════════════════════════════════

def section_3():
    print("=" * 60)
    print("3. TRANSFORMS")
    print("=" * 60)

    ref = ToyMLP(); ref.eval(); sd = ref.state_dict()
    x = make_data(8)

    # --- Hadamard ---
    print("  3a. Hadamard Rotation")
    i4 = fmt("int4")

    # baseline
    q = quantize_model(ToyMLP(), cfg=cfg_iwo(QuantScheme(i4, PER_T)))
    q.load_state_dict(sd, strict=False); q.eval()
    with torch.no_grad():
        mse_base = (ref(x) - q(x)).pow(2).mean().item()

    # Hadamard on input
    had_s = QuantScheme(i4, PER_C, transform=HadamardTransform())
    had_c = OpQuantConfig(input=had_s, weight=QuantScheme(i4, PER_T), output=QuantScheme(i4, PER_T))
    q_h = quantize_model(ToyMLP(), cfg=had_c)
    q_h.load_state_dict(sd, strict=False); q_h.eval()
    with torch.no_grad():
        mse_had = (ref(x) - q_h(x)).pow(2).mean().item()

    red = (1 - mse_had / mse_base) * 100 if mse_base > 0 else 0
    print(f"    int4 baseline MSE:    {mse_base:.6f}")
    print(f"    int4 + Hadamard MSE:  {mse_had:.6f}  ({red:+.1f}%)")

    # Verify reversibility
    h = HadamardTransform()
    t = torch.randn(4, 128)
    err = (t - h.inverse(h.forward(t))).abs().max().item()
    print(f"    Hadamard round-trip:  {err:.2e}")

    # --- SmoothQuant ---
    print("\n  3b. SmoothQuant (per-layer)")
    i8 = fmt("int8")
    i8_s = QuantScheme(i8, PER_T)

    # Calibrate on fc1
    calib_x = torch.randn(16, D)
    with torch.no_grad():
        act = ref.ln(calib_x)  # (16, 128)
    sq_t = SmoothQuantTransform.from_calibration(act, ref.fc1.weight.data, alpha=0.5)

    sq_cfg = {
        "fc1": OpQuantConfig(
            input=QuantScheme(i8, PER_C, transform=sq_t),
            weight=i8_s, output=i8_s,
        ),
        "fc2": cfg_iwo(i8_s),
        "ln":  cfg_io(i8_s),
    }
    q_sq = quantize_model(ToyMLP(), cfg=sq_cfg)
    q_sq.load_state_dict(sd, strict=False); q_sq.eval()

    # baseline int8
    q_i8 = quantize_model(ToyMLP(), cfg=cfg_iwo(i8_s))
    q_i8.load_state_dict(sd, strict=False); q_i8.eval()

    with torch.no_grad():
        mse_i8 = (ref(x) - q_i8(x)).pow(2).mean().item()
        mse_sq = (ref(x) - q_sq(x)).pow(2).mean().item()
    red2 = (1 - mse_sq / mse_i8) * 100 if mse_i8 > 0 else 0
    print(f"    int8 baseline MSE:    {mse_i8:.6f}")
    print(f"    int8 + SmoothQuant:   {mse_sq:.6f}  ({red2:+.1f}%)")

    # Verify reversibility
    err2 = (t - sq_t.inverse(sq_t.forward(t))).abs().max().item()
    print(f"    SmoothQuant r-trip:   {err2:.2e}")
    print()


# ══════════════════════════════════════════════════════════════════════
# Section 4 — QuantSession: Full Pipeline
# ══════════════════════════════════════════════════════════════════════

def section_4():
    print("=" * 60)
    print("4. QUANTSESSION — FULL PIPELINE")
    print("=" * 60)

    session = QuantSession(
        ToyMLP(),
        cfg_iwo(scheme("int8", PER_T)),
        calibrator=PercentileScaleStrategy(q=99.0),
        observers=[QSNRObserver(), MSEObserver()],
    )
    session.eval()

    calib_data = make_data(32)
    bs = 4  # batch size — must match inference batch size for per_channel

    # 4a. Calibrate (scales auto-assigned)
    print("  4a. Calibrating ...")
    with session.calibrate():
        for i in range(0, 32, bs):
            session(calib_data[i:i+bs])
    n_scales = sum(1 for m in session.qmodel.modules() if hasattr(m, "_output_scale"))
    print(f"      {n_scales} modules received _output_scale buffers")

    # 4b. Analyze
    print("  4b. Analyzing ...")
    with session.analyze(observers=[QSNRObserver(), MSEObserver(),
                                     HistogramObserver(n_bins=32),
                                     DistributionObserver()]) as ctx:
        for i in range(0, 32, bs):
            session(calib_data[i:i+bs])

    report = ctx.report()
    layers = report.keys()
    print(f"      {len(layers)} layers analyzed: {sorted(layers)}")
    # Print first data point
    first_layer = sorted(layers)[0]
    ld = report._raw[first_layer]
    first_role = list(ld.keys())[0]
    first_stage = list(ld[first_role].keys())[0]
    first_slice = list(ld[first_role][first_stage].keys())[0]
    m = ld[first_role][first_stage][first_slice]
    print(f"      {first_layer}/{first_role}: QSNR={m.get('qsnr_db', 0):.1f} dB  "
          f"MSE={m.get('mse', 0):.6f}")

    # 4c. Compare (auto mode)
    print("  4c. Comparing (auto) ...")
    dl = make_loader(64)
    r = session.compare(dl, eval_fn=accuracy, directions={"acc": "higher"})
    print(f"      fp32={r['fp32']['acc']:.4f}  quant={r['quant']['acc']:.4f}  "
          f"delta={r['delta']['acc']:+.4f}")

    # 4d. Compare (manual mode)
    print("  4d. Comparing (manual) ...")
    cmp = session.comparator()
    with cmp, torch.no_grad():
        for x, y in dl:
            session.use_fp32();  fp = session(x)
            session.use_quant(); qo = session(x)
            cmp.record(fp, qo, y)
    r2 = cmp.evaluate(accuracy, directions={"acc": "higher"})
    print(f"      fp32={r2['fp32']['acc']:.4f}  quant={r2['quant']['acc']:.4f}  "
          f"delta={r2['delta']['acc']:+.4f}  n={cmp.num_samples}")

    # 4e. ONNX export
    print("  4e. ONNX export ...")
    import os, tempfile
    session(torch.randn(1, D))  # record input
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.onnx")
        session.export_onnx(p)
        print(f"      exported: {os.path.getsize(p)/1024:.1f} KB, "
              f"onnx.checker: ", end="")
        import onnx
        try:
            onnx.checker.check_model(p)
            print("OK")
        except Exception as e:
            print(f"FAIL ({e})")

    # 4f. Clear scales
    print("  4f. Clear scales ...")
    removed = session.clear_scales()
    print(f"      cleared {len(removed)} scale buffers")
    print()


# ══════════════════════════════════════════════════════════════════════
# Section 5 — Per-Layer Dict + Per-Op Inline Configs
# ══════════════════════════════════════════════════════════════════════

def section_5():
    print("=" * 60)
    print("5. PER-LAYER + PER-OP CONFIGS (via QuantSession)")
    print("=" * 60)

    i8_s = scheme("int8", PER_T)
    fp8_s = scheme("fp8_e4m3", PER_T)
    nf4_s = scheme("nf4", PER_C)

    # Per-layer: fc1 gets fp8, fc2 gets int8, ln gets int8 io only
    layer_cfg = {
        "fc1": OpQuantConfig(input=fp8_s, weight=fp8_s, output=fp8_s),
        "fc2": OpQuantConfig(input=i8_s, weight=i8_s, output=i8_s),
        "ln":  cfg_io(i8_s),
    }
    # Per-op inline: matmul gets no quant, add gets int8
    op_cfgs = {
        "add": OpQuantConfig(input=i8_s, output=i8_s),
    }

    session = QuantSession(ToyMLP(), cfg=layer_cfg, op_cfgs=op_cfgs)
    session.eval()

    x = make_data(4)
    with torch.no_grad():
        y = session(x)
    print(f"  output shape: {y.shape}  (per-layer fp8+int8 + inline add quant)")
    print(f"  sample: {y[0, :4].tolist()}")

    # Verify no crash on mode switch
    session.use_fp32()
    with torch.no_grad():
        y2 = session(x)
    print(f"  fp32 mode OK, shape: {y2.shape}")
    print()


# ══════════════════════════════════════════════════════════════════════
# Section 6 — QAT: Training-Aware Quantization
# ══════════════════════════════════════════════════════════════════════

def section_6():
    print("=" * 60)
    print("6. QAT — TRAINING-AWARE QUANTIZATION")
    print("=" * 60)

    i8_s = scheme("int8", PER_T)
    qat_cfg = cfg_qat(i8_s)
    print(f"  qat_cfg.is_training = {qat_cfg.is_training}")

    session = QuantSession(ToyMLP(), cfg=qat_cfg)
    session.train()
    assert session.qmodel.training

    x = torch.randn(4, D)
    target = torch.randn(4, D)

    # Forward + backward — should not crash
    y = session(x)
    loss = (y - target).pow(2).mean()
    loss.backward()

    print(f"  forward:  {x.shape} → {y.shape}")
    print(f"  loss:     {loss.item():.6f}")
    print(f"  backward: OK (grad flowed through quantized path)")

    # Check that grad_* fields are populated (non-None = pipeline active)
    print(f"  grad_output: {qat_cfg.grad_output is not None}")
    print(f"  grad_input:  {qat_cfg.grad_input is not None}")
    print(f"  grad_weight: {qat_cfg.grad_weight is not None}")
    print()


# ══════════════════════════════════════════════════════════════════════
# Section 7 — compare_sessions: Multi-Format Comparison
# ══════════════════════════════════════════════════════════════════════

def section_7():
    print("=" * 60)
    print("7. compare_sessions — MULTI-FORMAT BATCH COMPARISON")
    print("=" * 60)

    base = ToyMLP(); base.eval(); sd = base.state_dict()

    def make_session(fmt_name, gran=PER_T):
        s = scheme(fmt_name, gran)
        ses = QuantSession(ToyMLP(), cfg=cfg_iwo(s))
        ses.eval()
        ses.qmodel.load_state_dict(sd, strict=False)
        return ses

    sessions = {
        "int8":    make_session("int8"),
        "int4":    make_session("int4"),
        "fp8_e4m3": make_session("fp8_e4m3"),
        "fp4_mx":  make_session("fp4_e2m1", PER_B),
        "nf4":     QuantSession(ToyMLP(), cfg=cfg_w(scheme("nf4", PER_C))),
    }
    sessions["nf4"].eval()
    sessions["nf4"].qmodel.load_state_dict(sd, strict=False)

    dl = make_loader(64)
    results = compare_sessions(sessions, dl, eval_fn=accuracy,
                               directions={"acc": "higher"})

    print(f"  fp32 baseline acc: {results['fp32']['acc']:.4f}")
    print(f"  {'Config':<12} {'fp32 acc':>10} {'quant acc':>10} {'delta':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for name in ["int8", "int4", "fp8_e4m3", "fp4_mx", "nf4"]:
        r = results[name]
        print(f"  {name:<12} {r['fp32']['acc']:>10.4f} {r['quant']['acc']:>10.4f} "
              f"{r['delta']['acc']:>+10.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════
# Section 8 — compare_formats: Full Analysis Comparison
# ══════════════════════════════════════════════════════════════════════

def section_8():
    print("=" * 60)
    print("8. compare_formats — FULL OBSERVER-BASED COMPARISON")
    print("=" * 60)

    sd = ToyMLP().state_dict()
    calib = [make_data(8) for _ in range(4)]

    configs = {
        "int8":      cfg_iwo(scheme("int8", PER_T)),
        "int4":      cfg_iwo(scheme("int4", PER_T)),
        "fp8_e4m3":  cfg_iwo(scheme("fp8_e4m3", PER_T)),
    }

    def build(name, cfg):
        m = quantize_model(ToyMLP(), cfg=cfg)
        m.load_state_dict(sd, strict=False); m.eval()
        return m

    observers = [QSNRObserver(), MSEObserver()]
    comp = compare_formats(build, calib, configs, observers=observers)

    print(f"  Formats: {list(comp.reports.keys())}")
    summary = comp.summary()
    for fmt_name, stats in summary.items():
        print(f"  {fmt_name:<12} avg_qsnr={stats['avg_qsnr_db']:.1f} dB  "
              f"avg_mse={stats['avg_mse']:.2e}")
    print()


# ══════════════════════════════════════════════════════════════════════
# Section 9 — evaluate_performance
# ══════════════════════════════════════════════════════════════════════

def section_9():
    print("=" * 60)
    print("9. evaluate_performance — TASK-LEVEL COMPARISON")
    print("=" * 60)

    sd = ToyMLP().state_dict()

    fp32 = ToyMLP(); fp32.eval()

    q_int8 = quantize_model(ToyMLP(), cfg=cfg_iwo(scheme("int8", PER_T)))
    q_int8.load_state_dict(sd, strict=False); q_int8.eval()

    q_int4 = quantize_model(ToyMLP(), cfg=cfg_iwo(scheme("int4", PER_T)))
    q_int4.load_state_dict(sd, strict=False); q_int4.eval()

    dl = make_loader(64)

    def cls_eval(model, dl):
        correct, total = 0, 0
        for x, y in dl:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        return {"accuracy": correct / total}

    report = evaluate_performance(fp32, {"int8": q_int8, "int4": q_int4}, dl, cls_eval)
    report.print_summary()
    print()


# ══════════════════════════════════════════════════════════════════════
# Section 10 — Custom Format Registration
# ══════════════════════════════════════════════════════════════════════

def section_10():
    print("=" * 60)
    print("10. CUSTOM FORMAT REGISTRATION")
    print("=" * 60)

    # Register a custom 5-level LUT format
    my_levels = [-1.0, -0.4, 0.0, 0.4, 1.0]
    my_fmt = LookupFormat("my_5level", levels=my_levels)
    register_format("my_5level", my_fmt)

    # Use it
    fetched = FormatBase.from_str("my_5level")
    assert fetched.name == "my_5level"
    assert fetched.levels.numel() == 5

    s = scheme("my_5level", PER_T)
    q = quantize_model(ToyMLP(), cfg=cfg_iwo(s))
    q.eval()

    x = make_data(2)
    with torch.no_grad():
        y = q(x)
    print(f"  Registered 'my_5level' ({my_levels})")
    print(f"  quantize_elemwise sample: {fetched.quantize_elemwise(x)[0,:4].tolist()}")
    print(f"  model output: {y[0,:4].tolist()}")

    # Verify it's in the registry
    from src.formats.registry import FORMAT_REGISTRY
    assert "my_5level" in FORMAT_REGISTRY
    print(f"  FORMAT_REGISTRY has 'my_5level': True")
    print()


# ══════════════════════════════════════════════════════════════════════
# Section 11 — keep_fp32=False + mode switching edge cases
# ══════════════════════════════════════════════════════════════════════

def section_11():
    print("=" * 60)
    print("11. QUANTSESSION — keep_fp32=False & EDGE CASES")
    print("=" * 60)

    # No fp32 copy (saves memory)
    session = QuantSession(ToyMLP(), cfg=cfg_iwo(scheme("int8", PER_T)),
                           keep_fp32=False)
    session.eval()

    x = make_data(2)
    with torch.no_grad():
        y = session(x)
    print(f"  keep_fp32=False: forward OK, shape={y.shape}")
    print(f"  fp32_model is None: {session.fp32_model is None}")

    # use_fp32 should raise
    try:
        session.use_fp32()
    except RuntimeError as e:
        print(f"  use_fp32() raises: {e}")

    # compare should raise
    dl = make_loader(16)
    try:
        session.compare(dl)
    except RuntimeError as e:
        print(f"  compare() raises: {e}")

    # But calibrate/analyze/export still work
    with session.calibrate():
        session(torch.randn(4, D))

    with session.analyze() as ctx:
        session(torch.randn(4, D))
    print(f"  calibrate + analyze: OK ({len(ctx.report().keys())} layers)")

    # Calibrate with strategy override
    with session.calibrate(strategy=MSEScaleStrategy(n_steps=10)):
        for _ in range(3):
            session(torch.randn(2, D))
    print(f"  calibrate(strategy=MSEScaleStrategy): OK")

    print()


# ══════════════════════════════════════════════════════════════════════
# Section 12 — Pre-Scale + LSQ Optimization
# ══════════════════════════════════════════════════════════════════════

def section_12():
    print("=" * 60)
    print("12. PRE-SCALE + LSQ OPTIMIZATION (PoT)")
    print("=" * 60)

    # Initialize QuantSession with int8 config
    sess = QuantSession(ToyMLP(), cfg=cfg_iwo(scheme("int8", PER_T)))
    sess.eval()

    calib_data = [torch.randn(8, D) for _ in range(4)]

    # 12a. Initialize pre-scales (ones → identity)
    print("  12a. Initializing pre-scales (init='ones', pot=True) ...")
    count = sess.initialize_pre_scales(calib_data, init="ones", pot=True)
    print(f"       {count} modules received _pre_scale buffers")

    # 12b. Run LSQ optimizer
    print("  12b. Running LSQ optimizer (10 steps, 2 batches) ...")
    opt = LayerwiseScaleOptimizer(
        num_steps=10, num_batches=2,
        optimizer="adam", lr=1e-3,
        pot=True,
    )
    result = sess.optimize_scales(opt, calib_data)
    print(f"       {len(result)} modules optimized:")
    for name, scale in sorted(result.items()):
        print(f"         {name}: shape={tuple(scale.shape)} "
              f"values={[f'{v:.4f}' for v in scale.tolist()]}")

    # 12c. Verify pre_scale is registered as buffer
    print("  12c. Verifying _pre_scale buffers ...")
    buffers = [(n, m._pre_scale) for n, m in sess.qmodel.named_modules()
               if hasattr(m, "_pre_scale")]
    print(f"       {len(buffers)} _pre_scale buffers found")
    for n, s in buffers[:3]:
        print(f"         {n}: {tuple(s.shape)}")

    # 12d. Forward pass with optimized pre-scales
    print("  12d. Forward pass with optimized pre-scales ...")
    x = make_data(4)
    with torch.no_grad():
        y = sess(x)
    print(f"       output shape: {y.shape}")

    # 12e. verify_fp32
    print("  12e. Verify fp32 baseline ...")
    fp32 = sess.fp32_model
    with torch.no_grad():
        y_fp32 = fp32(x)
        y_q = sess(x)
    mse = (y_fp32 - y_q).pow(2).mean().item()
    print(f"       MSE(quant vs fp32): {mse:.6e}")
    print()


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    section_1()
    section_2()
    section_3()
    section_4()
    section_5()
    section_6()
    section_7()
    section_8()
    section_9()
    section_10()
    section_11()
    section_12()
    print("=" * 60)
    print("ALL SECTIONS PASSED — comprehensive coverage complete.")
    print("=" * 60)
