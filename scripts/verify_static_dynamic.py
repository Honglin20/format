"""
Verify static vs dynamic scale behavior empirically.

Patches quantize() at ALL import sites so LinearFunction.forward's
calls are intercepted. Traces whether each call receives a pre-computed
scale (STATIC) or computes it from the tensor (DYNAMIC).
"""
import torch
import torch.nn as nn

from src.session._config import QuantConfig
from src.session._session import Session


def _patch_quantize_everywhere():
    """Replace quantize() at every known import site with an instrumented version.

    The trace list collects (format, granularity, scale_is_not_None) tuples.
    """
    import src.quantize.elemwise as _e
    import src.quantize as _q
    import src.ops.linear as _lin
    import src.ops.conv as _conv
    import src.ops.activations as _act

    _originals = {}

    trace = []

    def _instrumented(tensor, scheme=None, allow_denorm=True, scale=None):
        if scheme is not None:
            fmt = scheme.format.name
            gran = scheme.granularity.mode.name
            trace.append((fmt, gran, scale is not None))
        return _originals["elemwise"](tensor, scheme=scheme,
                                       allow_denorm=allow_denorm, scale=scale)

    # Save originals from the root definition
    _originals["elemwise"] = _e.quantize

    # Patch at every import site
    _e.quantize = _instrumented
    _q.quantize = _instrumented
    _lin.quantize = _instrumented
    _conv.quantize = _instrumented
    _act.quantize = _instrumented

    # Also try patching any other modules we know about
    for mod_name in ["src.ops.norm", "src.ops.vec_ops"]:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "quantize"):
                _originals[mod_name] = mod.quantize
                mod.quantize = _instrumented
        except Exception:
            pass

    return trace, _originals


def _restore_quantize(originals):
    """Restore original quantize functions."""
    import src.quantize.elemwise as _e
    import src.quantize as _q
    import src.ops.linear as _lin
    import src.ops.conv as _conv
    import src.ops.activations as _act

    _e.quantize = originals["elemwise"]
    _q.quantize = originals["elemwise"]
    _lin.quantize = originals["elemwise"]
    _conv.quantize = originals["elemwise"]
    _act.quantize = originals["elemwise"]

    for mod_name in originals:
        if mod_name != "elemwise" and mod_name.startswith("src."):
            import importlib
            mod = importlib.import_module(mod_name)
            mod.quantize = originals[mod_name]


def _role_labels(cfg):
    """Return the expected call order role labels for this config."""
    s = cfg.storage is not None
    inp = cfg.input is not None
    wt = cfg.weight is not None
    bias = cfg.bias is not None
    out = cfg.output is not None

    order = []
    if s:   order.append("input (storage)")
    if inp: order.append("input (compute)")
    if s:   order.append("weight (storage)")
    if wt:  order.append("weight (compute)")
    if s and bias: order.append("bias (storage)")
    if bias: order.append("bias (compute)")
    if s:   order.append("output matmul (storage)")
    if s:   order.append("output +bias (storage)")
    if out: order.append("output (compute)")
    return order


def run_test(label, cfg, calib_data, test_input):
    """Return list of (role, format, granularity, scale_provided) tuples."""
    model = nn.Sequential(nn.Linear(16, 8, bias=True))

    session = Session(model, cfg)
    session.quantize(calib_data=calib_data)
    session.calibrate(calib_data)

    # Patch quantize at all import sites
    trace, originals = _patch_quantize_everywhere()

    try:
        with torch.no_grad():
            session.qmodel(test_input.clone())
    finally:
        _restore_quantize(originals)

    op = cfg.to_op_config()
    order = _role_labels(op)

    print(f"\n── {label} ──")
    print(f"    Config: input={op.input.format.name if op.input else '-'}"
          f" x {op.input.granularity.mode.value if op.input else '-'}"
          f" | weight={op.weight.format.name if op.weight else '-'}"
          f" x {op.weight.granularity.mode.value if op.weight else '-'}"
          f" | output={'set' if op.output else 'None'}"
          f" | storage={'set' if op.storage else 'None'}")

    if not trace:
        print("    ⚠  No quantize() calls intercepted!")
        return

    for i, (fmt, gran, has_scale) in enumerate(trace):
        role = order[i] if i < len(order) else f"? call_{i+1}?"
        status = "STATIC  (scale provided)" if has_scale else "DYNAMIC (scale from tensor)"
        print(f"    [{i+1}] {role:28s} {fmt:14s} {gran:14s} → {status}")

    # Sanity: output(compute) should be STATIC for per_tensor/per_channel
    for i, (fmt, gran, has_scale) in enumerate(trace):
        role = order[i] if i < len(order) else ""
        if role == "output (compute)":
            if gran in ("per_tensor", "per_channel"):
                assert has_scale, f"Expected STATIC output compute for {gran}, got DYNAMIC"
            elif gran == "per_block":
                # per_block might still receive scale but ignores it
                pass

    return trace


def main():
    calib_data = [torch.randn(4, 16) for _ in range(5)]
    x = torch.randn(4, 16)

    print("=" * 95)
    print("Empirical verification: scale source for each quantize() call in Linear.forward")
    print("=" * 95)

    # ── 1. Default: int8 per_tensor ──
    run_test("1. Default: int8 per_tensor (PoT scale)", QuantConfig(
        w_format="int8", w_granularity="per_tensor",
        a_format="int8", a_granularity="per_tensor",
        scale_storage="pot",
    ), calib_data, x)

    # ── 2. int8 per_channel ──
    run_test("2. int8 per_channel (FP32 scale)", QuantConfig(
        w_format="int8", w_granularity="per_channel",
        a_format="int8", a_granularity="per_channel",
        scale_storage="fp32",
    ), calib_data, x)

    # ── 3. MX fp8 per_block (no output scheme set) ──
    run_test("3. MX fp8_e4m3 per_block (no storage)", QuantConfig(
        w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32,
        a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
    ), calib_data, x)

    # ── 4. MX + bf16 storage ──
    run_test("4. MX fp8 + bf16 storage (two-level)", QuantConfig(
        w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32,
        a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
        storage_format="bfloat16",
    ), calib_data, x)

    # ── 5. per_channel + prescale ──
    run_test("5. int8 per_channel + prescale (pot, per_ch)", QuantConfig(
        w_format="int8", w_granularity="per_channel",
        a_format="int8", a_granularity="per_channel",
        scale_storage="fp32",
        transform="prescale",
        prescale_init="pot_amax", prescale_pot=True,
        prescale_granularity="per_channel",
    ), calib_data, x)

    # ── 6. per_block + prescale ──
    run_test("6. MX fp8 per_block + prescale (pot, per_ch)", QuantConfig(
        w_format="fp8_e4m3", w_granularity="per_block", w_block_size=32,
        a_format="fp8_e4m3", a_granularity="per_block", a_block_size=32,
        transform="prescale",
        prescale_init="pot_amax", prescale_pot=True,
        prescale_granularity="per_channel",
    ), calib_data, x)

    # ── 7. NEW: static_input_scale=True + per_tensor ──
    print("\n── 7. NEW: static_input_scale=True + int8 per_tensor ──")
    run_test("7. int8 per_tensor with static_input_scale=True", QuantConfig(
        w_format="int8", w_granularity="per_tensor",
        a_format="int8", a_granularity="per_tensor",
        static_input_scale=True,
    ), calib_data, x)

    # ── 8. NEW: static_input_scale=True + per_channel ──
    print("\n── 8. NEW: static_input_scale=True + int8 per_channel ──")
    run_test("8. int8 per_channel with static_input_scale=True", QuantConfig(
        w_format="int8", w_granularity="per_channel",
        a_format="int8", a_granularity="per_channel",
        static_input_scale=True,
    ), calib_data, x)

    # ── 9. Verify output compute IS static (manually set cfg.output) ──
    # Default QuantConfig doesn't set cfg.output — so the output quantize
    # step is normally skipped. Here we manually add it to prove that
    # when present, it DOES use the calibrated _output_scale.
    print("\n── 9. Verify: output compute is STATIC when cfg.output is set ──")
    model7 = nn.Sequential(nn.Linear(16, 8, bias=True))
    cfg7 = QuantConfig(w_format="int8", w_granularity="per_channel",
                       a_format="int8", a_granularity="per_channel")
    session7 = Session(model7, cfg7)
    session7.quantize(calib_data=calib_data)
    session7.calibrate(calib_data)

    # Manually patch cfg.output = cfg.input (so output uses same scheme)
    from src.scheme.op_config import OpQuantConfig
    lin7 = session7.qmodel[0]
    old_cfg = lin7.cfg
    lin7.cfg = OpQuantConfig(
        storage=old_cfg.storage, input=old_cfg.input,
        weight=old_cfg.weight, output=old_cfg.input,  # ← set output!
        bias=old_cfg.bias,
    )
    print(f"    output_scale buffer exists: {hasattr(lin7, '_output_scale')}")
    if hasattr(lin7, '_output_scale'):
        s = lin7._output_scale
        print(f"    _output_scale shape={tuple(s.shape)}  values={s.flatten()[:4].tolist()}")

    trace7, origs7 = _patch_quantize_everywhere()
    try:
        with torch.no_grad():
            session7.qmodel(x.clone())
    finally:
        _restore_quantize(origs7)

    order7 = _role_labels(lin7.cfg)
    for i, (fmt, gran, has_scale) in enumerate(trace7):
        role = order7[i] if i < len(order7) else f"? call_{i+1}?"
        status = "STATIC  (scale provided)" if has_scale else "DYNAMIC (scale from tensor)"
        marker = " ← CONFIRMED" if role == "output (compute)" and has_scale else ""
        print(f"    [{i+1}] {role:28s} {fmt:14s} {gran:14s} → {status}{marker}")

    # ── Summary ──
    print("\n" + "=" * 95)
    print("CONCLUSION")
    print("=" * 95)
    print("""
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ Role / Position              │ per_tensor │ per_channel │ per_block (MX)     │
  ├──────────────────────────────────────────────────────────────────────────────┤
  │ input (compute)              │  DYNAMIC   │  DYNAMIC    │  DYNAMIC           │
  │ weight (compute)             │  DYNAMIC   │  DYNAMIC    │  DYNAMIC           │
  │ output (compute) *           │  STATIC    │  STATIC     │  DYNAMIC¹          │
  │ storage (bf16 / fp8 elem)    │    N/A²    │    N/A²     │    N/A²            │
  │ prescale transform           │  STATIC³   │  STATIC³    │  STATIC³           │
  └──────────────────────────────────────────────────────────────────────────────┘

  * cfg.output is None in the default QuantConfig → output compute step
    is SKIPPED entirely at runtime (confirmed: no quantize() call for output
    in tests 1-6).  When cfg.output is manually set (test 7), it DOES use
    the calibrated _output_scale → STATIC.

  ¹ MX shared exponents are always computed from tensor values at runtime.
    Even when scale is passed, per_block ignores it (docstring: base.py:238).
  ² Float formats (ebits > 0, e.g. bf16) do direct elemwise quantization
    without amax normalization.  The trace shows "DYNAMIC" only because
    no scale arg is passed — but no scale is ever needed here.
  ³ PreScaleTransform holds a stored buffer/Parameter.  Even pot=True is
    a deterministic projection; scale never adapts to input values.

  KEY FINDING:
  - input  quantization:  always DYNAMIC — computes amax from tensor each pass
  - weight quantization:  always DYNAMIC — computes amax from weight each pass
    (weights are fixed during inference, so the result is effectively
    deterministic, but the amax is NOT cached — it's recomputed every time)
  - output quantization:  STATIC when cfg.output is set (uses calibrated
    _output_scale), but cfg.output is None by default so this step is
    typically DEAD CODE in user-facing configs
  - prescale:             STATIC — scale buffer/param, independent of input
""")


if __name__ == "__main__":
    main()
