"""
02 — QuantSession Unified Workflow: calibrate → analyze → compare → export.

Demonstrates the full pipeline through a single QuantSession object.
Run:  PYTHONPATH=. python examples/02_session_workflow.py
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from _model import ToyMLP
from src.formats.base import FormatBase
from src.scheme.quant_scheme import QuantScheme
from src.scheme.granularity import GranularitySpec
from src.scheme.op_config import OpQuantConfig
from src.session import QuantSession
from src.analysis.observers import QSNRObserver, MSEObserver


def make_cfg():
    fmt = FormatBase.from_str("int8")
    scheme = QuantScheme(format=fmt, granularity=GranularitySpec.per_tensor())
    return OpQuantConfig(input=(scheme,), weight=(scheme,), output=(scheme,))


def make_calib_loader(n_batches=8, batch_size=4):
    data = torch.randn(n_batches * batch_size, 128)
    return DataLoader(TensorDataset(data), batch_size=batch_size)


def make_eval_loader(n_batches=8, batch_size=4):
    data = torch.randn(n_batches * batch_size, 128)
    labels = torch.randint(0, 2, (n_batches * batch_size,))
    return DataLoader(TensorDataset(data, labels), batch_size=batch_size)


def main():
    print("=" * 55)
    print("QuantSession Unified Workflow")
    print("=" * 55)

    # ── 1. Create session ─────────────────────────────────────────
    print("\n1. Creating QuantSession ...")
    model = ToyMLP()
    session = QuantSession(
        model, make_cfg(),
        observers=[QSNRObserver(), MSEObserver()],
    )
    session.eval()
    print(f"   mode: {session.mode}")
    print(f"   fp32_model kept: {session.fp32_model is not None}")

    # ── 2. Calibrate ──────────────────────────────────────────────
    print("\n2. Calibrating (scales auto-assigned on exit) ...")
    with session.calibrate():
        for batch, in make_calib_loader():
            session(batch)

    has_scale = any(hasattr(m, "_output_scale") for m in session.qmodel.modules())
    print(f"   scales assigned: {has_scale}")

    # ── 3. Analyze ────────────────────────────────────────────────
    print("\n3. Layer-wise error analysis ...")
    with session.analyze() as ctx:
        for i, (batch,) in enumerate(make_calib_loader()):
            session(batch)

    report = ctx.report()
    for layer_name in sorted(report.keys()):
        layer_data = report._raw[layer_name]
        for role in layer_data:
            for stage in layer_data[role]:
                for sk, metrics in layer_data[role][stage].items():
                    qsnr = metrics.get("qsnr_db", "N/A")
                    mse = metrics.get("mse", "N/A")
                    print(f"   {layer_name:<20} {role:<10} QSNR={qsnr:>8.1f} dB  MSE={mse:>12.6f}"
                          if isinstance(qsnr, float) else
                          f"   {layer_name:<20} {role:<10} QSNR={'N/A':>8}     MSE={'N/A':>12}")

    # ── 4. Compare ────────────────────────────────────────────────
    print("\n4. End-to-end accuracy comparison ...")
    dl = make_eval_loader()

    # 4a — auto mode
    result = session.compare(dl)
    print(f"   auto mode:")
    print(f"     fp32 accuracy:  {result['fp32'].get('accuracy', 'N/A'):.4f}")
    print(f"     quant accuracy: {result['quant'].get('accuracy', 'N/A'):.4f}")
    print(f"     delta:          {result['delta'].get('accuracy', 'N/A'):.4f}")

    # 4b — manual mode (custom metric)
    session.use_quant()
    cmp = session.comparator()
    with cmp, torch.no_grad():
        for inputs, labels in dl:
            session.use_fp32()
            fp32_out = session(inputs)
            session.use_quant()
            q_out = session(inputs)
            cmp.record(fp32_out, q_out, labels)

    result2 = cmp.evaluate(
        lambda logits, labels: {
            "top1": (logits.argmax(-1) == labels).float().mean().item()
        },
        directions={"top1": "higher"},
    )
    print(f"   manual mode (custom top1):")
    print(f"     fp32:  {result2['fp32']['top1']:.4f}")
    print(f"     quant: {result2['quant']['top1']:.4f}")
    print(f"     delta: {result2['delta']['top1']:+.4f}")

    # ── 5. Compare multiple sessions ──────────────────────────────
    print("\n5. Comparing multiple quantization configs ...")
    from src.analysis.e2e import compare_sessions
    from src.calibration.strategies import PercentileScaleStrategy

    # Create a second session with different calibration
    s2 = QuantSession(
        ToyMLP(), make_cfg(),
        calibrator=PercentileScaleStrategy(q=99.0),
    )
    s2.eval()
    s2.qmodel.load_state_dict(session.qmodel.state_dict(), strict=False)
    with s2.calibrate():
        for batch, in make_calib_loader():
            s2(batch)

    results = compare_sessions({"max": session, "p99": s2}, dl)
    print(f"   fp32 baseline accuracy: {results['fp32']['accuracy']:.4f}")
    for name in ["max", "p99"]:
        acc = results[name]["quant"]["accuracy"]
        delta = results[name]["delta"]["accuracy"]
        print(f"   {name}: accuracy={acc:.4f}  delta={delta:+.4f}")

    # ── 6. Clear scales and re-infer ──────────────────────────────
    print("\n6. Clear scales & verify model still runs ...")
    removed = session.clear_scales()
    print(f"   removed scales from {len(removed)} modules")
    with torch.no_grad():
        y = session(torch.randn(2, 128))
    print(f"   output shape after clear: {y.shape}")
    print(f"   OK — model still runs without pre-computed scales")

    print("\n" + "=" * 55)
    print("QuantSession workflow complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
